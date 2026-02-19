import os
import json
import time
import random
import asyncio
import threading
from google.genai.types import Part
from collections import defaultdict
from typing import List, Any, Literal, Dict
from utils.core.fuzzy_search import find_exact_text_match
from tools.contract.prompts_contract import ANALYZER_SYS, REVIEW_SYS
from utils.core.log import pid_tool_logger, set_logger, get_logger

from utils.core.errors import _make_error_payload
from utils.llm.LLM import call_llm_async_cache, count_tokens, RateLimiter
from utils.core.slack import SlackActivityLogger, SlackActivityMeta, fmt_dur
from utils.core.jsonval import validate_and_correct_response, validate_source_response
from utils.storage.bucket import (
    get_file,
    upload_file,
    delete_file,
    serve_precomputed_or_recompute,
)

from tools.contract.prompts_contract import (
    prompts_NEC,
    prompts_FIDIC,
    high_risk_clauses_review,
    ambiguous_clauses_review,
    duplicate_clauses_review,
    contradictory_clauses_review,
    consolidate_responses,
    modified_clauses_review,
    retrieve_clause_details,
)

from utils.llm.context_cache import (
    cache_enabled,
    create_cache,
    delete_cache,
    make_composite_contents,
    _min_cache_tokens,
    count_cached_tokens,
)

from utils.storage.gcs import spill_media_and_text_to_gcs, delete_gs_prefix


def build_cached_context_parts(contract_parts: list[Part]) -> list[Part]:
    return (
        [Part.from_text(text="### BEGIN CONTRACT-DOCS\n")]
        + list(contract_parts or [])
        + [Part.from_text(text="### END CONTRACT-DOCS\n")]
    )


class ContractReview:
    CFG: dict[str, Any] = {
        "temperature": 0,
        "top_p": 0.95,
        "max_output_tokens": 20000,
        "thinking_config": {"thinking_budget": 0},
        "response_mime_type": "application/json",
    }

    def __init__(
        self,
        user_type: str,
        package_id: str,
        document_parts: List[Part],
        tokenized_text: Dict,
        model_name: str = "gemini-2.5-flash",
        cached_content_name: str | None = None,
    ):

        set_logger(pid_tool_logger(package_id, "contract_review"))
        self.logger = get_logger()
        self.logger.debug("ContractReview initialized")
        self.user_type = user_type.lower()
        self.model_name = model_name
        self.package_id = package_id
        self.cached_content_name = cached_content_name

        self.document_parts = document_parts

        # get page number from tokenized version of document
        raw_tok = tokenized_text.get("tokenized_content", tokenized_text)

        pages: list[dict] = []
        if isinstance(raw_tok, dict):
            for src, lst in raw_tok.items():
                if isinstance(lst, list):
                    for item in lst:
                        if isinstance(item, dict):
                            item.setdefault("document_name", os.path.basename(src))
                            pages.append(item)
        elif isinstance(raw_tok, list):
            pages = raw_tok

        self._tokenized_pages = pages

        by_doc: dict[str, list[dict]] = defaultdict(list)
        for it in self._tokenized_pages:
            doc = it.get("document_name") or "unknown"
            norm = dict(it)
            norm["document_name"] = doc

            by_doc[doc].append(norm)

        self._tokenized_by_doc = dict(by_doc)

        if not self.document_parts:
            raise ValueError("[Review] document_parts cannot be empty.")

        if not self._tokenized_by_doc:
            self.logger.warning(
                "[Review] tokenized pages are empty; source finding will fail."
            )

        self.logger.debug(
            f"[Init] Loaded tokenized pages — {len(self._tokenized_pages)} pages"
        )

        self.doc_token_count = max(0, count_tokens(model_name, self.document_parts))
        self.logger.debug(
            f"[Review] {len(self.document_parts)} Parts ready "
            f"({self.doc_token_count} tokens)"
        )
        self._hits = 0
        self._misses = 0
        self._stat_lock = asyncio.Lock()

    # LLM helpers
    async def _ask_llm(self, prompt: str, cfg: dict | None = None) -> str:
        """One async LLM call (shared client pool)."""
        use_cache = bool(self.cached_content_name) and cache_enabled()
        parts = (
            [Part.from_text(text=prompt)]
            if use_cache
            else (self.document_parts + [Part.from_text(text=prompt)])
        )

        return await call_llm_async_cache(
            parts,
            model=self.model_name,
            system_instruction=None if use_cache else REVIEW_SYS,
            cfg=cfg or self.CFG,
            cached_content=self.cached_content_name if use_cache else None,
        )

    async def _retry_with_temp(self, prompt: str) -> dict | None:
        """Secondary attempt with higher temperature."""
        temp_cfg = self.CFG | {"temperature": random.uniform(0.3, 0.6)}
        try:
            raw = await self._ask_llm(prompt, cfg=temp_cfg)
            ok, corr = validate_source_response(raw)
            return corr if ok else None
        except Exception:
            return None

    # generic fan-out runner
    async def _fanout_prompts(
        self,
        prompts: list[str],
        *,
        answer_type: str,
        concurrency: int = 15,
        limiter: RateLimiter | None = None,
    ) -> List[dict]:
        limiter = limiter or RateLimiter()
        sem = asyncio.Semaphore(concurrency)

        async def worker(p: str) -> dict | None:
            async with sem:
                await limiter.acquire(
                    0 if self.cached_content_name else self.doc_token_count
                )
                try:
                    raw = await self._ask_llm(p)
                    ok, fixed = validate_and_correct_response(raw)
                    if ok:
                        fixed["type"] = answer_type
                        return fixed
                    self.logger.warning(f"un-fixable response: {p[:40]}...")
                except Exception as e:
                    self.logger.error(f"prompt failed: {e}")
            return None

        return [r for r in await asyncio.gather(*(worker(p) for p in prompts)) if r]

    # clause extraction (with fuzzy match)
    async def _process_clause(
        self,
        clause_id: str,
        *,
        limiter: RateLimiter,
        sem: asyncio.Semaphore,
    ) -> dict | None:
        prompt = retrieve_clause_details(clause_id)
        async with sem:
            await limiter.acquire(
                0 if self.cached_content_name else self.doc_token_count
            )

            raw = await self._ask_llm(prompt)
            ok, data = validate_source_response(raw)
            if not ok:
                data = await self._retry_with_temp(prompt)
                if not data:
                    return None

            # fuzzy locate in text
            match = find_exact_text_match(data["clauseText"], self._tokenized_by_doc)

            if match:
                data.update(
                    page=match["page_number"],
                    sourceDocumentName=match["document_name"],
                    exactText=match["exact_text"],
                )
                async with self._stat_lock:
                    self._hits += 1
            else:
                data.update(
                    page="Not Found",
                    sourceDocumentName="Not Found",
                    exactText="Not Found",
                )
                async with self._stat_lock:
                    self._misses += 1
            return data

    async def _extract_clauses_async(self, clause_ids: list[str]) -> list[dict]:
        limiter = RateLimiter()
        sem = asyncio.Semaphore(12)
        tasks = [
            self._process_clause(cid, limiter=limiter, sem=sem) for cid in clause_ids
        ]
        results = [r for r in await asyncio.gather(*tasks) if r]
        self.logger.debug(f"[extract] hits={self._hits} misses={self._misses}")
        return results

    # high-level public workflow
    async def run_reviews_async(self, analysis: dict) -> dict:
        """Main entry - returns {'reviews': ...}."""
        if not isinstance(analysis, dict):
            self.logger.error(
                f"Analysis is not a dict (got {type(analysis).__name__}); skipping review."
            )
            return {"reviews": []}

        if self.user_type != "client":
            self.logger.warning(
                f"ContractReview running in non-client mode: '{self.user_type}' - review steps will be skipped."
            )
            return {"reviews": []}

        out: list[dict] = []

        batches = [
            ("high_risk", high_risk_clauses_review(analysis)),
            ("contradiction", contradictory_clauses_review(analysis)),
            ("ambiguous", ambiguous_clauses_review(analysis)),
            ("duplicate", duplicate_clauses_review(analysis)),
            ("modified", modified_clauses_review(analysis)),
        ]
        for typ, prompts in batches:
            self.logger.debug(f"[Review] {typ} ... {len(prompts)} prompts")
            rsp = await self._fanout_prompts(prompts, answer_type=typ)
            out.extend(consolidate_responses(rsp, typ))

        # clause-source extraction
        clause_ids = self._gather_clause_ids(analysis)
        clause_data = await self._extract_clauses_async(clause_ids)
        out = self._merge_sources(out, clause_data)

        return {"reviews": out}

    # small helpers
    @staticmethod
    def _merge_sources(reviews: list[dict], sources: list[dict]) -> list[dict]:
        idx = {c["clauseNumber"]: c for c in sources}
        for r in reviews:
            r["mainClause"]["source"] = idx.get(r["mainClause"]["id"])
            for cl in r.get("clauses", []):
                cl["source"] = idx.get(cl["id"])
        return reviews

    @staticmethod
    def _gather_clause_ids(data: dict) -> List[str]:
        if not isinstance(data, dict):
            return []
        ids = set()
        ids.update(
            c.get("clauseIdentifier", "")
            for c in data.get("highRiskClauses", {}).get("clauses", [])
        )
        for main, others in data.get("contradictions", {}).items():
            ids.add(main)
            ids.update(others)
        ids.update(data.get("ambiguousClauses", {}).get("ambiguousClausesList", []))
        for grp in data.get("duplicateClauses", []):
            ids.update(grp)
        ids.update(
            data.get("modifiedStandardClauses", {}).get("listOfModifiedClauses", [])
        )
        return list(filter(None, ids))


# MAIN FOR CONTRACT TOOL
async def contract_review_main(
    package_id: str = None,
    company_id: str = None,
    user_name: str = None,
    contract_type: str = "NEC",
    user_type: str = "client",
    country_code: str = None,
    request_method: str = None,
    payment_option: str = "A",
    governed_law: str = "USA",
    remote_ip: str | None = None,
    compute_reanalysis: bool = True,
    user_id: str | None = None,
    package_name: str | None = None,
) -> dict:
    """
    Handle Contract Analysis and Review (GET/POST).
    Sequentially runs ContractAnalyzer and ContractReview.
    """
    country_code = country_code or "USA"
    base_logger = pid_tool_logger(package_id=package_id, tool_name="contract_review")
    set_logger(
        base_logger,
        tool_name="contract_review_main",
        tool_base="CONTRACT",
        package_id=package_id or "unknown",
        ip_address=remote_ip or "no_ip",
        request_type=request_method or "N/A",
    )
    logger = get_logger()

    try:
        if not package_id:
            return {"error": "projectId is required", "status": "error"}

        contract_type = (contract_type or "NEC").upper()

        # Decide which prompts to use
        if contract_type.startswith("NEC"):
            prompt_set = prompts_NEC
        elif contract_type == "FIDIC":
            prompt_set = prompts_FIDIC
        else:
            return {
                "error": f"Unsupported contract type: {contract_type}",
                "status": "error",
            }

        # GET request
        if request_method == "GET":
            return await handle_contract_get_request(package_id, company_id)

        # POST request
        elif request_method == "POST":
            overview_path = "data/contract_overview.json"
            recommendation_path = "data/contract_recommendation.json"

            async def start_compute():
                return await handle_contract_post_request(
                    package_id,
                    company_id,
                    user_type,
                    prompt_set,
                    remote_ip,
                    user_name,
                    user_id,
                    package_name,
                )

            return await serve_precomputed_or_recompute(
                package_id=package_id,
                compute_reanalysis=compute_reanalysis,
                start_compute=start_compute,
                result_path=[overview_path, recommendation_path],
                names=["overview", "recommendation"],
                logger=logger,
                miss_policy="error",
                company_id=company_id,
            )

        else:
            return {
                "error": f"Unsupported request method: {request_method}",
                "status": "error",
            }

    except Exception as e:
        logger.exception(f"Error in contract_review_main: {e}")
        return {"error": "Internal server error", "details": str(e), "status": "error"}


async def handle_contract_get_request(package_id: str, company_id: str) -> dict:
    """
    Handle GET requests to fetch contract analysis and review results.
    """
    logger = get_logger()
    try:
        overview = get_file(package_id, "data/contract_overview.json", company_id)
        recommendation = get_file(
            package_id, "data/contract_recommendation.json", company_id
        )

        if overview is None or recommendation is None:
            return {"status": "in progress"}

        return {
            "status": "completed",
            "overview": overview,
            "recommendation": recommendation,
        }

    except Exception as e:
        logger.exception(f"Error retrieving results: {e}")
        return {
            "error": "Failed to retrieve results",
            "details": str(e),
            "status": "error",
        }


async def handle_contract_post_request(
    package_id: str,
    company_id: str,
    user_type: str,
    prompt_set: list,
    remote_ip: str | None = None,
    user_name: str | None = None,
    user_id: str | None = None,
    package_name: str | None = None,
) -> dict:
    """
    Handle POST requests to initiate contract analysis + review.
    Runs ContractAnalyzer first, then ContractReview sequentially in a background thread.
    """
    logger = get_logger()
    act = SlackActivityLogger(
        SlackActivityMeta(
            package_id=package_id, tool="CONTRACT", user=user_name, company=company_id
        )
    )
    parent_ts = act.start()
    act.sub(f"📑 Contract workflow started.")

    for obj in ["data/contract_overview.json", "data/contract_recommendation.json"]:
        try:
            delete_file(package_id, obj, company_id)
        except Exception as exc:
            logger.warning(f"Could not delete {obj}: {exc}")

    def background_process(parent_ip: str | None, thread_ts: str):
        try:
            worker_base = pid_tool_logger(package_id, "contract")
            set_logger(
                worker_base,
                tool_name="contract_worker",
                tool_base="CONTRACT",
                package_id=package_id or "unknown",
                ip_address=parent_ip or "no_ip",
                request_type="WORKER",
            )
            log = get_logger()
            wact = SlackActivityLogger(
                SlackActivityMeta(
                    package_id=package_id,
                    tool="CONTRACT",
                    user=user_name,
                    company=company_id,
                ),
                thread_ts=thread_ts,
            )
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                run_contract_analysis_and_review(
                    package_id,
                    company_id,
                    user_type,
                    prompt_set,
                    log,
                    wact,
                    user_id,
                    package_name,
                )
            )
            loop.close()
        except Exception as e:
            log.exception(f"Background processing error: {e}")
            SlackActivityLogger(
                SlackActivityMeta(
                    package_id=package_id,
                    tool="CONTRACT",
                    user=user_name,
                    company=company_id,
                ),
                thread_ts=thread_ts,
            ).error(f"Background processing error — {e}")
            _upload_error_artifacts(
                package_id=package_id,
                company_id=company_id,
                stage="thread",
                err=e,
            )

    # Start background thread
    thread = threading.Thread(
        target=background_process, args=(remote_ip, parent_ts), daemon=True
    )
    thread.start()

    return {
        "status": "in progress",
        "message": f"Contract analysis and review started for project {package_id}",
    }


def _upload_error_artifacts(
    *, package_id: str, company_id: str, stage: str, err: Exception | str
):
    payload = _make_error_payload(stage, err)
    # Always upload BOTH files so GET becomes "completed" with errors
    upload_file(
        package_id,
        "data/contract_overview.json",
        json_string=json.dumps(payload, indent=2),
        company_id=company_id,
    )
    upload_file(
        package_id,
        "data/contract_recommendation.json",
        json_string=json.dumps(payload, indent=2),
        company_id=company_id,
    )


async def run_contract_analysis_and_review(
    package_id: str,
    company_id: str,
    user_type: str,
    prompt_set: list,
    logger,
    act: SlackActivityLogger,
    user_id: str | None = None,
    package_name: str | None = None,
):
    """
    Sequentially run ContractAnalyzer and ContractReview, save results,
    and upload them to MinIO storage.
    """
    from utils.storage.bucket import list_files
    from tools.contract.contract_analyzer import ContractAnalyzer

    start_t = time.perf_counter()
    an_cache: str | None = None
    rev_cache: str | None = None
    tmp_prefix_cr: str | None = None

    try:
        # STEP 1 Contract Analysis
        step_t = time.perf_counter()
        analyzer = ContractAnalyzer(package_id=package_id, company_id=company_id)

        contract_source_dir = "contracts/"
        contract_files = list_files(package_id, contract_source_dir, company_id)

        if not contract_files:
            raise FileNotFoundError(
                "No contract files found in the 'contracts/' directory."
            )

        document_metas = [
            {"name": os.path.basename(key), "source_key": key} for key in contract_files
        ]

        await analyzer.add_client_data(document_metas)
        # If caching is off or fails, keep non-cache requests small as well
        if not cache_enabled() or not analyzer.cached_content_name:
            try:
                sp, pref = spill_media_and_text_to_gcs(
                    analyzer.document_parts,
                    prefix=f"tmp/{package_id}/contracts-nocache",
                    force_spill_all_text=False,  # only if needed
                )
                logger.debug(
                    "cache payload prepared via spill; inline body kept tiny; prefix=%s",
                    pref,
                )

                analyzer.document_parts = sp
                analyzer._tmp_prefix = analyzer._tmp_prefix or pref
            except Exception:
                pass

        if cache_enabled():
            try:
                # only cache if the doc context is "big enough"
                pre = make_composite_contents(analyzer.document_parts)
                total = count_cached_tokens(analyzer.model_name, pre)
                min_needed = _min_cache_tokens(analyzer.model_name)

                if total >= min_needed:
                    # Wrap in explicit bucket markers
                    cache_parts = (
                        [Part.from_text(text="### BEGIN CONTRACT-DOCS\n")]
                        + list(analyzer.document_parts or [])
                        + [Part.from_text(text="### END CONTRACT-DOCS\n")]
                    )

                    # Keep the request body tiny: spill media + text to two files
                    tmp_prefix_cr = f"tmp/{package_id}/contracts-cr"  # separate prefix for these caches
                    cache_parts, _ = spill_media_and_text_to_gcs(
                        cache_parts,
                        prefix=tmp_prefix_cr,
                        force_spill_all_text=True,
                    )
                    logger.debug(
                        "cache payload prepared via spill; inline body kept tiny; prefix=%s",
                        tmp_prefix_cr,
                    )

                    contents = make_composite_contents(cache_parts)

                    # Create two caches with different system instructions (analyzer and reviewer)
                    an_cache = create_cache(
                        model=analyzer.model_name,
                        contents=contents,
                        display=f"{package_id}:contracts:analyze",
                        system_instruction=ANALYZER_SYS,
                    )
                    rev_cache = create_cache(
                        model=analyzer.model_name,
                        contents=contents,
                        display=f"{package_id}:contracts:review",
                        system_instruction=REVIEW_SYS,
                    )

                    if not (an_cache and rev_cache):
                        # If either failed, nuke both and purge the whole tmp prefix
                        if an_cache:
                            delete_cache(an_cache)
                        if rev_cache:
                            delete_cache(rev_cache)
                        an_cache = rev_cache = None
                        try:
                            delete_gs_prefix(prefix=tmp_prefix_cr)
                        except Exception:
                            pass
                else:
                    logger.debug(f"cache skip: total tokens {total} < {min_needed}")
            except Exception as e:
                logger.warning(f"cache build failed; running non-cached. {e}")

        # Wire analyzer to the paired cache; delete any prior cache name to avoid leaks
        if analyzer.cached_content_name and analyzer.cached_content_name != an_cache:
            delete_cache(analyzer.cached_content_name)
        analyzer.cached_content_name = an_cache

        # analyzer phase
        analysis_result = await analyzer.analyze_contract_async(list(prompt_set))
        upload_file(
            package_id,
            "data/contract_overview.json",
            json_string=json.dumps(analysis_result, indent=2),
            company_id=company_id,
        )

        overview_dur = time.perf_counter() - step_t
        act.sub(f"📝 Contract overview done in {fmt_dur(overview_dur)}")

        # reset timer
        step_t = time.perf_counter()
        # STEP 2 Contract Review
        reviewer = ContractReview(
            user_type=user_type,
            package_id=package_id,
            document_parts=analyzer.document_parts,
            tokenized_text=analyzer.tokenized_text,
            model_name=analyzer.model_name,
            cached_content_name=rev_cache,
        )

        review_result = await reviewer.run_reviews_async(analysis_result)
        recommendation = review_result["reviews"]

        upload_file(
            package_id=package_id,
            location="data/contract_recommendation.json",
            json_string=json.dumps(recommendation, indent=2),
            company_id=company_id,
        )

        review_dur = time.perf_counter() - step_t
        act.sub(f"🔍 Contract review done in {fmt_dur(review_dur)}")

        total_dur = fmt_dur(time.perf_counter() - start_t)
        logger.info(f"Contract overview + review completed in {total_dur}")
        act.sub(f"✅ Contract workflow finished in {total_dur}")
        act.done()

        return True

    except Exception as e:
        total_dur = fmt_dur(time.perf_counter() - start_t)
        logger.exception(
            f"Error during contract analysis/review after {total_dur}: {e}"
        )
        act.error(f"🔥 Contract workflow FAILED after {total_dur} — {e}")
        _upload_error_artifacts(
            package_id=package_id,
            company_id=company_id,
            stage="worker",
            err=e,
        )
        return False

    finally:
        # Cleanup caches + temp uploads (if any)
        try:
            if an_cache:
                delete_cache(an_cache)
            if rev_cache:
                delete_cache(rev_cache)
            # Prefer bulk delete by prefix(s)
            if getattr(analyzer, "_tmp_prefix", None):
                try:
                    delete_gs_prefix(prefix=analyzer._tmp_prefix)
                except Exception:
                    pass
        except Exception as clean_e:
            logger.warning(f"cache cleanup issue: {clean_e}")


def main() -> bool:
    """
    Lightweight self-test for ContractReview:
      - In-memory document + tokenized pages (DICT, not list)
      - Stubs fanout/extraction to avoid LLM/I-O
      - Prints OK/FAIL and returns a boolean
    """
    import asyncio

    print("CONTRACT_REVIEW TEST START")
    try:
        # Make token counting a no-op
        globals()["count_tokens"] = lambda *a, **k: 0
        # Keep consolidate_responses trivial
        globals()["consolidate_responses"] = lambda rsp, typ: rsp

        # Stub internals so no LLM/network is used
        async def _fake_fanout(self, prompts, *, answer_type, **_):
            out = []
            for i, _ in enumerate(prompts, 1):
                cid = f"{answer_type}_{i}"
                out.append(
                    {
                        "type": answer_type,
                        "mainClause": {
                            "id": cid,
                            "clauseIdentifier": cid,
                            "keywords": [answer_type],
                            "description": f"{answer_type} clause",
                        },
                        "clauses": [],
                    }
                )
            return out

        async def _fake_extract(self, clause_ids):
            return [
                {
                    "clauseNumber": cid,
                    "clauseText": f"Source text for {cid}",
                    "page": 1,
                    "sourceDocumentName": "dummy_contract.txt",
                    "exactText": f"Exact match for {cid}",
                }
                for cid in clause_ids
            ]

        ContractReview._fanout_prompts = _fake_fanout
        ContractReview._extract_clauses_async = _fake_extract

        # Minimal in-memory doc + dict-shaped tokenized text
        doc_parts = [
            Part.from_text(
                text="Dummy contract with Clause 1.1, 2.1, 3.4, 5.1, 5.2, 6.1."
            )
        ]
        tokenized_text = {
            "dummy_contract.txt": [
                {
                    "page_number": 1,
                    "document_name": "dummy_contract.txt",
                    "text": "Clause 1.1 The Contractor shall …",
                }
            ]
        }

        reviewer = ContractReview(
            user_type="client",
            package_id="SELFTEST",
            document_parts=doc_parts,
            tokenized_text=tokenized_text,
        )

        # Minimal analysis payload to exercise all review branches
        analysis = {
            "highRiskClauses": {"clauses": [{"clauseIdentifier": "1.1"}]},
            "contradictions": {"2.1": ["3.1"]},
            "ambiguousClauses": {"ambiguousClausesList": ["3.4"]},
            "duplicateClauses": [["5.1", "5.2"]],
            "modifiedStandardClauses": {"listOfModifiedClauses": ["6.1"]},
        }

        result = asyncio.run(reviewer.run_reviews_async(analysis))
        ok = isinstance(result, dict) and isinstance(result.get("reviews"), list)
        if ok:
            print("CONTRACT_REVIEW OK")
            return True
        else:
            print("CONTRACT_REVIEW FAIL: unexpected result shape")
            return False

    except Exception as e:
        print(f"CONTRACT_REVIEW ERROR: {e}")
        return False


if __name__ == "__main__":
    main()
