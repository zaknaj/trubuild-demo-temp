import time
import json
import asyncio
from typing import List, Dict
from google.genai.types import Part
from utils.document.docingest import get_documents
from utils.core.jsonval import parse_json_strings
from tools.contract.prompts_contract import ANALYZER_SYS
from utils.core.log import pid_tool_logger, set_logger, get_logger
from utils.llm.LLM import call_llm_async_cache, count_tokens, RateLimiter
from utils.llm.context_cache import cache_enabled

"""
pip install -U google-genai
"""


class ContractAnalyzer:
    def __init__(
        self,
        package_id,
        company_id,
        model_name="gemini-2.5-flash",
    ):
        set_logger(pid_tool_logger(package_id, "contract_analyzer"))
        self.logger = get_logger()
        self.model_name = model_name
        self.package_id = package_id
        self.company_id = company_id
        self.cached_content_name: str | None = None
        self._tmp_prefix: str | None = None

        self.logger.debug("ContractAnalyzer initialized")
        self.document_parts = []
        self.tokenized_text = {}

    async def add_client_data(self, document_metas: List[Dict[str, str]]):
        """
        Loads processed document data using the central document_loader.

        Args:
            document_metas: A list of dicts, each with 'name' and 'source_key'.
        """
        if not document_metas:
            raise ValueError("No documents provided for analysis.")

        self.logger.debug(f"Loading data for {len(document_metas)} documents...")

        # The single call to get all processed data
        self.document_parts, meta = await get_documents(
            package_id=self.package_id,
            company_id=self.company_id,
            document_metas=document_metas,
        )

        self.tokenized_text = meta.get("tokenized_content", {})
        if not self.document_parts:
            raise ValueError("Failed to load any usable document content.")

        tokens = count_tokens(self.model_name, self.document_parts)
        self.logger.debug(
            f"Prepared {len(self.document_parts)} Parts ({tokens} tokens)"
        )

    async def _ask_llm(self, prompt: str) -> str:
        cfg = {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1.0,
            "max_output_tokens": 20000,
            "thinking_config": {"thinking_budget": 0},
            "response_mime_type": "application/json",
            "seed": 12345,
        }

        use_cache = bool(self.cached_content_name) and cache_enabled()
        parts = (
            [Part.from_text(text=prompt)]
            if use_cache
            else (self.document_parts + [Part.from_text(text=prompt)])
        )

        return await call_llm_async_cache(
            parts,
            model=self.model_name,
            system_instruction=None if use_cache else ANALYZER_SYS,
            cfg=cfg,
            cached_content=self.cached_content_name if use_cache else None,
        )

    async def analyze_contract_async(
        self,
        prompt_list: list[str],
        *,
        concurrency: int = 10,
        limiter: RateLimiter | None = None,
    ) -> dict:
        logger = get_logger()
        if not getattr(self, "document_parts", None):
            raise RuntimeError("Call add_client_data first")

        limiter = limiter or RateLimiter()
        sem = asyncio.Semaphore(concurrency)
        doc_token_count = count_tokens(self.model_name, self.document_parts)
        if doc_token_count < 0:
            doc_token_count = 0

        # If cached, each prompt carries only a tiny question; skip counting the whole doc
        if self.cached_content_name:
            doc_token_count = 0

        async def work(prompt: str) -> str | None:
            async with sem:
                try:
                    await limiter.acquire(doc_token_count)
                    return await self._ask_llm(prompt)
                except Exception as e:
                    self.logger.exception(f"Prompt failed: {prompt[:40]} - {e}")
                    return None

        t0 = time.perf_counter()
        raw = await asyncio.gather(*(work(p) for p in prompt_list))
        logger.debug(f"Raw from LLM: {raw}")
        cleaned = parse_json_strings([r for r in raw if r])
        logger.debug(f"Cleaned: {cleaned}")
        result = self._post_process(cleaned)
        logger.debug(f"after post processing: {result}")
        self.logger.debug(f"Contract analysed in {time.perf_counter() - t0:.2f}s")
        return result

    def _post_process(self, cleaned_responses: list) -> dict:
        try:

            def _unwrap_singleton_list_dict(x):
                if isinstance(x, list) and len(x) == 1 and isinstance(x[0], dict):
                    return x[0]
                return x

            json_data: dict = {}
            for item in cleaned_responses:
                item = _unwrap_singleton_list_dict(item)
                if isinstance(item, dict):
                    json_data.update(item)
                else:
                    self.logger.warning(
                        f"Skipping non-dict top-level item: {type(item).__name__}"
                    )

            # Fuse payment terms, but only if p1/p2 are dicts
            p1 = json_data.pop("paymentTerms1", {})
            p2 = json_data.pop("paymentTerms2", {})
            if not isinstance(p1, dict):
                p1 = {}
            if not isinstance(p2, dict):
                p2 = {}

            if p1 or p2:
                merged = {**p1, **{k: v for k, v in p2.items() if k != "source"}}
                srcs = [s for s in (p1.get("source"), p2.get("source")) if s]
                if srcs:
                    merged["Source"] = ", ".join(srcs)
                json_data["paymentTerms"] = merged
            else:
                self.logger.warning("no paymentTerms blocks returned by LLM")

            # 3) Normalize possibly-wrong shapes from the LLM
            def _norm_obj(val, list_key):
                if isinstance(val, dict):
                    return val
                if isinstance(val, list):
                    return {list_key: val}
                return {list_key: []}

            json_data["highRiskClauses"] = _norm_obj(
                json_data.get("highRiskClauses"), "clauses"
            )
            json_data["ambiguousClauses"] = _norm_obj(
                json_data.get("ambiguousClauses"), "ambiguousClausesList"
            )
            json_data["modifiedStandardClauses"] = _norm_obj(
                json_data.get("modifiedStandardClauses"), "listOfModifiedClauses"
            )

            if not isinstance(json_data.get("contradictions"), dict):
                json_data["contradictions"] = {}
            if not isinstance(json_data.get("duplicateClauses"), list):
                json_data["duplicateClauses"] = []

            # Deep-fill defaults
            json_data.setdefault("highRiskClauses", {}).setdefault("clauses", [])
            json_data.setdefault("contradictions", {})
            json_data.setdefault("ambiguousClauses", {}).setdefault(
                "ambiguousClausesList", []
            )
            json_data.setdefault("duplicateClauses", [])
            json_data.setdefault("modifiedStandardClauses", {}).setdefault(
                "listOfModifiedClauses", []
            )

            # metrics
            msc = len(
                json_data.get("modifiedStandardClauses", {}).get(
                    "listOfModifiedClauses", []
                )
            )
            crt = len(json_data.get("contradictions", {}))
            nsz = len(json_data.get("highRiskClauses", {}).get("clauses", []))
            json_data["riskNumber"] = nsz + msc

            amb = len(
                json_data.get("ambiguousClauses", {}).get("ambiguousClausesList", [])
            )
            dpl = len(json_data.get("duplicateClauses", []))
            json_data["ambiguousNumber"] = amb + crt
            json_data["duplicatesNumber"] = dpl

            self.logger.debug("Contract analyzed successfully.")
            self.logger.debug("==================================================")
            self.logger.debug("End of contract analysis")
            self.logger.debug("==================================================")

            return json_data

        except Exception as e:

            self.logger.error(
                f"Couldn't return full JSON because of {e}. Returning partial data."
            )
            self.logger.debug("==================================================")
            self.logger.debug("End of contract analysis")
            self.logger.debug("==================================================")
            return {"_partial": cleaned_responses}


def main() -> bool:
    """
    Lightweight self-test:
      - Seeds a tiny in-memory 'document'
      - Mocks _ask_llm to return deterministic JSON
      - Runs analyze_contract_async and sanity-checks the shape
      - Prints OK/FAIL and returns a boolean
    """
    import asyncio
    import json

    print("CONTRACT_ANALYZER TEST START")
    try:
        # 1) Minimal analyzer with an in-memory document (no I/O)
        analyzer = ContractAnalyzer(package_id="SELFTEST", company_id="SELFTEST")
        analyzer.document_parts = [
            Part.from_text(
                text="Sample contract with Clause 4.1, 5.2, 7.1, 9.1, 12.2, 15.1."
            )
        ]
        analyzer.tokenized_text = {}

        # 2) Mock the LLM call so the test is deterministic & fast
        async def mock_ask_llm(self, prompt: str) -> str:
            return json.dumps(
                {
                    "paymentTerms1": {
                        "amount": "10000",
                        "due": "30 days",
                        "source": "Clause 4.1",
                    },
                    "paymentTerms2": {"due": "60 days", "source": "Clause 5.2"},
                    "modifiedStandardClauses": {
                        "listOfModifiedClauses": ["Clause 9.1", "Clause 12.2"]
                    },
                    "contradictions": {"Clause 7.1": "Conflicts with 4.3"},
                    "highRiskClauses": {"clauses": ["Clause 15.1"]},
                    "ambiguousClauses": {"ambiguousClausesList": ["Clause 3.4"]},
                    "duplicateClauses": ["Clause 10.2"],
                }
            )

        ContractAnalyzer._ask_llm = mock_ask_llm  # monkey-patch

        # 3) Run the analysis
        prompts = [
            "What are the payment terms?",
            "Are there any modified standard clauses?",
            "Are there any contradictions or ambiguities?",
        ]
        result = asyncio.run(analyzer.analyze_contract_async(prompts))

        # 4) Minimal shape checks -> boolean outcome
        ok = (
            isinstance(result, dict)
            and "paymentTerms" in result
            and "riskNumber" in result
            and "ambiguousNumber" in result
            and "duplicatesNumber" in result
        )

        if ok:
            print("CONTRACT_ANALYZER OK")
            return True
        else:
            print("CONTRACT_ANALYZER FAIL: unexpected result shape")
            return False

    except Exception as e:
        print(f"CONTRACT_ANALYZER ERROR: {e}")
        return False


if __name__ == "__main__":
    main()
