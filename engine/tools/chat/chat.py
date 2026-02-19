"""
TruBuild Chatbot - Gemini-based context-aware assistant for contract and RFP analysis.

This module defines `TBAChatbot`, a single-session chatbot that uses Google's Gemini model
to process user prompts in combination with structured document summaries, previously
run analytics, and optional Google search integration.

Main responsibilities:
- Accept user prompts and associated document metadata.
- Dynamically inject contextual banners based on the analysis type (e.g., contract, technical RFP).
- Load document fragments from disk cache (or fallback to remote, TODO: from cloud in future).
- Track token count to ensure context fits Gemini's model limit.
- Persist conversation history per thread.
- Detect and extract CSV-like tables from model output.

Dependencies:
- `google.genai` SDK for LLM calls.
- Local modules for web search, and fuzzy table parsing.

Usage:
    bot = TBAChatbot(package_id="demo", base_path="/data", conversation_id="thread-001")
    response = bot.get_response(prompt="Summarize risks", doc_meta=[{"name": ..., "category": ...}])
"""

from __future__ import annotations
import re, os
import json
import time
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Sequence, Iterable

from google.genai import types
from google.genai.types import Content, Part
from google.genai.types import Part
from utils.llm.LLM import get_client
from utils.storage.bucket import get_file
from utils.core.jsonval import _coerce_json
from urllib.parse import urlparse, unquote
from utils.core.web_search import google_search
from utils.storage.bucket import upload_file

# from utils.core.fuzzy_search import extract_tables_as_csv
from utils.document.docingest import DOCSTORE_DIR, get_documents
from utils.core.log import pid_tool_logger, get_logger, set_logger
from utils.core.slack import SlackActivityLogger, SlackActivityMeta, fmt_dur
from tools.chat.prompts_chat import load_metric_context, build_system_instruction
from utils.llm.compactor_cache import set_compactor_context, load_doc_summary_for_source

_MODEL = "gemini-2.5-flash"
_OVERFLOW_MSG = (
    "There is now more information in this conversation than TruBuild can "
    "process reliably in a single analysis. Please start a new chat and attach "
    "only the documents you need."
)

_OVERFLOW_RE = re.compile(
    r"input token count\s*\((\d+)\)\s*exceeds\s*the\s*maximum\s*number\s*of\s*tokens\s*allowed\s*\((\d+)\)",
    re.IGNORECASE,
)
_OVERFLOW_RE_ALT = re.compile(
    r"input token count\s+is\s+(\d+)\s+but model only supports up to\s+(\d+)",
    re.IGNORECASE,
)


class TBAChatbot:
    """
    Stateless chatbot engine for a single project and conversation thread.

        Encapsulates logic to:
        - Load relevant cached document fragments from disk.
        - Inject structured banners based on analysis tool usage (contract review, RFP, etc.).
        - Maintain a window of past messages for coherent multi-turn exchanges.
        - Call Gemini via Google Generative AI SDK and handle function calls.
        - Store chat history on disk.
        - Detect and extract structured tables in CSV form.

        lifecycle:
            bot = TBAChatbot(package_id="demo", base_path="/data", conversation_id="t1")
            reply = bot.get_response(prompt="What's risky here?", doc_meta=[{"name": ..., "category": ...}])
    """

    def __init__(
        self,
        package_id: str,
        *,
        conversation_id: str = "default",
        internet: bool = False,
        company_id: str | None = None,
        org_only: bool = False,
    ):
        """
        Initialize a chatbot session for a given project and conversation ID.

        Parameters:
            package_id (str): ID of the project folder.
            conversation_id (str): thread identifier.
            internet (bool): Enables Google search tool via Gemini function calls.

        Creates necessary folders, sets up system prompt, loads prior history.
        """
        set_logger(pid_tool_logger(package_id, "chat"))
        self.logger = get_logger()
        self.logger.debug(
            f"[init] Chatbot initialized conversation_id={conversation_id}"
        )

        self.package_id = package_id
        self.conv_id = conversation_id
        self.internet = internet

        # metrics and system prompts
        self.metrics_context = load_metric_context()
        self.system_instruction = build_system_instruction(internet)

        # google search tool
        self.search_tool = self._build_search_tool() if self.internet else None

        # history
        self.company_id = company_id
        self.org_only = org_only
        self._history: List[Content] = []
        self._history_loaded: bool = False

        # not to reappend documents in history already
        self._seen_docs: set[str] = set()

        self._manifest: dict[str, Any] | None = None
        self._manifest_loaded: bool = False
        # Send heavy base banner once, but send analysis banners when new categories appear
        self._banner_sent: bool = False
        self._seen_cats: set[str] = set()

    def _detect_categories(self, doc_meta: Iterable[Dict[str, str]]) -> set[str]:
        cats = set()
        for m in doc_meta or []:
            c = (m.get("category") or "").lower()
            sk = (m.get("source_key") or "").lower()

            if ("comm_rfp" in sk) or ("commercial" in c) or ("commercialrfp" in c):
                cats.add("comm_rfp")
            if ("tech_rfp" in sk) or ("techrfp" in c) or ("tech" in c):
                cats.add("tech_rfp")
            if ("contract" in c) or ("/contract/" in sk) or ("contractreview" in c):
                cats.add("contract")
        return cats

    # public entrypoint
    def get_response(
        self,
        prompt: str,
        *,
        doc_parts: List[Part],
        doc_meta: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        """
        Process a user prompt with contextual document and tool data.

        Parameters:
            prompt (str): The user's natural language query.
            doc_meta (List[Dict]): Metadata for each uploaded document with:
                - name (str): filename without `.json`
                - category (str): one of contractReview, techRfp, commercialRfp

        Returns:
            Dict with:
                - "answer": Gemini-generated reply
                - "table_csv": List of extracted CSV strings (if any)

        Workflow:
            1. Build banners (context summary + tool results).
            2. Load cached document Parts.
            3. Send full prompt context to Gemini.
            4. Store chat history.
            5. Return result + detected CSV tables.
        """

        self._ensure_history_loaded()

        doc_meta = doc_meta or []
        turn_parts: List[Part] = []

        # base banner + tool banners + cached docs
        if not self._banner_sent:
            manifest = self._load_manifest()
            uploaded_files = sorted({os.path.basename(k) for k in manifest.keys() if k})
            if uploaded_files:
                shown = uploaded_files[:30]
                more = len(uploaded_files) - len(shown)
                file_list = ", ".join(shown) + (
                    f" ... (+{more} more)" if more > 0 else ""
                )
                # base_banner_text = (
                #     "SYSTEM NOTE"
                #     "Filenames below are for identification only; their contents are NOT loaded by default.\n"
                #     f"Uploaded files: {file_list or 'None'}\n"
                #     "Do NOT assume access to any document unless its full text has been provided "
                #     "in the active conversation context (this turn or prior turns still present in history). "
                #     "If the user asks to analyze a file but no content was provided, ask them to select files via the file picker "
                #     "and resend the prompt.\n"
                #     "---------------------------------------------------------------------\n"
                # )
                # turn_parts.append(Part.from_text(text=base_banner_text))
                self._banner_sent = True

        cats_now = self._detect_categories(doc_meta)
        new_cats = [c for c in cats_now if c not in self._seen_cats]
        if new_cats:
            turn_parts.extend(self._analysis_banners(new_cats))
            self._seen_cats.update(new_cats)

        # Compute which docs are new this turn (by source_key) so that we don't reappend document content
        # multiple times in a conversation
        incoming_keys = [m.get("source_key") for m in doc_meta if m.get("source_key")]
        unseen_this_turn = [k for k in incoming_keys if k not in self._seen_docs]

        # If any new docs, include the doc_parts payload once and record keys as seen
        doc_parts_appended = False
        if doc_parts and unseen_this_turn:
            turn_parts.extend(doc_parts)
            for k in unseen_this_turn:
                self._seen_docs.add(k)
            doc_parts_appended = True
        # context update line (to avoid big banner repeat)
        # if unseen_this_turn:
        #     names = [m["name"] for m in doc_meta if m.get("source_key") in unseen_this_turn]
        #     turn_parts.append(
        #         Part.from_text(text=
        #             f"Context update: new documents added this turn: {', '.join(names)}. "
        #             f"Keys: {', '.join(unseen_this_turn)}."
        #         )
        #     )
        elif not incoming_keys:
            turn_parts.extend(doc_parts)
        # keep a debug marker in history (not sent to the model)
        if doc_parts_appended:
            self._history.append(
                Content(
                    role="system",
                    parts=[
                        Part.from_text(
                            text=f"__DOCS_ADDED__:{json.dumps(unseen_this_turn)}"
                        )
                    ],
                )
            )

        # finally the user prompt
        turn_parts.append(Part.from_text(text=prompt))

        # this turn as a proper user message
        current_user_msg = Content(role="user", parts=turn_parts)

        # call LLM
        answer_text, model_content = self._send_to_llm(current_user_msg)

        # persist history
        self._history.append(current_user_msg)
        if model_content:
            self._history.append(model_content)
        self._save_banner_flag_to_history()
        self._save_seen_cats_to_history()
        self._save_seen_docs_to_history()
        self._save_cloud_history(self._history)

        # CSV extraction
        # csv_tables = extract_tables_as_csv(answer)
        return {"answer": answer_text}

    def _analysis_banners(self, categories: Iterable[str]) -> List[Part]:
        banners: List[Part] = []

        def _load_json(path: str):
            raw = get_file(self.package_id, path, company_id=self.company_id)
            try:
                return _coerce_json(raw) or {}
            except Exception:
                return {}

        cats = set(c.lower() for c in (categories or []))

        # Contract: overview + recommendations
        if "contract" in cats:
            ovw = _load_json("data/contract_overview.json")
            rec = _load_json("data/contract_recommendation.json")
            if ovw:
                summary = (
                    "Contract analysis overview loaded. Key risks, obligations, and opportunities."
                    f"\nMetrics info:\n{self.metrics_context.strip()}"
                )
                banners.append(
                    Part.from_text(
                        text=self._fmt_banner(
                            "Contract Review - Overview", summary, ovw
                        )
                    )
                )
            if rec:
                banners.append(
                    Part.from_text(
                        text=self._fmt_banner(
                            "Contract Review - Recommendations",
                            "Recommended changes and negotiation levers.",
                            rec,
                        )
                    )
                )

        # Technical RFP
        if "tech_rfp" in cats:
            tech = _load_json("data/tech_rfp_report.json")
            if tech:
                banners.append(
                    Part.from_text(
                        text=self._fmt_banner(
                            "Technical RFP - Results",
                            "Fit analysis, integration complexity, and vendor scoring.",
                            tech,
                        )
                    )
                )

        return banners

    # LLM call
    def _build_search_tool(self) -> types.Tool:
        """
        Declares a Gemini-compatible function for `google_search(query)`.
        Used only when internet=True.
        """
        decl = {
            "name": "google_search",
            "description": "Performs a Google search and returns snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        return types.Tool(function_declarations=[decl])

    def _send_to_llm(self, current_user_msg: Content) -> tuple[str, Content | None]:
        """
        Sends the current chat window to Gemini.
        If the model requests a Google search, this is handled in a second round.
        Returns:
            The model's textual reply.
        """

        client = get_client()
        llm_history = [m for m in (self._history or []) if m.role in ("user", "model")]
        chat = client.chats.create(
            model=_MODEL,
            history=llm_history,
            config={"system_instruction": self.system_instruction},
        )

        cfg = types.GenerateContentConfig(
            tools=[self.search_tool] if self.search_tool else None,
            temperature=0.4,
            max_output_tokens=20000,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="text/plain",
            system_instruction=self.system_instruction,
        )

        def _first_fn_call(resp):
            if not (resp and resp.candidates):
                return None
            parts = resp.candidates[0].content.parts or []
            return getattr(parts[0], "function_call", None) if parts else None

        try:
            resp = chat.send_message(current_user_msg.parts, config=cfg)
        except Exception as e:
            is_ovf, used, limit = self._is_overflow_error(e)
            if is_ovf:
                if used and limit:
                    self.logger.warning(
                        "Context overflow: input %s > max %s", used, limit
                    )
                else:
                    self.logger.warning(
                        "Context overflow detected from SDK error: %s", e
                    )
                return _OVERFLOW_MSG, None
            raise

        # tool calling (google search)
        for _ in range(3):
            fc = _first_fn_call(resp)
            if not fc:
                break
            name = getattr(fc, "name", "")
            args = getattr(fc, "args", {}) or {}
            if name == "google_search":
                query = args.get("query", "")
                result = google_search(query)
                try:
                    resp = chat.send_message(
                        types.Part.from_function_response(
                            name="google_search",
                            response={"content": result},
                        ),
                        config=cfg,
                    )
                except Exception as e:
                    is_ovf, used, limit = self._is_overflow_error(e)
                    if is_ovf:
                        if used and limit:
                            self.logger.warning(
                                "Context overflow (after tool): input %s > max %s",
                                used,
                                limit,
                            )
                        else:
                            self.logger.warning(
                                "Context overflow after tool call: %s", e
                            )
                        return _OVERFLOW_MSG, None
                    raise
            else:
                break

        text = getattr(resp, "text", None)
        model_content = None
        if resp and resp.candidates:
            model_content = resp.candidates[0].content
            if not text:
                # fallback: concatenate text parts
                buf = []
                for p in model_content.parts or []:
                    if getattr(p, "text", None):
                        buf.append(p.text)
                text = "\n".join(buf).strip() or "I couldn’t generate a response."

        return text, model_content

    def _fmt_banner(self, title: str, summary: str, data: Dict[str, Any]) -> str:
        """
        Pretty-formats a section of context to include in the LLM request.
        """
        compact_data = json.dumps(data, separators=(",", ":"))
        return (
            f"---- {title} -----------------------------------------------\n"
            f"{summary}\n\n"
            f"{compact_data}\n"
            "---------------------------------------------------------------------\n"
        )

    def _hist_key(self) -> str:
        # SPECIAL CASE:
        # If org-only (no package_id), store chat history at:
        #   companyId/chathistory_overview/<conversation_id>.json
        # i.e., NO "data/chat_history/" prefix.
        if self.org_only:
            return f"{self.conv_id}.json"
        # Default behavior (project scoped)
        return f"data/chat_history/{self.conv_id}.json"

    def _load_cloud_history(self) -> List[Content]:
        try:
            raw = get_file(
                self.package_id, self._hist_key(), company_id=self.company_id
            )
            if not raw:
                return []
            data = _coerce_json(raw) or []
            return [Content.model_validate(r) for r in data]
        except Exception as e:
            self.logger.error("Corrupt or missing history; starting fresh: %s", e)
            return []

    def _save_cloud_history(self, hist: Sequence[Content]) -> None:
        try:
            payload = json.dumps([c.to_json_dict() for c in hist], indent=2)
            upload_file(
                package_id=self.package_id,
                location=self._hist_key(),
                json_string=payload,
                company_id=self.company_id,
            )
        except Exception as e:
            self.logger.error("Failed to save history: %s", e)

    def _seen_docs_key(self) -> str:
        return "__SEEN_DOCS__"

    def _load_seen_docs_from_history(self) -> None:
        self._seen_docs = set()
        for msg in self._history:
            if msg.role != "system":
                continue
            for p in msg.parts or []:
                txt = getattr(p, "text", "")
                if txt.startswith(self._seen_docs_key()):
                    try:
                        payload = txt.split(":", 1)[1].strip()
                        self._seen_docs = set(json.loads(payload))
                    except Exception:
                        self._seen_docs = set()
                    return

    def _save_seen_docs_to_history(self) -> None:
        marker_text = f"{self._seen_docs_key()}: {json.dumps(sorted(self._seen_docs))}"
        marker = Content(role="system", parts=[Part.from_text(text=marker_text)])
        for i, msg in enumerate(self._history):
            if (
                msg.role == "system"
                and msg.parts
                and getattr(msg.parts[0], "text", "").startswith(self._seen_docs_key())
            ):
                self._history[i] = marker
                break
        else:
            self._history.append(marker)

    def _ensure_history_loaded(self) -> None:
        if not self._history_loaded:
            self._history = self._load_cloud_history()
            self._history_loaded = True
            self._load_seen_docs_from_history()
            self._load_banner_flag_from_history()
            self._load_seen_cats_from_history()

    def unseen_source_keys(self, doc_meta: list[dict]) -> list[str]:
        """Returns only source_keys not yet injected this conversation."""
        self._ensure_history_loaded()
        incoming = [
            m.get("source_key") for m in (doc_meta or []) if m.get("source_key")
        ]
        return [k for k in incoming if k not in self._seen_docs]

    def _load_manifest(self) -> dict:
        if self._manifest_loaded and self._manifest is not None:
            return self._manifest
        raw = get_file(
            self.package_id,
            f"{DOCSTORE_DIR}manifest.json",
            company_id=self.company_id,
        )
        obj = _coerce_json(raw)
        self._manifest = obj or {}
        self._manifest_loaded = True
        return self._manifest

    def _is_overflow_error(self, exc: Exception) -> tuple[bool, int | None, int | None]:
        """
        Return (is_overflow, input_tokens, max_tokens). Works with the Gemini SDK's
        INVALID_ARGUMENT / 400 error that says:
        "The input token count (X) exceeds the maximum number of tokens allowed (Y)."
        """
        msg = str(exc)
        msg_lower = msg.lower()
        status_ok = ("invalid_argument" in msg_lower) or ("400" in msg_lower)
        m = _OVERFLOW_RE.search(msg)
        if not m:
            m = _OVERFLOW_RE_ALT.search(msg)
        if m:
            try:
                used = int(m.group(1))
                limit = int(m.group(2))
            except Exception:
                used = limit = None
            return True, used, limit

        # Fallback: incaseof version update, error msg may vary slightly — still treat as overflow
        if status_ok and (
            "exceeds" in msg_lower and "maximum" in msg_lower and "token" in msg_lower
        ):
            return True, None, None

        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if code in (400, "INVALID_ARGUMENT"):
            if "token" in msg_lower and "exceed" in msg_lower:
                return True, None, None

        return False, None, None

    def _load_banner_flag_from_history(self):
        for msg in self._history:
            if msg.role == "system":
                for p in msg.parts or []:
                    if getattr(p, "text", "").strip() == "__BANNER_SENT__":
                        self._banner_sent = True
                        return

    def _save_banner_flag_to_history(self):
        marker = Content(role="system", parts=[Part.from_text(text="__BANNER_SENT__")])
        for i, msg in enumerate(self._history):
            if (
                msg.role == "system"
                and msg.parts
                and getattr(msg.parts[0], "text", "").strip() == "__BANNER_SENT__"
            ):
                self._history[i] = marker
                return
        self._history.append(marker)

    def _load_seen_cats_from_history(self):
        for msg in self._history:
            if msg.role == "system":
                for p in msg.parts or []:
                    if getattr(p, "text", "").startswith("__SEEN_CATS__:"):
                        try:
                            payload = getattr(p, "text", "").split(":", 1)[1].strip()
                            self._seen_cats = set(json.loads(payload))
                        except Exception:
                            self._seen_cats = set()
                        return

    def _save_seen_cats_to_history(self):
        marker = Content(
            role="system",
            parts=[
                Part.from_text(
                    text=f"__SEEN_CATS__:{json.dumps(sorted(self._seen_cats))}"
                )
            ],
        )
        for i, msg in enumerate(self._history):
            if (
                msg.role == "system"
                and msg.parts
                and getattr(msg.parts[0], "text", "").startswith("__SEEN_CATS__:")
            ):
                self._history[i] = marker
                return
        self._history.append(marker)


async def chat_main(
    *,
    package_id: str | None = None,
    conversation_id: str = "default",
    prompt: str | None = None,
    documents: List[Dict[str, str]] | None = None,
    internet: bool = False,
    country_code: str,
    company_id: str | None = None,
    request_method: str | None = None,
    remote_ip: str | None = None,
    user_name: str | None = None,
) -> Dict[str, Any]:
    """
    Handle POST requests for the TruBuild Chatbot.

    Returns dict ready for handle() wrapper.
    """
    base_logger = pid_tool_logger(package_id=package_id or "unknown", tool_name="chat")
    set_logger(
        base_logger,
        tool_name="chat_main",
        tool_base="CHAT",
        package_id=package_id or "unknown",
        ip_address=remote_ip or "no_ip",
        request_type=request_method or "N/A",
    )
    logger = get_logger()

    act = SlackActivityLogger(
        SlackActivityMeta(
            package_id=package_id, tool="CHAT", user=user_name, company=company_id
        )
    )
    parent_ts = act.start()
    start_t = time.perf_counter()
    act.sub("💬 Chat turn started")
    try:
        # Validate
        if request_method != "POST":
            return {"error": "Only POST is supported for /chat", "status": "error"}
        if not prompt:
            return {"error": "prompt is required", "status": "error"}
        if not (package_id or company_id):
            return {
                "error": "Either packageId or companyId is required",
                "status": "error",
            }

        documents = documents or []
        country_code = country_code or "USA"

        ORG_ONLY = bool(company_id and not package_id)
        effective_package_id = package_id or "chathistory_overview"
        comm_ctx_raw = None
        set_compactor_context(package_id=effective_package_id, company_id=company_id)

        # Ensure all document fragments are extracted and cached
        def _normalize_source_key(
            company_id: str | None, effective_package_id: str, url_or_path: str
        ) -> str:
            """
            Make a nested cloud key:
            - org-only:    company/effective_project/...
            - normal proj: company/project/...
            """
            p = unquote(urlparse(url_or_path).path).lstrip("/")
            if company_id and p.startswith(f"{company_id}/{effective_package_id}/"):
                return p
            if company_id and package_id and p.startswith(f"{package_id}/"):
                # legacy shape; add company
                return f"{company_id}/{p}"
            # default: join under effective root
            return (
                f"{company_id}/{effective_package_id}/{p}"
                if company_id
                else f"{effective_package_id}/{p}"
            )

        def _looks_like_alias_hash(value: str | None) -> bool:
            s = (value or "").strip().lower()
            return bool(re.fullmatch(r"[0-9a-f]{64}", s))

        def _extract_source_locator(doc: Dict[str, Any]) -> str | None:
            # Prefer explicit source-key style fields; fall back to URL/path.
            for key in (
                "source_key",
                "sourceKey",
                "storage_key",
                "storageKey",
                "path",
                "file_path",
                "url",
            ):
                v = doc.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # Some clients only send a precomputed document hash/alias key.
            for key in ("hash", "doc_hash", "documentHash", "sourceHash"):
                v = doc.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return None

        processed_documents_meta = []
        for doc in documents:
            if "name" in doc:
                try:
                    raw_locator = _extract_source_locator(doc)
                    if not raw_locator:
                        logger.warning(
                            "Skipping document with no URL/source key/hash: name=%s",
                            doc.get("name"),
                        )
                        continue

                    # If this is already a 64-hex alias hash, pass through as-is.
                    source_key = (
                        raw_locator
                        if _looks_like_alias_hash(raw_locator)
                        else _normalize_source_key(
                            company_id, effective_package_id, raw_locator
                        )
                    )

                    processed_documents_meta.append(
                        {
                            "name": doc["name"],
                            "source_key": source_key,
                            "category": doc.get("category", "unknown"),
                        }
                    )
                except (IndexError, AttributeError):
                    logger.warning(
                        "Could not parse a valid source locator for document: %s",
                        doc.get("name"),
                    )

        # spin up the chatbot
        bot = TBAChatbot(
            package_id=effective_package_id,
            conversation_id=conversation_id,
            internet=True,
            company_id=company_id,
            org_only=ORG_ONLY,
        )

        doc_parts: list[Part] = []
        # 1) Try to load precomputed Comm RFP chat context for this packageId
        # If we have a real package_id, try that first (matches new_comm_rfp)
        if package_id:
            comm_ctx_raw = get_file(
                package_id=package_id,
                file_path="comm_rfp/data/chat_context.json",
                company_id=company_id,
            )

        # 2) Fallback: org-only / overview scope
        if not comm_ctx_raw:
            comm_ctx_raw = get_file(
                package_id=effective_package_id,
                file_path="comm_rfp/data/chat_context.json",
                company_id=company_id,
            )
        logger.debug(
            "comm_ctx lookup company_id=%s package_id=%s effective_package_id=%s found=%s",
            company_id,
            package_id,
            effective_package_id,
            bool(comm_ctx_raw),
        )
        comm_ctx = _coerce_json(comm_ctx_raw) if comm_ctx_raw else None

        if isinstance(comm_ctx, dict):
            assets_ctx = (
                comm_ctx.get("assets")
                or comm_ctx.get("comm_rfp", {}).get("assets")
                or {}
            )

            if assets_ctx:
                lines: list[str] = []
                for asset_id, a in assets_ctx.items():
                    asset_name = a.get("assetName") or asset_id
                    summary = (a.get("summary") or "").strip()
                    key_findings = a.get("keyFindings") or []

                    # Build a compact but rich block per asset
                    block_lines = [f"Asset: {asset_name}"]
                    if summary:
                        block_lines.append(summary)
                    if key_findings:
                        block_lines.append("Key findings:")
                        block_lines.extend(f"- {kf}" for kf in key_findings)

                    lines.append("\n".join(block_lines))

                if lines:
                    doc_parts.append(
                        Part.from_text(
                            text=(
                                "Commercial evaluation context for this package: "
                                "\n\n" + "\n\n---\n\n".join(lines)
                            )
                        )
                    )

        unseen_keys = bot.unseen_source_keys(processed_documents_meta)
        unseen_metas = [
            m for m in processed_documents_meta if m.get("source_key") in unseen_keys
        ]
        selected_names = [
            m.get("name") for m in processed_documents_meta if m.get("name")
        ]

        if selected_names:
            doc_parts.append(
                Part.from_text(
                    text=(
                        "User-selected documents for this turn:\n- "
                        + "\n- ".join(selected_names)
                        + "\n"
                        "When helpful, explicitly name the file(s) you’re citing to avoid ambiguity."
                    )
                )
            )

        if unseen_metas:
            groups = {"default": [], "comm_rfp": [], "tech_eval": []}
            for m in unseen_metas:
                cat = (m.get("category") or "").lower()
                sk = (m.get("source_key") or "").lower()
                if "comm_rfp" in sk or "commercial" in cat or "commercialrfp" in cat:
                    groups["comm_rfp"].append(m)
                elif (
                    "tech" in cat
                    or "techrfp" in cat
                    or "tech_eval" in sk
                    or "tech_rfp/" in sk
                ):
                    groups["tech_eval"].append(m)
                else:
                    groups["default"].append(m)

            if groups["default"]:
                parts, _meta = await get_documents(
                    package_id=effective_package_id,
                    company_id=company_id,
                    document_metas=groups["default"],
                    tool_context=None,
                )
                doc_parts.extend(parts)

            if groups["comm_rfp"]:
                for m in groups["comm_rfp"]:
                    if not m.get("contractor"):
                        # heuristic: parent folder name; else file stem
                        try:
                            m["contractor"] = Path(m["source_key"]).parts[-2]
                        except Exception:
                            m["contractor"] = Path(m["name"]).stem or "Vendor"
                tables, meta_cr = await get_documents(
                    package_id=effective_package_id,
                    company_id=company_id,
                    document_metas=groups["comm_rfp"],
                    tool_context="comm_rfp",
                )
                currencies = (meta_cr or {}).get("contractor_currencies", {})
                for contractor, df in (tables or {}).items():
                    if df is None or getattr(df, "empty", True):
                        continue
                    csv = df.to_csv(index=False)
                    doc_parts.append(
                        Part.from_text(
                            text=f"<boq contractor='{contractor}' currency='{currencies.get(contractor, '')}'>\n{csv}\n</boq>"
                        )
                    )

            if groups["tech_eval"]:
                tech_parts: list[Part] = []
                missing_for_fallback: list[dict] = []

                logger.debug("CHAT: tech_eval incoming=%d", len(groups["tech_eval"]))

                def _is_eval_path(m: dict) -> bool:
                    sk = (m.get("source_key") or "").lower()
                    return "/tech_rfp/evaluation/" in sk or "tech_rfp/evaluation/" in sk

                eval_selected = any(_is_eval_path(m) for m in groups["tech_eval"])
                EVAL_SEEN_MARK = f"__EVAL_JSON__:{effective_package_id}"
                if eval_selected and EVAL_SEEN_MARK not in bot._seen_docs:
                    try:
                        # try to load cached evaluation json
                        raw = get_file(
                            effective_package_id,
                            "data/evaluation.json",
                            company_id=company_id,
                        )
                        eval_data = _coerce_json(raw) or {}
                        if eval_data:
                            eval_name = next(
                                (
                                    m.get("name")
                                    or Path(m.get("source_key") or "").name
                                    for m in groups["tech_eval"]
                                    if _is_eval_path(m)
                                ),
                                "evaluation.json",
                            )
                            body = json.dumps(eval_data, ensure_ascii=False)
                            tech_parts.append(
                                Part.from_text(
                                    text=f"<Start of File: {eval_name}>\n{body}\n</End of File: {eval_name}>\n"
                                )
                            )
                            bot._seen_docs.add(EVAL_SEEN_MARK)
                        else:
                            logger.warning(
                                "evaluation.json is empty or missing; falling back to raw loading."
                            )
                    except Exception as e:
                        logger.exception(
                            "Failed to load data/evaluation.json; falling back: %s", e
                        )

                # Always strip the eval file metas to avoid empty/raw reloads
                groups["tech_eval"] = [
                    m for m in groups["tech_eval"] if not _is_eval_path(m)
                ]

                if not groups["tech_eval"]:
                    doc_parts.extend(tech_parts)
                else:
                    # Try per-doc summary by source_key (alias pointer)
                    hits = misses = 0
                    for m in groups["tech_eval"]:
                        sk = (m.get("source_key") or "").strip()
                        name = m.get("name") or Path(sk).name

                        summary_parts = load_doc_summary_for_source(sk) if sk else None
                        if summary_parts:
                            tech_parts.append(
                                Part.from_text(text=f"<Start of File: {name}>\n")
                            )
                            tech_parts.extend(summary_parts)
                            tech_parts.append(
                                Part.from_text(text=f"</End of File: {name}>\n")
                            )
                            logger.debug("CHAT: per-doc summary HIT name=%s", name)
                            hits += 1
                        else:
                            missing_for_fallback.append(m)
                            logger.debug("CHAT: per-doc summary MISS name=%s", name)
                            misses += 1

                    logger.debug(
                        "CHAT: tech_eval summary hits=%d misses=%d", hits, misses
                    )

                    # For misses, load raw parts
                    if missing_for_fallback:
                        raw_budget_tokens = int(
                            os.getenv(
                                "CHAT_TECH_EVAL_RAW_FALLBACK_BUDGET_TOKENS", "160000"
                            )
                        )
                        raw_used_tokens = 0
                        raw_skipped_docs = 0

                        def _estimate_part_tokens(parts: list[Part]) -> int:
                            total = 0
                            for p in (parts or []):
                                t = getattr(p, "text", None)
                                if isinstance(t, str) and t:
                                    total += max(1, len(t) // 4)
                                else:
                                    total += 256
                            return total

                        def _estimate_payload_tokens(src_key: str, payload_obj: Any) -> int:
                            nm = Path(src_key or "").name or "document"
                            if (
                                isinstance(payload_obj, list)
                                and payload_obj
                                and all(isinstance(x, Part) for x in payload_obj)
                            ):
                                return _estimate_part_tokens(payload_obj)
                            try:
                                body = json.dumps(payload_obj, ensure_ascii=False)
                            except Exception:
                                body = str(payload_obj)
                            tagged = (
                                f"<Start of File: {nm}>\n"
                                f"{body}\n"
                                f"</End of File: {nm}>\n"
                            )
                            return max(1, len(tagged) // 4)

                        evals, _meta_te = await get_documents(
                            package_id=effective_package_id,
                            company_id=company_id,
                            document_metas=missing_for_fallback,
                            tool_context="tech_eval",
                        )

                        for source_key, payload in (evals or {}).items():
                            name = Path(source_key).name

                            def _label_line_for(
                                source_key: str | None, fallback_name: str | None
                            ) -> str:
                                nm = (fallback_name or "").strip() or Path(
                                    (source_key or "")
                                ).name
                                return (nm or "document").strip()

                            def _as_labeled_text_part(src_key: str, obj) -> Part:
                                name = _label_line_for(src_key, Path(src_key).name)
                                try:
                                    df = pd.DataFrame(obj)
                                    if not df.empty:
                                        df = df.loc[
                                            :,
                                            ~df.columns.astype(str).str.match(
                                                r"^unnamed:\s*\d+$", case=False
                                            ),
                                        ]
                                        csv = df.to_csv(index=False)
                                        body = csv
                                    else:
                                        body = json.dumps(obj, ensure_ascii=False)
                                except Exception:
                                    body = json.dumps(obj, ensure_ascii=False)

                                return Part.from_text(
                                    text=f"<Start of File: {name}>\n{body}\n</End of File: {name}>\n"
                                )

                            if isinstance(payload, list):
                                # If it's already Parts, prepend a label and add a spacer; otherwise coerce to CSV/JSON text
                                if payload and all(
                                    isinstance(x, Part) for x in payload
                                ):
                                    est = _estimate_payload_tokens(source_key, payload)
                                    if raw_used_tokens + est > raw_budget_tokens:
                                        raw_skipped_docs += 1
                                        tech_parts.append(
                                            Part.from_text(
                                                text=(
                                                    f"<Start of File: {name}>\n"
                                                    "[Raw content omitted in chat fallback to stay within context budget. "
                                                    "Cached compacted summary was not found for this file.]\n"
                                                    f"</End of File: {name}>\n"
                                                )
                                            )
                                        )
                                        logger.warning(
                                            "CHAT: skipping RAW parts (budget) name=%s est_tokens=%d used=%d budget=%d",
                                            name,
                                            est,
                                            raw_used_tokens,
                                            raw_budget_tokens,
                                        )
                                        continue
                                    tech_parts.append(
                                        Part.from_text(
                                            text=f"{_label_line_for(source_key, name)}\n\n"
                                        )
                                    )
                                    tech_parts.extend(payload)
                                    tech_parts.append(Part.from_text(text="\n"))
                                    raw_used_tokens += est
                                    logger.debug(
                                        "CHAT: using RAW parts name=%s count=%d est_tokens=%d used=%d/%d",
                                        name,
                                        len(payload),
                                        est,
                                        raw_used_tokens,
                                        raw_budget_tokens,
                                    )
                                else:
                                    est = _estimate_payload_tokens(source_key, payload)
                                    if raw_used_tokens + est > raw_budget_tokens:
                                        raw_skipped_docs += 1
                                        tech_parts.append(
                                            Part.from_text(
                                                text=(
                                                    f"<Start of File: {name}>\n"
                                                    "[Raw content omitted in chat fallback to stay within context budget. "
                                                    "Cached compacted summary was not found for this file.]\n"
                                                    f"</End of File: {name}>\n"
                                                )
                                            )
                                        )
                                        logger.warning(
                                            "CHAT: skipping RAW list payload (budget) name=%s est_tokens=%d used=%d budget=%d",
                                            name,
                                            est,
                                            raw_used_tokens,
                                            raw_budget_tokens,
                                        )
                                        continue
                                    tech_parts.append(
                                        _as_labeled_text_part(source_key, payload)
                                    )
                                    raw_used_tokens += est
                                    logger.debug(
                                        "CHAT: coerced LIST payload to text Part name=%s est_tokens=%d used=%d/%d",
                                        name,
                                        est,
                                        raw_used_tokens,
                                        raw_budget_tokens,
                                    )
                            else:
                                # Non-list: try DataFrame/CSV first, then JSON
                                est = _estimate_payload_tokens(source_key, payload)
                                if raw_used_tokens + est > raw_budget_tokens:
                                    raw_skipped_docs += 1
                                    tech_parts.append(
                                        Part.from_text(
                                            text=(
                                                f"<Start of File: {name}>\n"
                                                "[Raw content omitted in chat fallback to stay within context budget. "
                                                "Cached compacted summary was not found for this file.]\n"
                                                f"</End of File: {name}>\n"
                                            )
                                        )
                                    )
                                    logger.warning(
                                        "CHAT: skipping RAW non-list payload (budget) name=%s est_tokens=%d used=%d budget=%d",
                                        name,
                                        est,
                                        raw_used_tokens,
                                        raw_budget_tokens,
                                    )
                                    continue
                                tech_parts.append(
                                    _as_labeled_text_part(source_key, payload)
                                )
                                raw_used_tokens += est
                                logger.debug(
                                    "CHAT: coerced NON-LIST payload to text Part name=%s est_tokens=%d used=%d/%d",
                                    name,
                                    est,
                                    raw_used_tokens,
                                    raw_budget_tokens,
                                )

                        logger.debug(
                            "CHAT: tech_eval compiled parts count=%d raw_used=%d raw_budget=%d raw_skipped_docs=%d",
                            len(tech_parts),
                            raw_used_tokens,
                            raw_budget_tokens,
                            raw_skipped_docs,
                        )

                    # Inject all tech parts into the turn
                    doc_parts.extend(tech_parts)

        response = bot.get_response(
            prompt=prompt,
            doc_parts=doc_parts,
            doc_meta=processed_documents_meta,
        )

        dur = fmt_dur(time.perf_counter() - start_t)
        logger.info(f"Chat turn completed in {dur}")
        act.sub(f"💬 Chat turn completed in {dur}")

        payload: Dict[str, Any] = {
            "status": "success",
            "answer": response["answer"],
        }
        act.done()
        return payload

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        dur = fmt_dur(time.perf_counter() - start_t)
        act.error(f"🔥Chat turn FAILED after {dur} — {e}")
        return {"error": str(e), "status": "error"}


def main() -> bool:
    """
    Lightweight self-test for the Chat tool:
      - Stubs get_documents / bucket I/O in-module
      - Stubs the LLM call (_send_to_llm) to avoid network
      - Runs chat_main once and validates a minimal success shape
      - Prints OK/FAIL and returns a boolean
    """
    import asyncio
    import json
    from google.genai.types import Part

    print("CHAT TOOL TEST START")
    try:
        # Stub dependencies used inside this module
        async def _fake_get_documents(*_, **__):
            parts = [
                Part.from_text(
                    text="<test_doc.pdf>\nThis is a tiny test doc.\n</test_doc.pdf>"
                )
            ]
            return parts, {}  # (doc_parts, meta)

        def _fake_get_file(*_, **__):
            # Return empty JSON so banner loaders don't blow up
            return json.dumps({})

        def _fake_upload_file(*_, **__):
            return None

        # Bind stubs to the names this module uses
        global get_documents, get_file, upload_file
        get_documents = (
            _fake_get_documents  # overrides imported name from utils.document.docingest
        )
        get_file = _fake_get_file  # overrides imported name from utils.storage.bucket
        upload_file = (
            _fake_upload_file  # overrides imported name from utils.storage.bucket
        )

        # Stub the LLM call so no SDK/network is used
        def _fake_send_to_llm(self, current_user_msg):
            return ("Stubbed reply: I summarized your document.", None)

        TBAChatbot._send_to_llm = _fake_send_to_llm  # type: ignore

        # Run one POST turn through chat_main
        async def _run_once():
            return await chat_main(
                package_id="test_package_id",
                conversation_id="thread-001",
                prompt="Summarize the attached document for me.",
                documents=[
                    {
                        "name": "test_doc.pdf",
                        "url": "https://fake-bucket.s3.amazonaws.com/test_package_id/chat_uploads/test_doc.pdf",
                        "category": "chatUpload",
                    }
                ],
                internet=False,
                country_code="USA",
                request_method="POST",
                remote_ip="127.0.0.1",
            )

        result = asyncio.run(_run_once())

        ok = (
            isinstance(result, dict)
            and result.get("status") == "success"
            and isinstance(result.get("answer"), str)
            and len(result.get("answer", "").strip()) > 0
        )

        if ok:
            print("CHAT TOOL OK")
            return True
        else:
            print("CHAT TOOL FAIL: unexpected result shape")
            return False

    except Exception as e:
        print(f"CHAT TOOL ERROR: {e}")
        return False


if __name__ == "__main__":
    main()
