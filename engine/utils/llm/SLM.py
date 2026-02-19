"""
SLM.py - Small Language Model utilities on top of OpenRouter (uses utils/LLM_OR.py)

Key goals
- Fast, inexpensive "niche" tasks on small models (text & vision), with per-task overrides
- JSON-first responses with parsing & controlled retries
- Clean async API with a tiny sync func for testing and convenience
- timeouts, provider pinning, data_collection="deny", optional streaming

Environment
- OPENROUTER_API_KEY (required)

Usage (async):
    from utils.llm.SLM import SLM
    slm = SLM()
    result = await slm.classify_text("Is this urgent? Limited time only!", labels=["spam","ham"])

Usage (sync):
    from utils.llm.SLM import slm_sync
    result = slm_sync.classify_text("...", labels=["a","b"])

Run demo:
    python -m utils.SLM.py
"""

from __future__ import annotations

import os
import json
import base64
import random
import asyncio
import logging
from dataclasses import dataclass
from utils.core.log import set_logger, get_logger
from utils.vault import secrets
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from utils.llm.LLM_OR import (
    AsyncOpenRouterLLM,
    ChatMessage,
    ProviderPreferences,
    OpenRouterError,
)

logger = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_VISION_MODEL = "qwen/qwen-2.5-vl-7b-instruct"

# Prefer SLM-friendly providers (override per call). From Vault or ephemeral env.
_provider_order_raw = (
    secrets.get("SLM_PROVIDER_ORDER", default="") or os.getenv("SLM_PROVIDER_ORDER", "")
).strip()
DEFAULT_PROVIDER_ORDER: Optional[List[str]] = (
    [p.strip() for p in _provider_order_raw.split(",") if p.strip()]
    if _provider_order_raw
    else None
)
ALLOW_FALLBACKS: bool = False  # be explicit; flip to True if  resilience > determinism
DENY_DATA_COLLECTION: bool = True  # prefer providers that don't store data
REQUIRE_JSON: bool = True  # add response_format={"type":"json_object"} and require_parameters for providers that guarantee JSON

# Model-agnostic sampling defaults (cheap/reliable)
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 800  # small outputs; override for longer summaries

# Task-local JSON retry policy
JSON_MAX_ATTEMPTS = 3
JSON_BACKOFF_BASE = 0.4  # seconds

def _image_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _make_provider(require_json: bool = REQUIRE_JSON) -> Optional[ProviderPreferences]:
    return ProviderPreferences(
        order=DEFAULT_PROVIDER_ORDER or None,
        allow_fallbacks=ALLOW_FALLBACKS,
        data_collection="deny" if DENY_DATA_COLLECTION else None,
        require_parameters=True if require_json else None,  # enforce JSON-compatible providers
        # sort="price"  # uncomment for OR to optimize on price among the allowed providers (could be speed too)
    )

def _response_format(require_json: bool = REQUIRE_JSON) -> Optional[Dict[str, Any]]:
    return {"type": "json_object"} if require_json else None

def _messages_for_text(system: Optional[str], user_prompt: str) -> List[ChatMessage]:
    msgs = []
    if system:
        msgs.append(ChatMessage(role="system", content=system))
    msgs.append(ChatMessage(role="user", content=user_prompt))
    return msgs

def _messages_for_vision(system: Optional[str], prompt_text: str, images: List[Union[str, bytes]]) -> List[ChatMessage]:
    """
    images: list of either
      - bytes (will be embedded as data URLs), or
      - str URLs (https:// or data: already)
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for img in images:
        if isinstance(img, (bytes, bytearray)):
            url = _image_data_url(img, "image/png")
        else:
            url = str(img)
        content.append({"type": "image_url", "image_url": {"url": url}})
    msgs = []
    if system:
        msgs.append(ChatMessage(role="system", content=system))
    msgs.append(ChatMessage(role="user", content=content))
    return msgs

def _safe_json_parse(text: str) -> Tuple[bool, Any]:
    """Parse JSON or JSON-ish (including fenced code). Returns (ok, obj_or_text)."""
    if not isinstance(text, str) or not text.strip():
        return False, text
    s = text.strip()
    # Remove ```json fences
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1 and s.endswith("```"):
            s = s[first_nl + 1 : -3].strip()
    try:
        return True, json.loads(s)
    except json.JSONDecodeError:
        return False, s

def _maybe_schema_check(obj: Any, required_keys: Sequence[str]) -> bool:
    """Very light schema check without dependencies: all required keys must exist."""
    if not isinstance(obj, dict):
        return False
    return all(k in obj for k in required_keys)

def _sample_cfg(attempt: int, temperature: float) -> Dict[str, Any]:
    """Jitter temperature/seed across retries to escape local minima."""
    if attempt == 1:
        return {"temperature": temperature, "top_p": DEFAULT_TOP_P}
    return {
        "temperature": round(min(1.0, max(0.0, random.uniform(temperature, temperature + 0.5))), 2),
        "top_p": DEFAULT_TOP_P,
        "seed": random.randrange(1, 10_000),
    }

# Core Client

@dataclass
class SLM:
    """
    orchestration layer over AsyncOpenRouterLLM for SLM-style tasks.
    """
    default_text_model: str = DEFAULT_TEXT_MODEL
    default_vision_model: str = DEFAULT_VISION_MODEL

    # generic helpers
    async def _chat_json(
        self,
        *,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        require_json: bool = REQUIRE_JSON,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 60.0,
        user: str = "slm",
        provider: Optional[ProviderPreferences] = None,
        attempts: int = JSON_MAX_ATTEMPTS,
    ) -> Any:
        """
        JSON-oriented call with resilient parsing and up to `attempts` retries.
        """
        model = model or self.default_text_model
        provider = provider or _make_provider(require_json)
        resp_format = _response_format(require_json)

        last_text = None
        for attempt in range(1, max(1, attempts) + 1):
            cfg = _sample_cfg(attempt, temperature)
            try:
                async with AsyncOpenRouterLLM.from_env(timeout=timeout) as client:
                    resp = await client.chat(
                        model=model,
                        messages=messages,
                        response_format=resp_format,
                        provider=provider,
                        user=user,
                        max_tokens=max_tokens,
                        **cfg,
                    )
                text = resp.text or ""
            except OpenRouterError as e:
                # Retry only for transient errors; client already has its own internal retries,
                # but we still bounce once or twice for result-shape issues or flakiness.
                logger.warning("OpenRouter error: %s (attempt %d/%d)", e, attempt, attempts)
                text = ""

            ok, obj = _safe_json_parse(text)
            if ok:
                return obj

            last_text = text
            if attempt < attempts:
                backoff = JSON_BACKOFF_BASE * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)

        # Last resort: return best-effort shape to not break callers
        logger.warning("Returning unparsed text after %d attempts.", attempts)
        return {"_raw": (last_text or ""), "_parsed": False}

    async def _chat_text(
        self,
        *,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 60.0,
        user: str = "slm",
        provider: Optional[ProviderPreferences] = None,
    ) -> str:
        model = model or self.default_text_model
        provider = provider or _make_provider(False)
        try:
            async with AsyncOpenRouterLLM.from_env(timeout=timeout) as client:
                resp = await client.chat(
                    model=model,
                    messages=messages,
                    response_format=None,
                    provider=provider,
                    user=user,
                    temperature=temperature,
                    top_p=DEFAULT_TOP_P,
                    max_tokens=max_tokens,
                )
            return resp.text or ""
        except OpenRouterError as e:
            logger.error("OpenRouter error: %s", e)
            raise

    # Tasks (text)

    async def classify_text(
        self,
        text: str,
        *,
        labels: Sequence[str],
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Zero-shot / few-shot classification.
        Returns: {"label": str, "scores": {label: float}}
        """
        labels_l = [str(x) for x in labels]
        prompt = (
            "You are a strict JSON classifier. Choose exactly one label from the provided list, "
            "and provide a probability distribution over all labels that sums to 1. "
            "Return JSON with keys: label (string), scores (object). "
            f"Labels: {labels_l}\n\nText:\n{text}"
        )
        msgs = _messages_for_text(system, prompt)
        obj = await self._chat_json(messages=msgs, model=model or self.default_text_model)
        # Minimal sanitation
        if not _maybe_schema_check(obj, ["label", "scores"]):
            return {"label": None, "scores": {}, "_raw": obj, "_parsed": False}
        return obj

    async def extract_entities(
        self,
        text: str,
        *,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Structured IE: pass a simple schema describing required keys.
        Example schema={"vendor": "str", "total": "number", "items": [{"desc":"str","qty":"number"}]}
        """
        prompt = (
            "Extract the requested fields from the text. Return JSON ONLY that matches the keys & shapes requested. "
            "When a value is missing, use null. Do not add extra fields.\n\n"
            f"Schema (keys & types): {json.dumps(schema)}\n\nText:\n{text}"
        )
        msgs = _messages_for_text(system, prompt)
        obj = await self._chat_json(messages=msgs, model=model or self.default_text_model)
        return obj

    async def redact_pii(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns: {"redacted": str, "entities": [{"type": "...", "value": "...", "start": int, "end": int}]}
        """
        prompt = (
            "Identify and redact PII (names, emails, phones, addresses, SSN, tax IDs). "
            "Return JSON with keys: redacted (string), entities (array of {type,value,start,end}). "
            "Redact by replacing characters with '█' of equal length."
        )
        msgs = _messages_for_text(system, prompt)
        return await self._chat_json(messages=msgs, model=model or self.default_text_model)

    async def summarize(
        self,
        text: str,
        *,
        style: str = "concise",
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 600,
    ) -> Dict[str, Any]:
        """
        Returns: {"summary": str, "bullets": [str]}
        """
        prompt = (
            f"Summarize the text in a {style} style. Return strict JSON with keys: summary (string), bullets (string[]). "
            "Avoid hallucinating numbers."
        )
        msgs = _messages_for_text(system, prompt + "\n\nText:\n" + text)
        return await self._chat_json(messages=msgs, model=model or self.default_text_model, max_tokens=max_tokens)

    async def rewrite(
        self,
        text: str,
        *,
        tone: str = "professional",
        length: str = "original",
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns: {"output": str}
        """
        prompt = (
            f"Rewrite the text in a {tone} tone while keeping meaning. Length: {length}. "
            "Return strictly JSON as {\"output\": string}."
        )
        msgs = _messages_for_text(system, prompt + "\n\nText:\n" + text)
        return await self._chat_json(messages=msgs, model=model or self.default_text_model)

    # Tasks (vision)

    async def image_ocr(
        self,
        image: Union[bytes, str],
        *,
        language_hint: Optional[str] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        OCR for a single image. Returns: {"text": str}
        """
        text_hint = f"Language hint: {language_hint}" if language_hint else "Language hint: none"
        prompt = f"Extract all readable text from the image. {text_hint}. Return JSON: {{\"text\": string}}."
        msgs = _messages_for_vision(system, prompt, [image])
        return await self._chat_json(messages=msgs, model=model or self.default_vision_model)

    async def table_from_image(
        self,
        image: Union[bytes, str],
        *,
        columns: Sequence[str],
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract a simple table. Returns: {"rows": [ {<columns...>} ]}
        """
        prompt = (
            "Extract a table from the image. "
            f"Columns (exact order): {list(columns)}. "
            "Return JSON as {\"rows\": [{...}]} with one object per row. "
            "If no table, return {\"rows\": []}."
        )
        msgs = _messages_for_vision(system, prompt, [image])
        obj = await self._chat_json(messages=msgs, model=model or self.default_vision_model)
        if isinstance(obj, dict) and "rows" in obj and isinstance(obj["rows"], list):
            # ensure all keys exist (fill missing with "")
            fixed = []
            for r in obj["rows"]:
                row = {k: (r.get(k, "") if isinstance(r, dict) else "") for k in columns}
                fixed.append(row)
            return {"rows": fixed}
        return {"rows": [], "_raw": obj, "_parsed": False}

    # Streaming example
    async def stream_generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 600,
        on_token=lambda t: print(t, end="", flush=True),
    ) -> None:
        """
        Simple streaming text generator (prints tokens via on_token).
        """
        model = model or self.default_text_model
        provider = _make_provider(False)
        msgs = _messages_for_text(system, prompt)

        async with AsyncOpenRouterLLM.from_env(timeout=60).__aenter__() as client:
            async for chunk in client.stream(
                model=model,
                messages=msgs,
                provider=provider,
                response_format=None,
                temperature=temperature,
                top_p=DEFAULT_TOP_P,
                max_tokens=max_tokens,
                yield_events=False,
                user="slm-stream",
            ):
                if isinstance(chunk, str):
                    on_token(chunk)

# Sync shim
class slm_sync:
    """
    sync wrapper to avoid running async loop - windows issue (sam code)
    """
    _slm = SLM()

    @classmethod
    def classify_text(cls, *args, **kwargs): return asyncio.run(cls._slm.classify_text(*args, **kwargs))
    @classmethod
    def extract_entities(cls, *args, **kwargs): return asyncio.run(cls._slm.extract_entities(*args, **kwargs))
    @classmethod
    def redact_pii(cls, *args, **kwargs): return asyncio.run(cls._slm.redact_pii(*args, **kwargs))
    @classmethod
    def summarize(cls, *args, **kwargs): return asyncio.run(cls._slm.summarize(*args, **kwargs))
    @classmethod
    def rewrite(cls, *args, **kwargs): return asyncio.run(cls._slm.rewrite(*args, **kwargs))
    @classmethod
    def image_ocr(cls, *args, **kwargs): return asyncio.run(cls._slm.image_ocr(*args, **kwargs))
    @classmethod
    def table_from_image(cls, *args, **kwargs): return asyncio.run(cls._slm.table_from_image(*args, **kwargs))
    @classmethod
    def stream_generate(cls, *args, **kwargs): return asyncio.run(cls._slm.stream_generate(*args, **kwargs))

# test
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    async def run():
        slm = SLM()

        # Classification
        out1 = await slm.classify_text(
            "Limited time only-buy now and save 50%!!!",
            labels=["spam", "ham"],
        )
        print("\n\n[CLASSIFY]\n", json.dumps(out1, indent=2))

        # Entity extraction (simple schema)
        schema = {"vendor": "str", "total": "number", "date": "str"}
        out2 = await slm.extract_entities(
            "Invoice from Acme LLC dated 2025-08-01 total $2,340.50 (USD).",
            schema=schema,
        )
        print("\n\n[EXTRACT]\n", json.dumps(out2, indent=2))

        # PII redaction
        out3 = await slm.redact_pii("Email John Doe at john.doe@example.com or call +1 (555) 123-4567.")
        print("\n\n[REDACT]\n", json.dumps(out3, indent=2))

        # Summarize
        out4 = await slm.summarize("This is a long passage about...", style="executive")
        print("\n\n[SUMMARIZE]\n", json.dumps(out4, indent=2))

        # Vision OCR (demo with a tiny 1x1 PNG dot just to exercise the path)
        dot_png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=")
        out5 = await slm.image_ocr(dot_png)
        print("\n\n[OCR]\n", json.dumps(out5, indent=2))

        # Streaming (prints tokens inline)
        print("\n\n[STREAM]")
        await slm.stream_generate("Write a 2-sentence fun fact about otters.")

    try:
        asyncio.run(run())
        return 0
    except OpenRouterError as e:
        print(f"ERROR: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

if __name__ == "__main__":
    main()
