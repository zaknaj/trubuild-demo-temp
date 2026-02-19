
"""
Features
- OpenRouter Chat Completions (unified across providers).
- Timeouts, retries w/ exponential backoff + jitter, and Retry-After support.
- Robust streaming (SSE) that ignores OpenRouter 'comment' lines and yields deltas.
- Flexible model choice + provider routing + transforms + structured outputs.
- Tool calling passthrough (OpenAI-compatible shape).
- List models, credit/key info, basic cost/usage surfaced.
- Minimal deps: `httpx` (recommended for timeouts/streaming). Install: `pip install httpx`.

Environment
- OPENROUTER_API_KEY (required)
- OPENROUTER_APP_URL (optional, sets HTTP-Referer header for attribution)
- OPENROUTER_APP_TITLE (optional, sets X-Title header for attribution)

Usage (sync, non-streaming):
    from llm import OpenRouterLLM, ChatMessage
    client = OpenRouterLLM.from_env()
    resp = client.chat(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=[ChatMessage(role="user", content="Say hello!")],
    )
    print(resp.text)  # "Hello!"

Usage (streaming):
    for token in client.stream(
        model="openai/gpt-4o",
        messages=[ChatMessage(role="user", content="Write a haiku")],
    ):
        print(token, end="", flush=True)

CLI:
    python llm.py -m openai/gpt-4o -p "Quick test" --stream
"""

from __future__ import annotations

import os
import sys
import json
import time
import httpx
import logging
import random
from dataclasses import dataclass, field
from utils.core.log import set_logger, get_logger
from utils.vault import secrets
from typing import Any, Dict, Generator, List, Optional, Union

@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: Union[str, List[Dict[str, Any]]]  # text or [{"type":"image_url","image_url":{"url":...}}]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None  # only for role="tool"


@dataclass
class ProviderPreferences:
    order: Optional[List[str]] = None
    allow_fallbacks: Optional[bool] = None
    require_parameters: Optional[bool] = None
    data_collection: Optional[str] = None  # "allow" | "deny"
    only: Optional[List[str]] = None
    ignore: Optional[List[str]] = None
    quantizations: Optional[List[str]] = None
    sort: Optional[str] = None  # "price" | "throughput" | "latency"
    max_price: Optional[Dict[str, Any]] = None  # per OpenRouter schema


@dataclass
class LLMResponse:
    """Normalized response surface."""
    raw: Dict[str, Any]
    model: str
    created: int
    usage: Optional[Dict[str, int]]
    finish_reason: Optional[str]
    text: str  # best-effort concatenation of the first choice's content


# Exceptions

class OpenRouterError(Exception):
    def __init__(self, status_code: int, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{status_code}] {message}")
        self.status_code = status_code
        self.message = message
        self.metadata = metadata or {}

# Client
@dataclass
class OpenRouterLLM:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float = 60.0  # seconds
    connect_timeout: float = 10.0
    max_retries: int = 4
    backoff_initial: float = 0.7
    backoff_max: float = 15.0
    app_url: Optional[str] = None
    app_title: Optional[str] = None
    default_headers: Dict[str, str] = field(default_factory=dict)
    _client: Optional[httpx.Client] = field(default=None, init=False)
    logger: Any = field(default=None, init=False)


    # Construction
    @classmethod
    def from_env(cls, **kwargs) -> "OpenRouterLLM":
        api_key = secrets.get("OPENROUTER_API_KEY", default="") or ""
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set (set in Vault)")

        set_logger("openrouter_llm")
        instance = cls(api_key=api_key, **kwargs)
        instance.logger = get_logger()
        instance.logger.debug("OpenRouterLLM initialized from environment.")
        return instance

    # Lifecycle
    def __enter__(self) -> "OpenRouterLLM":
        self._client = self._make_client()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def _make_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
            headers=self._build_headers(),
            http2=True,
        )

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional attribution headers (enable rankings/analytics)
        # Docs: HTTP-Referer / X-Title
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-Title"] = self.app_title
        headers.update(self.default_headers)
        return headers

    # Public API

    def list_models(self) -> Dict[str, Any]:
        """GET /models."""
        return self._request("GET", "/models")

    def key_info(self) -> Dict[str, Any]:
        """GET /key — rate/credits info (deprecated rate_limit object safe to ignore)."""
        return self._request("GET", "/key")

    def credits(self) -> Dict[str, Any]:
        """GET /credits — total credits purchased/used."""
        return self._request("GET", "/credits")

    def chat(
        self,
        model: str,
        messages: List[ChatMessage] | None = None,
        prompt: Optional[str] = None,
        stream: bool = False,
        provider: Optional[ProviderPreferences] = None,
        transforms: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        # sampling & advanced params (pass-through)
        **params: Any,
    ) -> LLMResponse:
        """
        Non-streaming chat completion. If `stream=True`, use `stream()` instead (generator).

        Notes:
        - Either `messages` or `prompt` is required.
        - `response_format={"type":"json_object"}` works on OpenAI, Nitro, and some others;
          set provider.require_parameters=True if you need to enforce support.
        """
        if stream:
            raise ValueError("For streaming, use .stream(...)")

        payload = self._build_payload(
            model=model,
            messages=messages,
            prompt=prompt,
            provider=provider,
            transforms=transforms,
            response_format=response_format,
            user=user,
            stream=False,
            **params,
        )
        data = self._request_json_with_retries("POST", "/chat/completions", json=payload)
        return self._to_llm_response(data)

    def stream(
        self,
        model: str,
        messages: List[ChatMessage] | None = None,
        prompt: Optional[str] = None,
        provider: Optional[ProviderPreferences] = None,
        transforms: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        yield_events: bool = False,
        **params: Any,
    ) -> Generator[str | Dict[str, Any], None, None]:
        """
        Streaming generator (SSE). Yields text deltas by default, or full SSE events if `yield_events=True`.

        SSE quirks:
        - OpenRouter occasionally sends comment lines (e.g., ': OPENROUTER PROCESSING'); ignore those.
        - Final event may include usage with empty choices.
        """
        payload = self._build_payload(
            model=model,
            messages=messages,
            prompt=prompt,
            provider=provider,
            transforms=transforms,
            response_format=response_format,
            user=user,
            stream=True,
            **params,
        )

        # Best-effort single retry before streaming starts (can't safely retry mid-stream)
        attempt = 0
        while True:
            attempt += 1
            try:
                with self._stream_request("POST", "/chat/completions", json=payload) as r:
                    if r.status_code != 200:
                        # Error body is JSON per docs; raise with details.
                        err = _read_error_safely(r)
                        raise OpenRouterError(err.get("code", r.status_code), err.get("message", "Stream error"), err.get("metadata"))

                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            s = line.decode("utf-8", errors="ignore")
                        except AttributeError:
                            s = str(line)
                        s = s.strip()
                        # Comments (ignore)
                        if s.startswith(":"):
                            continue
                        if not s.startswith("data: "):
                            continue
                        data = s[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            evt = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if yield_events:
                            yield evt
                        else:
                            # Best effort: yield any assistant delta content
                            choices = evt.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta") or {}
                                content = delta.get("content")
                                if content:
                                    yield content
                    break  # streamed successfully
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                if attempt <= self.max_retries:
                    self._sleep_backoff(attempt, retry_after=None)
                    continue
                raise OpenRouterError(408, f"Streaming connection error: {e}") from e
            except OpenRouterError as e:
                if self._should_retry_status(e.status_code) and attempt <= self.max_retries:
                    self._sleep_backoff(attempt, retry_after=None)
                    continue
                raise

    # Internals

    def _build_payload(
        self,
        *,
        model: str,
        messages: Optional[List[ChatMessage]],
        prompt: Optional[str],
        provider: Optional[ProviderPreferences],
        transforms: Optional[List[str]],
        response_format: Optional[Dict[str, Any]],
        user: Optional[str],
        stream: bool,
        **params: Any,
    ) -> Dict[str, Any]:
        if not messages and not prompt:
            raise ValueError("Either `messages` or `prompt` must be provided.")

        # Convert dataclasses to dicts where needed
        payload: Dict[str, Any] = {
            "model": model,
            "stream": stream,
        }
        if messages:
            payload["messages"] = [m.__dict__ for m in messages]
        if prompt:
            payload["prompt"] = prompt
        if provider:
            payload["provider"] = {k: v for k, v in provider.__dict__.items() if v is not None}
        if transforms is not None:
            payload["transforms"] = transforms  # e.g. ["middle-out"]
        if response_format is not None:
            payload["response_format"] = response_format
        if user:
            payload["user"] = user  # stable end-user identifier

        # Pass through model params (temperature, top_p, max_tokens, tools, etc.)
        payload.update(params)
        return payload

    def _to_llm_response(self, data: Dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        usage = data.get("usage")
        model = data.get("model", "")
        created = data.get("created", 0)

        text = ""
        finish_reason = None
        if choices:
            c0 = choices[0]
            msg = c0.get("message") or {}
            text = (msg.get("content") or "") if isinstance(msg, dict) else ""
            finish_reason = c0.get("finish_reason")

        return LLMResponse(
            raw=data,
            model=model,
            created=created,
            usage=usage,
            finish_reason=finish_reason,
            text=text or "",
        )

    # Core request with retry
    def _request_json_with_retries(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        attempt = 0
        last_error: Optional[OpenRouterError] = None

        while True:
            attempt += 1
            try:
                resp = self._request(method, path, **kwargs)
                if resp.status_code == 200:
                    return resp.json()
                # OpenRouter returns non-200 for validation/auth/billing etc; parse error object
                err = _read_error_safely(resp)
                raise OpenRouterError(err.get("code", resp.status_code), err.get("message", "Request error"), err.get("metadata"))
            except OpenRouterError as e:
                last_error = e
                # Retry on 408/429/5xx; don't retry on 4xx like 401/402/403/400
                if self._should_retry_status(e.status_code) and attempt <= self.max_retries:
                    retry_after = None
                    if hasattr(e, "metadata"):
                        # If upstream bubbled Retry-After into metadata, respect it
                        retry_after = _parse_retry_after(e.metadata.get("retry_after")) if isinstance(e.metadata, dict) else None
                    self._sleep_backoff(attempt, retry_after)
                    continue
                raise
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                if attempt <= self.max_retries:
                    self._sleep_backoff(attempt, retry_after=None)
                    continue
                raise OpenRouterError(408, f"Network error: {e}") from e

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        client = self._client or self._make_client()
        # Using absolute path to avoid double base_url if a full URL is passed
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        return client.request(method, url, **kwargs)

    def _stream_request(self, method: str, path: str, **kwargs):
        client = self._client or self._make_client()
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        return client.stream(method, url, **kwargs)

    def _sleep_backoff(self, attempt: int, retry_after: Optional[float]) -> None:
        if retry_after is not None and retry_after >= 0:
            delay = float(retry_after)
        else:
            # Exponential backoff with full jitter
            base = self.backoff_initial * (2 ** (attempt - 1))
            delay = min(base, self.backoff_max) * random.uniform(0.5, 1.5)
        time.sleep(delay)

    @staticmethod
    def _should_retry_status(status: int) -> bool:
        # 408 timeout, 429 rate limit, 502 model down, 503 no available provider
        if status in (408, 429, 502, 503):
            return True
        # Be conservative, also retry generic 500/504 if seen
        if status in (500, 504):
            return True
        return False


@dataclass
class AsyncOpenRouterLLM:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float = 60.0
    connect_timeout: float = 10.0
    max_retries: int = 4
    backoff_initial: float = 0.7
    backoff_max: float = 15.0
    app_url: Optional[str] = None
    app_title: Optional[str] = None
    default_headers: Dict[str, str] = field(default_factory=dict)
    _client: Optional[httpx.AsyncClient] = field(default=None, init=False)

    # Construction
    @classmethod
    def from_env(cls, **kwargs) -> "AsyncOpenRouterLLM":
        api_key = secrets.get("OPENROUTER_API_KEY", default="") or ""
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set (set in Vault)")
        return cls(api_key=api_key, **kwargs)

    # Lifecycle
    async def __aenter__(self) -> "AsyncOpenRouterLLM":
        self._client = self._make_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
            headers=self._build_headers(),
            http2=True,
        )

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-Title"] = self.app_title
        headers.update(self.default_headers)
        return headers

    # Public API (async)

    async def list_models(self) -> Dict[str, Any]:
        r = await self._request("GET", "/models")
        return r.json()

    async def key_info(self) -> Dict[str, Any]:
        r = await self._request("GET", "/key")
        return r.json()

    async def credits(self) -> Dict[str, Any]:
        r = await self._request("GET", "/credits")
        return r.json()

    async def chat(
        self,
        model: str,
        messages: List[ChatMessage] | None = None,
        prompt: Optional[str] = None,
        provider: Optional[ProviderPreferences] = None,
        transforms: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        **params: Any,
    ) -> LLMResponse:
        payload = self._build_payload(
            model=model,
            messages=messages,
            prompt=prompt,
            provider=provider,
            transforms=transforms,
            response_format=response_format,
            user=user,
            stream=False,
            **params,
        )
        data = await self._request_json_with_retries("POST", "/chat/completions", json=payload)
        # Reuse the same normalizer from the sync class:
        return OpenRouterLLM._to_llm_response(self, data)  # uses identical logic

    async def stream(
        self,
        model: str,
        messages: List[ChatMessage] | None = None,
        prompt: Optional[str] = None,
        provider: Optional[ProviderPreferences] = None,
        transforms: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        yield_events: bool = False,
        **params: Any,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        payload = self._build_payload(
            model=model,
            messages=messages,
            prompt=prompt,
            provider=provider,
            transforms=transforms,
            response_format=response_format,
            user=user,
            stream=True,
            **params,
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._stream_request("POST", "/chat/completions", json=payload) as r:
                    if r.status_code != 200:
                        err = _read_error_safely(r)
                        raise OpenRouterError(err.get("code", r.status_code), err.get("message", "Stream error"), err.get("metadata"))
                    async for s in r.aiter_lines():
                        if not s:
                            continue
                        s = s.strip()
                        if s.startswith(":"):
                            continue
                        if not s.startswith("data: "):
                            continue
                        data = s[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            evt = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if yield_events:
                            yield evt
                        else:
                            choices = evt.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta") or {}
                                content = delta.get("content")
                                if content:
                                    yield content
                break
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                if attempt <= self.max_retries:
                    await self._sleep_backoff(attempt, retry_after=None)
                    continue
                raise OpenRouterError(408, f"Streaming connection error: {e}") from e
            except OpenRouterError as e:
                if self._should_retry_status(e.status_code) and attempt <= self.max_retries:
                    await self._sleep_backoff(attempt, retry_after=None)
                    continue
                raise

    def _build_payload(
        self,
        *,
        model: str,
        messages: Optional[List[ChatMessage]],
        prompt: Optional[str],
        provider: Optional[ProviderPreferences],
        transforms: Optional[List[str]],
        response_format: Optional[Dict[str, Any]],
        user: Optional[str],
        stream: bool,
        **params: Any,
    ) -> Dict[str, Any]:
        if not messages and not prompt:
            raise ValueError("Either `messages` or `prompt` must be provided.")
        payload: Dict[str, Any] = {"model": model, "stream": stream}
        if messages:
            payload["messages"] = [m.__dict__ for m in messages]
        if prompt:
            payload["prompt"] = prompt
        if provider:
            payload["provider"] = {k: v for k, v in provider.__dict__.items() if v is not None}
        if transforms is not None:
            payload["transforms"] = transforms
        if response_format is not None:
            payload["response_format"] = response_format
        if user:
            payload["user"] = user
        payload.update(params)
        return payload

    async def _request_json_with_retries(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        attempt = 0
        last_error: Optional[OpenRouterError] = None

        while True:
            attempt += 1
            try:
                resp = await self._request(method, path, **kwargs)
                if resp.status_code == 200:
                    return resp.json()
                err = _read_error_safely(resp)
                raise OpenRouterError(err.get("code", resp.status_code), err.get("message", "Request error"), err.get("metadata"))
            except OpenRouterError as e:
                last_error = e
                if self._should_retry_status(e.status_code) and attempt <= self.max_retries:
                    retry_after = None
                    if isinstance(e.metadata, dict):
                        retry_after = _parse_retry_after(e.metadata.get("retry_after"))
                    await self._sleep_backoff(attempt, retry_after)
                    continue
                raise
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                if attempt <= self.max_retries:
                    await self._sleep_backoff(attempt, retry_after=None)
                    continue
                raise OpenRouterError(408, f"Network error: {e}") from e

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        client = self._client or self._make_client()
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        return await client.request(method, url, **kwargs)

    def _stream_request(self, method: str, path: str, **kwargs):
        client = self._client or self._make_client()
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        return client.stream(method, url, **kwargs)

    async def _sleep_backoff(self, attempt: int, retry_after: Optional[float]) -> None:
        if retry_after is not None and retry_after >= 0:
            delay = float(retry_after)
        else:
            base = self.backoff_initial * (2 ** (attempt - 1))
            delay = min(base, self.backoff_max) * random.uniform(0.5, 1.5)
        await asyncio.sleep(delay)

    @staticmethod
    def _should_retry_status(status: int) -> bool:
        return OpenRouterLLM._should_retry_status(status)

# Helpers
def _read_error_safely(resp: httpx.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
    except Exception:
        return {"code": resp.status_code, "message": f"HTTP {resp.status_code}"}
    err = data.get("error") or {}
    return {
        "code": err.get("code", resp.status_code),
        "message": err.get("message") or err.get("error") or f"HTTP {resp.status_code}",
        "metadata": err.get("metadata") or {},
    }

def _parse_retry_after(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None

def main() -> int:
    """
    Single entrypoint with a toggle for async vs sync.
    it requyires OPENROUTER_API_KEY in the environment.
    """
    import asyncio
    import sys
    import json

    USE_ASYNC   = True            # True => AsyncOpenRouterLLM, False => OpenRouterLLM
    MODEL       = "openai/gpt-5-mini"
    STREAM      = True            # True to stream tokens, False for a single final response
    PROMPT      = "Say hello from the unified main!"
    MESSAGES    = None            # Or a list of dicts/ChatMessage: [{"role":"user","content":"Write a haiku"}]

    TEMPERATURE = 0.7
    MAX_TOKENS  = 256
    JSON_OUTPUT = False           # Print the full JSON response instead of just the text
    REQUIRE_JSON = False          # Enforce JSON output (response_format + provider.require_parameters)
    PROVIDER_ORDER = None         # e.g., "openai,together,nitro"
    SORT = None                   # One of: "price", "throughput", "latency"
    USER = "cli-demo-user"        # Stable end-user identifier

    try:
        provider = None
        if PROVIDER_ORDER or SORT:
            provider = ProviderPreferences(
                order=[x.strip() for x in PROVIDER_ORDER.split(",")] if PROVIDER_ORDER else None,
                sort=SORT,
                require_parameters=True if REQUIRE_JSON else None,
            )

        response_format = {"type": "json_object"} if REQUIRE_JSON else None

        # Prepare messages
        messages = None
        if MESSAGES:
            if isinstance(MESSAGES, list) and MESSAGES and isinstance(MESSAGES[0], dict):
                messages = [ChatMessage(**m) for m in MESSAGES]
            else:
                messages = MESSAGES
        elif PROMPT:
            messages = [ChatMessage(role="user", content=PROMPT)]
        else:
            raise ValueError("Either PROMPT or MESSAGES must be set.")

        if USE_ASYNC:
            async def run_async() -> int:
                client = AsyncOpenRouterLLM.from_env()
                async with client:
                    if STREAM:
                        async for chunk in client.stream(
                            model=MODEL,
                            messages=messages,
                            response_format=response_format,
                            provider=provider,
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS,
                            user=USER,
                        ):
                            print(chunk, end="", flush=True)
                        print()
                        return 0
                    resp = await client.chat(
                        model=MODEL,
                        messages=messages,
                        response_format=response_format,
                        provider=provider,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        user=USER,
                    )
                    if JSON_OUTPUT:
                        print(json.dumps(resp.raw, indent=2))
                    else:
                        print(resp.text)
                    return 0

            return asyncio.run(run_async())

        client = OpenRouterLLM.from_env()
        if STREAM:
            for chunk in client.stream(
                model=MODEL,
                messages=messages,
                response_format=response_format,
                provider=provider,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                user=USER,
            ):
                print(chunk, end="", flush=True)
            print()
            return 0

        resp = client.chat(
            model=MODEL,
            messages=messages,
            response_format=response_format,
            provider=provider,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            user=USER,
        )
        if JSON_OUTPUT:
            print(json.dumps(resp.raw, indent=2))
        else:
            print(resp.text)
        return 0

    except OpenRouterError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    import logging, sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
