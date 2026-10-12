"""Thin LLM client for the agents: structured (Pydantic) outputs, disk cache keyed by request hash,
and graceful degradation. Every caller has a deterministic fallback, so the system runs (and the
evaluation is reproducible) with no API key at all.

Providers (env GRIDMIND_PROVIDER):
  auto       (default) Gemini if GEMINI_API_KEY / GOOGLE_API_KEY is set, else Claude if
             ANTHROPIC_API_KEY is set, else offline
  gemini     Google Gemini (default model gemini-3.5-flash, override with GEMINI_MODEL)
  anthropic  Claude (default model claude-opus-5-5, override with ANTHROPIC_MODEL)

Modes (env GRIDMIND_LLM):
  auto    (default) call the provider when a key is configured, else run offline
  off     never call an API (deterministic agents only)
  replay  only serve cached responses (deterministic demo recordings)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .config import ROOT

DEFAULT_MODELS = {"gemini": "gemini-3.5-flash", "anthropic": "claude-opus-5-5"}
CACHE_DIR = ROOT / "cache" / "llm"
T = TypeVar("T", bound=BaseModel)


def _gemini_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def resolve_provider(requested: str | None = None) -> str | None:
    p = (requested or os.environ.get("GRIDMIND_PROVIDER", "auto")).lower()
    if p in ("gemini", "anthropic"):
        return p
    if _gemini_key():
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


@dataclass
class LLMResult:
    output: BaseModel | None
    source: str          # provider name ("gemini" | "anthropic") | "cache" | "offline" | "error"
    latency_ms: float = 0.0
    error: str = ""


class LLM:
    def __init__(self, mode: str | None = None, provider: str | None = None, model: str | None = None):
        self.mode = mode or os.environ.get("GRIDMIND_LLM", "auto")
        self.provider = resolve_provider(provider) or "gemini"
        env_model = os.environ.get("GEMINI_MODEL" if self.provider == "gemini" else "ANTHROPIC_MODEL")
        self.model = model or env_model or DEFAULT_MODELS[self.provider]
        self._client = None
        if self.mode == "auto" and resolve_provider(provider) is None:
            self.mode = "off"

    @property
    def online(self) -> bool:
        return self.mode in ("auto", "on")

    @property
    def available(self) -> bool:
        return self.mode != "off"

    @property
    def label(self) -> str:
        name = {"gemini": "Gemini", "anthropic": "Claude"}[self.provider]
        return {"auto": f"{name} live ({self.model})", "on": f"{name} live ({self.model})",
                "replay": f"{name} cached replay ({self.model})", "off": "Offline · deterministic agents"}[self.mode]

    def _client_or_none(self):
        if self._client is None and self.online:
            if self.provider == "gemini":
                from google import genai
                self._client = genai.Client()           # reads GEMINI_API_KEY / GOOGLE_API_KEY
            else:
                import anthropic
                self._client = anthropic.Anthropic(max_retries=2, timeout=60)
        return self._client

    def structured(self, system: str, text: str, schema: type[T], pdf_bytes: bytes | None = None) -> LLMResult:
        """One structured call. `text` is the user content; `pdf_bytes` optionally attaches a PDF."""
        if self.mode == "off":
            return LLMResult(None, "offline")
        pdf_digest = hashlib.sha256(pdf_bytes).hexdigest() if pdf_bytes else None
        key = _hash(self.provider, self.model, system, text, pdf_digest, schema.__name__)
        cached = CACHE_DIR / f"{key}.json"
        if cached.exists():
            return LLMResult(schema.model_validate_json(cached.read_text()), "cache")
        if self.mode == "replay":
            return LLMResult(None, "offline", error="no cached response")

        t0 = time.perf_counter()
        call = self._call_gemini if self.provider == "gemini" else self._call_anthropic
        output, error = call(system, text, schema, pdf_bytes)
        latency = (time.perf_counter() - t0) * 1000
        if output is None:
            return LLMResult(None, "error", latency, error=error)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(output.model_dump_json(indent=2))
        return LLMResult(output, self.provider, latency)

    # ------------------------------------------------------------------ providers
    def _call_gemini(self, system, text, schema, pdf_bytes):
        from google.genai import errors, types

        parts = []
        if pdf_bytes:
            parts.append(types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))
        parts.append(text)
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=os.environ.get("GEMINI_THINKING_LEVEL", "low")),
        )
        try:
            resp = self._client_or_none().models.generate_content(model=self.model, contents=parts, config=config)
        except errors.ClientError as e:      # 4xx, including 429 rate limits and bad keys
            return None, f"Gemini {e.code}: {e.message}"
        except errors.ServerError as e:
            return None, f"Gemini server {e.code}: {e.message}"
        except errors.APIError as e:
            return None, f"Gemini API {e.code}: {e.message}"
        except Exception as e:               # network and other transport failures -> fallback path
            return None, f"{type(e).__name__}: {e}"
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, schema):
            return parsed, ""
        try:                                 # SDK could not coerce: validate the raw JSON ourselves
            return schema.model_validate_json(resp.text or ""), ""
        except (ValidationError, ValueError) as e:
            return None, f"Gemini returned no valid {schema.__name__}: {e.__class__.__name__}"

    def _call_anthropic(self, system, text, schema, pdf_bytes):
        import base64

        import anthropic

        content = text
        if pdf_bytes:
            content = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                       "data": base64.b64encode(pdf_bytes).decode()}},
                       {"type": "text", "text": text}]
        try:
            resp = self._client_or_none().beta.messages.parse(
                model=self.model,
                max_tokens=8000,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_format=schema,
                output_config={"effort": "low"},
                # Server-side refusal fallback: if the model declines, the API re-runs the request
                # on a fallback model inside the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            return None, f"rate limited: {e}"
        except anthropic.APIStatusError as e:
            return None, f"API {e.status_code}: {e.message}"
        except anthropic.APIConnectionError as e:
            return None, f"connection: {e}"
        except Exception as e:  # malformed structured output and similar -> fallback path
            return None, f"{type(e).__name__}: {e}"
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return None, f"no parsed output (stop_reason={resp.stop_reason})"
        return resp.parsed_output, ""


def _hash(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:24]
