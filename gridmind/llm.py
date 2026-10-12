"""Thin Claude client: structured (Pydantic) outputs, disk cache keyed by request hash, and
graceful degradation. Every caller has a deterministic fallback, so the system runs (and the
evaluation is reproducible) with no API key at all.

Modes (env GRIDMIND_LLM):
  auto    (default) use Claude if ANTHROPIC_API_KEY is set, else run offline
  off     never call the API (deterministic agents only)
  replay  only serve cached responses (deterministic demo recordings)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from .config import ROOT

MODEL = os.environ.get("GRIDMIND_MODEL", "claude-opus-5-5")
CACHE_DIR = ROOT / "cache" / "llm"
T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult:
    output: BaseModel | None
    source: str          # "claude" | "cache" | "offline" | "error"
    latency_ms: float = 0.0
    error: str = ""


class LLM:
    def __init__(self, mode: str | None = None, model: str = MODEL):
        self.mode = mode or os.environ.get("GRIDMIND_LLM", "auto")
        self.model = model
        self._client = None
        if self.mode == "auto" and not os.environ.get("ANTHROPIC_API_KEY"):
            self.mode = "off"

    @property
    def online(self) -> bool:
        return self.mode in ("auto", "on")

    @property
    def available(self) -> bool:
        return self.mode != "off"

    def _client_or_none(self):
        if self._client is None and self.online:
            import anthropic
            self._client = anthropic.Anthropic(max_retries=2, timeout=60)
        return self._client

    def structured(self, system: str, content: list[dict] | str, schema: type[T], effort: str = "low") -> LLMResult:
        if self.mode == "off":
            return LLMResult(None, "offline")
        key = _hash(self.model, system, content, schema.__name__)
        cached = CACHE_DIR / f"{key}.json"
        if cached.exists():
            return LLMResult(schema.model_validate_json(cached.read_text()), "cache")
        if self.mode == "replay":
            return LLMResult(None, "offline", error="no cached response")

        import anthropic
        client = self._client_or_none()
        messages = [{"role": "user", "content": content}]
        t0 = time.perf_counter()
        try:
            resp = client.beta.messages.parse(
                model=self.model,
                max_tokens=8000,
                system=system,
                messages=messages,
                output_format=schema,
                output_config={"effort": effort},
                # Server-side refusal fallback: if the model declines, the API re-runs the request
                # on a fallback model inside the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            return LLMResult(None, "error", error=f"rate limited: {e}")
        except anthropic.APIStatusError as e:
            return LLMResult(None, "error", error=f"API {e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            return LLMResult(None, "error", error=f"connection: {e}")
        except Exception as e:  # malformed structured output and similar -> fallback path
            return LLMResult(None, "error", error=f"{type(e).__name__}: {e}")
        latency = (time.perf_counter() - t0) * 1000
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return LLMResult(None, "error", latency, error=f"no parsed output (stop_reason={resp.stop_reason})")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(resp.parsed_output.model_dump_json(indent=2))
        return LLMResult(resp.parsed_output, "claude", latency)


def _hash(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:24]
