"""The LLM path (Gemini and Claude), exercised with stub clients (no network): acceptance, guardrail
retry, fallback, provider selection and the response cache."""
import os

os.environ["GRIDMIND_LLM"] = "off"

from types import SimpleNamespace

from gridmind import llm as llm_mod
from gridmind.config import load_portfolio
from gridmind.llm import LLM, LLMResult
from gridmind.strategist import Reserve, Situation_, StrategyDecision, Strategist, Weights

P = load_portfolio()
SIT = Situation_("storm", {"active_alerts": ["storm_alert"]}, ["storm alert"])


def _decision(rel=0.5, total=1.0):
    other = (total - rel) / 4
    return StrategyDecision(situation_label="storm",
                            weights=Weights(cost=other, carbon=other, curtailment=other, degradation=other, reliability=rel),
                            battery_reserves=[Reserve(battery_id=b.id, reserve_pct=40) for b in P.batteries],
                            banned_actions=["sell_to_grid"], request_human_approval=["trigger_demand_response"],
                            rationale="Hold reserve through the storm.", confidence=0.8)


class StubLLM(LLM):
    def __init__(self, outputs):
        super().__init__(mode="on", provider="gemini")
        self.outputs, self.calls = list(outputs), 0

    def structured(self, system, text, schema, pdf_bytes=None):
        self.calls += 1
        out = self.outputs.pop(0)
        return LLMResult(out, "gemini") if out is not None else LLMResult(None, "error", error="timeout")


def test_valid_llm_decision_is_used():
    s = Strategist(P, StubLLM([_decision()]))
    d, meta = s.decide(SIT)
    assert meta["source"] == "gemini" and d.banned_actions == ["sell_to_grid"]


def test_unsafe_llm_decision_retried_then_falls_back():
    stub = StubLLM([_decision(rel=0.0), _decision(total=1.6)])
    d, meta = Strategist(P, stub).decide(SIT)
    assert stub.calls == 2 and meta["source"] == "rules (fallback)" and len(meta["rejections"]) == 2
    assert d.weights.reliability >= 0.05


def test_llm_outage_falls_back():
    d, meta = Strategist(P, StubLLM([None])).decide(SIT)
    assert meta["source"] == "rules (fallback)"


def test_situation_cache_avoids_repeat_calls():
    stub = StubLLM([_decision()])
    s = Strategist(P, stub)
    s.decide(SIT)
    _, meta = s.decide(SIT)
    assert stub.calls == 1 and meta["reused"]


def test_anthropic_client_parses_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path)
    resp = SimpleNamespace(stop_reason="end_turn", parsed_output=_decision())
    calls = []

    class FakeMessages:
        def parse(self, **kw):
            calls.append(kw)
            return resp

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    llm = LLM(mode="on", provider="anthropic")
    llm._client = client
    r1 = llm.structured("sys", "payload", StrategyDecision)
    r2 = llm.structured("sys", "payload", StrategyDecision)
    assert r1.source == "anthropic" and r2.source == "cache" and len(calls) == 1
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_format"] is StrategyDecision
    assert kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"]


def test_refusal_is_treated_as_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path)
    resp = SimpleNamespace(stop_reason="refusal", parsed_output=None)
    llm = LLM(mode="on", provider="anthropic")
    llm._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=lambda **kw: resp)))
    r = llm.structured("sys", "x", StrategyDecision)
    assert r.output is None and r.source == "error"


# ------------------------------------------------------------------------------------------ Gemini
def _gemini_llm(resp, calls, monkeypatch, tmp_path):
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path)

    def generate_content(**kw):
        calls.append(kw)
        if isinstance(resp, Exception):
            raise resp
        return resp

    llm = LLM(mode="on", provider="gemini")
    llm._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return llm


def test_gemini_parses_attaches_pdf_and_caches(tmp_path, monkeypatch):
    calls = []
    llm = _gemini_llm(SimpleNamespace(parsed=_decision(), text=""), calls, monkeypatch, tmp_path)
    r1 = llm.structured("sys", "payload", StrategyDecision, pdf_bytes=b"%PDF-1.4 fake")
    r2 = llm.structured("sys", "payload", StrategyDecision, pdf_bytes=b"%PDF-1.4 fake")
    assert r1.source == "gemini" and r2.source == "cache" and len(calls) == 1
    kw = calls[0]
    assert kw["model"] == "gemini-3.5-flash"
    assert kw["config"].response_schema is StrategyDecision
    assert kw["config"].response_mime_type == "application/json"
    assert kw["contents"][0].inline_data.mime_type == "application/pdf"


def test_gemini_raw_json_is_validated_when_sdk_cannot_parse(tmp_path, monkeypatch):
    raw = _decision().model_dump_json()
    llm = _gemini_llm(SimpleNamespace(parsed=None, text=raw), [], monkeypatch, tmp_path)
    assert llm.structured("sys", "x", StrategyDecision).output.situation_label == "storm"


def test_gemini_garbage_and_api_errors_fall_back(tmp_path, monkeypatch):
    from google.genai import errors

    llm = _gemini_llm(SimpleNamespace(parsed=None, text="not json"), [], monkeypatch, tmp_path)
    assert llm.structured("sys", "x", StrategyDecision).output is None
    err = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    llm = _gemini_llm(err, [], monkeypatch, tmp_path)
    r = llm.structured("sys", "y", StrategyDecision)
    assert r.output is None and "429" in r.error


def test_provider_selection(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "GRIDMIND_PROVIDER", "GEMINI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert LLM(mode="auto").mode == "off"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert LLM(mode="auto").provider == "anthropic"
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    llm = LLM(mode="auto")
    assert llm.provider == "gemini" and llm.model == "gemini-3.5-flash" and llm.mode == "auto"
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    assert LLM(mode="auto").model == "gemini-3.6-flash"
