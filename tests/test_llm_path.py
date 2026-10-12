"""The Claude path, exercised with a stub client (no network): acceptance, guardrail retry, fallback,
and the response cache."""
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
        super().__init__(mode="on")
        self.outputs, self.calls = list(outputs), 0

    def structured(self, system, content, schema, effort="low"):
        self.calls += 1
        out = self.outputs.pop(0)
        return LLMResult(out, "claude") if out is not None else LLMResult(None, "error", error="timeout")


def test_valid_llm_decision_is_used():
    s = Strategist(P, StubLLM([_decision()]))
    d, meta = s.decide(SIT)
    assert meta["source"] == "claude" and d.banned_actions == ["sell_to_grid"]


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


def test_llm_client_parses_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path)
    resp = SimpleNamespace(stop_reason="end_turn", parsed_output=_decision())
    calls = []

    class FakeMessages:
        def parse(self, **kw):
            calls.append(kw)
            return resp

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    llm = LLM(mode="on")
    llm._client = client
    r1 = llm.structured("sys", "payload", StrategyDecision)
    r2 = llm.structured("sys", "payload", StrategyDecision)
    assert r1.source == "claude" and r2.source == "cache" and len(calls) == 1
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_format"] is StrategyDecision
    assert kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"]


def test_refusal_is_treated_as_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path)
    resp = SimpleNamespace(stop_reason="refusal", parsed_output=None)
    llm = LLM(mode="on")
    llm._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=lambda **kw: resp)))
    r = llm.structured("sys", "x", StrategyDecision)
    assert r.output is None and r.source == "error"
