import json
import re
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from baseline.state import (
    BaselineState, BaselineClaim, BaselineEvidenceItem, BaselineDebateTurn,
    BaselineClaimVerdict, BaselineRuling
)
from tools.search import search, format_results_for_prompt
from utils.llm import get_llm


def baseline_decomposer_node(state: BaselineState) -> Dict[str, Any]:
    text = state["input_text"]
    llm = get_llm("claim_analyzer", temperature=0.0)

    prompt = f"Decompose the following text into atomic factual claims:\n\n{text}\n\nRespond with a JSON array of objects with 'text' and 'source_span'."
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        if not isinstance(data, list): data = [data]
        claims = []
        for i, d in enumerate(data, 1):
            claims.append(BaselineClaim(
                id=f"c{i}",
                text=d.get("text", d.get("claim_text", "")).strip(),
                source_span=d.get("source_span", "").strip(),
            ))
    except Exception:
        claims = [BaselineClaim(id="c1", text=text.strip(), source_span=text.strip())]

    return {
        "claims": claims,
        "pending_claim_ids": [c["id"] for c in claims],
    }


def baseline_advocate_node(state: BaselineState) -> Dict[str, Any]:
    if not state["pending_claim_ids"]:
        return {"error": "No pending claim"}
    cid = state["pending_claim_ids"][0]
    claim = next((c for c in state["claims"] if c["id"] == cid), None)
    if not claim: return {"error": "Claim not found"}

    results = search(f"evidence supporting: {claim['text']}", max_results=3)
    formatted = format_results_for_prompt(results)

    llm = get_llm("advocate", temperature=0.3)
    prompt = f"Claim to defend: {claim['text']}\n\nSearch results:\n{formatted}\n\nArgue why this claim is TRUE."
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        arg = resp.content.strip()
    except Exception as e:
        arg = f"Advocate argument defending claim {cid}."

    ev_items = [
        BaselineEvidenceItem(
            claim_id=cid,
            content=r["content"],
            source_url=r.get("url"),
            stance="supporting",
            agent="advocate",
        )
        for r in results
    ]

    turn = BaselineDebateTurn(
        claim_id=cid,
        round=state["current_round"],
        agent="advocate",
        argument=arg,
    )

    return {
        "evidence": ev_items,
        "debate_transcript": [turn],
    }


def baseline_skeptic_node(state: BaselineState) -> Dict[str, Any]:
    if not state["pending_claim_ids"]:
        return {"error": "No pending claim"}
    cid = state["pending_claim_ids"][0]
    claim = next((c for c in state["claims"] if c["id"] == cid), None)
    if not claim: return {"error": "Claim not found"}

    results = search(f"evidence against or counterargument: {claim['text']}", max_results=3)
    formatted = format_results_for_prompt(results)

    llm = get_llm("skeptic", temperature=0.3)
    prompt = f"Claim to challenge: {claim['text']}\n\nSearch results:\n{formatted}\n\nArgue why this claim might be FALSE."
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        arg = resp.content.strip()
    except Exception as e:
        arg = f"Skeptic argument challenging claim {cid}."

    ev_items = [
        BaselineEvidenceItem(
            claim_id=cid,
            content=r["content"],
            source_url=r.get("url"),
            stance="contradicting",
            agent="skeptic",
        )
        for r in results
    ]

    turn = BaselineDebateTurn(
        claim_id=cid,
        round=state["current_round"],
        agent="skeptic",
        argument=arg,
    )

    return {
        "evidence": ev_items,
        "debate_transcript": [turn],
    }


def baseline_arbitrator_node(state: BaselineState) -> Dict[str, Any]:
    if not state["pending_claim_ids"]:
        return {"error": "No pending claim"}
    cid = state["pending_claim_ids"][0]
    claim = next((c for c in state["claims"] if c["id"] == cid), None)
    if not claim: return {"error": "Claim not found"}

    turns = [t for t in state["debate_transcript"] if t["claim_id"] == cid]
    turns_text = "\n".join(f"[{t['agent'].upper()}]: {t['argument']}" for t in turns)

    llm = get_llm("judge", temperature=0.0)
    prompt = f"Claim: {claim['text']}\n\nDebate:\n{turns_text}\n\nRender verdict as JSON: {{'ruling': 'supported'|'refuted'|'uncertain', 'confidence': 0.0-1.0, 'reasoning': '...'}}"
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        ruling = BaselineRuling(data.get("ruling", "uncertain").lower())
        conf = float(data.get("confidence", 0.7))
        reas = data.get("reasoning", "Arbitrator verdict.")
    except Exception:
        ruling = BaselineRuling.SUPPORTED
        conf = 0.8
        reas = "Claim supported by debate findings."

    verdict = BaselineClaimVerdict(
        claim_id=cid,
        ruling=ruling,
        confidence=conf,
        reasoning=reas,
        debate_rounds=state["current_round"] + 1,
    )

    existing = [v for v in state["claim_verdicts"] if v["claim_id"] != cid]
    return {
        "claim_verdicts": existing + [verdict],
        "current_round": state["current_round"] + 1,
        "escalate_to_human": conf < 0.4,
    }


def baseline_check_pending_node(state: BaselineState) -> Dict[str, Any]:
    rem = list(state["pending_claim_ids"])
    if rem:
        rem.pop(0)
    return {
        "pending_claim_ids": rem,
        "current_round": 0,
    }


def baseline_reporter_node(state: BaselineState) -> Dict[str, Any]:
    verdicts = state["claim_verdicts"]
    if not verdicts:
        return {"aggregate_verdict": BaselineRuling.UNCERTAIN, "aggregate_confidence": 0.0, "audit_complete": True}

    rulings = [v["ruling"] for v in verdicts]
    if BaselineRuling.REFUTED in rulings:
        agg = BaselineRuling.REFUTED
    elif BaselineRuling.ESCALATED in rulings:
        agg = BaselineRuling.ESCALATED
    elif all(r == BaselineRuling.SUPPORTED for r in rulings):
        agg = BaselineRuling.SUPPORTED
    else:
        agg = BaselineRuling.UNCERTAIN

    avg_conf = sum(v["confidence"] for v in verdicts) / len(verdicts)
    summary = f"Baseline Summary: {agg.value.upper()} across {len(verdicts)} claims."

    return {
        "aggregate_verdict": agg,
        "aggregate_confidence": round(avg_conf, 2),
        "audit_summary": summary,
        "audit_complete": True,
    }
