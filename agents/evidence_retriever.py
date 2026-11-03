from typing import List, Dict, Any
from urllib.parse import urlparse

from graph.state import VeritasState
from models.evidence import EvidenceItem, ResearchTask
from tools.search import search
from tools.evidence import evidence_pool
from utils.audit_trail import format_audit_log


def evidence_retriever_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Executes research tasks formulated by the Planner,
    deduplicates sources, extracts snippets, assigns unique IDs (E1, E2...),
    and indexes them into the centralized evidence pool.
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        return {
            "error": "Evidence Retriever: No current claim found in state.",
            "audit_trail": [format_audit_log("Evidence Retriever", "Failed: Target claim missing")],
        }

    claim_id = current_claim.claim_id
    tasks: List[ResearchTask] = state.get("research_plan", [])

    if not tasks:
        # Default fallback task if plan was empty
        tasks = [
            ResearchTask(
                task_id="T0",
                claim_id=claim_id,
                query=current_claim.claim_text,
                target_source_type="news",
                purpose="Direct claim search",
                priority=1,
            )
        ]

    new_evidence_items: List[EvidenceItem] = []

    for task in tasks:
        try:
            raw_results = search(task.query, max_results=3)
        except Exception as e:
            raw_results = []

        for r in raw_results:
            url = r.get("url", "")
            title = r.get("title", "")
            snippet = r.get("content", "").strip()
            domain = r.get("domain", "") or urlparse(url).netloc.lower().replace("www.", "")
            score = float(r.get("score", 0.85))

            # Initial heuristic stance tagging based on content
            supports = _infer_initial_stance(current_claim.claim_text, snippet)

            item = evidence_pool.add_item(
                claim_id=claim_id,
                task_id=task.task_id,
                title=title or domain,
                url=url,
                snippet=snippet,
                source_domain=domain or "unknown-domain.com",
                supports_claim=supports,
                relevance_score=score,
            )
            # Avoid duplicate references in the node update
            if item not in new_evidence_items:
                new_evidence_items.append(item)

    # Build audit log
    if new_evidence_items:
        id_range = f"{new_evidence_items[0].evidence_id} - {new_evidence_items[-1].evidence_id}"
        log_msg = format_audit_log(
            "Evidence Retriever",
            f"Retrieved {len(new_evidence_items)} evidence item(s) [{id_range}] for claim {claim_id}"
        )
    else:
        log_msg = format_audit_log(
            "Evidence Retriever",
            f"No external evidence retrieved for claim {claim_id}"
        )

    return {
        "evidence_pool": new_evidence_items,
        "audit_trail": [log_msg],
        "error": None,
    }


def _infer_initial_stance(claim_text: str, snippet: str) -> bool | None:
    """Simple heuristic to tag initial stance without deep LLM call."""
    snip_lower = snippet.lower()
    claim_lower = claim_text.lower()

    refutation_cues = ["false", "debunked", "incorrect", "denied", "no evidence", "refuted", "myth", "disproven"]
    support_cues = ["confirmed", "official", "verified", "built in", "completed in", "stands at", "acquired"]

    for r in refutation_cues:
        if r in snip_lower:
            return False
    for s in support_cues:
        if s in snip_lower:
            return True
    return None
