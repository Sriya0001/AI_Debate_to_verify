import json
import re
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import VeritasState
from models.evidence import ResearchTask
from utils.llm import get_llm
from utils.audit_trail import format_audit_log


SYSTEM_PROMPT = """You are the Autonomous Planner in VERITAS AI.
Your responsibility is to design a dynamic, multi-source research strategy for verifying a specific factual claim.

You must not rely on a single generic query. You must formulate distinct research tasks addressing:
1. Primary / Official sources (regulatory filings, government gazettes, corporate press releases, archival records).
2. Scientific or Academic consensus (if empirical, health, or technological).
3. Investigative reporting & independent journalism (checking for disputes, investigations, or retractions).
4. Potential counter-evidence and disconfirming searches (deliberately seeking contradictions).

Schema:
Respond with a JSON array of research tasks ONLY:
[
  {
    "task_id": "T1",
    "query": "<targeted search string>",
    "target_source_type": "official" | "academic" | "news" | "regulatory",
    "purpose": "<what this specific task verifies>",
    "priority": 1 | 2 | 3
  }
]"""


def planner_node(state: VeritasState) -> Dict[str, Any]:
    """
    LangGraph node: Formulates targeted research tasks for the current claim.
    """
    current_claim = state.get("current_claim")
    if not current_claim:
        # If current_claim not explicitly set, peek at first pending claim
        if state["pending_claim_ids"] and state["atomic_claims"]:
            cid = state["pending_claim_ids"][0]
            current_claim = next((c for c in state["atomic_claims"] if c.claim_id == cid), None)

    if not current_claim:
        return {
            "error": "Autonomous Planner could not identify current target claim.",
            "audit_trail": [format_audit_log("Autonomous Planner", "Failed: Target claim not found")],
        }

    claim_id = current_claim.claim_id
    claim_text = current_claim.claim_text
    questions_str = "\n".join(f"- {q}" for q in current_claim.verification_questions)
    entities_str = ", ".join(current_claim.entities) if current_claim.entities else "None specified"

    user_prompt = f"""Target Claim [{claim_id}]:
\"{claim_text}\"

Identified Entities: {entities_str}
Temporal Context: {current_claim.temporal_context or 'Unspecified'}

Key Verification Questions:
{questions_str}

Formulate 3 to 4 distinct, prioritized research tasks to rigorously verify this claim."""

    llm = get_llm("planner", temperature=0.1)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw_content = response.content.strip()
        tasks = _parse_tasks(raw_content, claim_id)
    except Exception as e:
        tasks = _fallback_plan(claim_id, claim_text)

    if not tasks:
        tasks = _fallback_plan(claim_id, claim_text)

    log_msg = format_audit_log(
        "Autonomous Planner",
        f"Generated {len(tasks)} targeted research tasks for claim {claim_id}"
    )

    return {
        "research_plan": tasks,
        "audit_trail": [log_msg],
        "error": None,
    }


def _parse_tasks(raw: str, claim_id: str) -> List[ResearchTask]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    if not isinstance(data, list):
        data = [data]

    tasks: List[ResearchTask] = []
    for i, item in enumerate(data, 1):
        tid = item.get("task_id") or f"T{i}"
        query = item.get("query", "").strip()
        if not query:
            continue
        tasks.append(
            ResearchTask(
                task_id=tid,
                claim_id=claim_id,
                query=query,
                target_source_type=item.get("target_source_type", "news"),
                purpose=item.get("purpose", f"Verify claim {claim_id}"),
                priority=int(item.get("priority", 1)),
            )
        )
    return tasks


def _fallback_plan(claim_id: str, claim_text: str) -> List[ResearchTask]:
    """Fallback planner generating dynamic multi-angle queries from claim words."""
    words = [w for w in re.findall(r"\w+", claim_text) if len(w) > 3][:6]
    core = " ".join(words)
    return [
        ResearchTask(
            task_id=f"T1",
            claim_id=claim_id,
            query=f"official documentation record {core}",
            target_source_type="official",
            purpose="Search official primary records",
            priority=1,
        ),
        ResearchTask(
            task_id=f"T2",
            claim_id=claim_id,
            query=f"investigative verification history {core}",
            target_source_type="news",
            purpose="Check established journalistic reporting",
            priority=2,
        ),
        ResearchTask(
            task_id=f"T3",
            claim_id=claim_id,
            query=f"counterargument criticism dispute {core}",
            target_source_type="news",
            purpose="Actively look for counter-evidence and contradictions",
            priority=2,
        ),
    ]
