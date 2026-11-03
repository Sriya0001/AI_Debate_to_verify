from typing import List, Dict, Optional
from datetime import datetime
from models.evidence import EvidenceItem


class EvidencePool:
    """
    Centralized repository for all retrieved evidence items.
    Assigns globally unique identifiers (E1, E2, E3...) and deduplicates sources.
    """
    def __init__(self):
        self._items: Dict[str, EvidenceItem] = {}
        self._url_to_id: Dict[str, str] = {}
        self._counter: int = 1

    def add_item(
        self,
        claim_id: str,
        title: str,
        url: str,
        snippet: str,
        source_domain: str,
        task_id: Optional[str] = None,
        supports_claim: Optional[bool] = None,
        relevance_score: float = 0.85,
    ) -> EvidenceItem:
        """
        Adds an item to the pool or returns existing if URL already recorded.
        """
        normalized_url = url.strip().rstrip("/")
        if normalized_url in self._url_to_id:
            existing_id = self._url_to_id[normalized_url]
            return self._items[existing_id]

        evidence_id = f"E{self._counter}"
        self._counter += 1

        item = EvidenceItem(
            evidence_id=evidence_id,
            claim_id=claim_id,
            task_id=task_id,
            title=title,
            url=url,
            snippet=snippet,
            source_domain=source_domain,
            retrieved_at=datetime.utcnow().isoformat(),
            supports_claim=supports_claim,
            relevance_score=relevance_score,
        )

        self._items[evidence_id] = item
        self._url_to_id[normalized_url] = evidence_id
        return item

    def get_by_id(self, evidence_id: str) -> Optional[EvidenceItem]:
        return self._items.get(evidence_id)

    def get_for_claim(self, claim_id: str) -> List[EvidenceItem]:
        return [item for item in self._items.values() if item.claim_id == claim_id]

    def all_items(self) -> List[EvidenceItem]:
        return list(self._items.values())

    def format_pool_for_prompt(self, claim_id: Optional[str] = None) -> str:
        """
        Formats evidence items with explicit IDs for agent prompt grounding.
        Example:
        [E1] Title (domain.com)
             URL: https://...
             Snippet: ...
        """
        items = self.get_for_claim(claim_id) if claim_id else self.all_items()
        if not items:
            return "No evidence retrieved yet."

        lines = []
        for e in items:
            stance_label = "SUPPORTS" if e.supports_claim is True else ("CONTRADICTS" if e.supports_claim is False else "NEUTRAL/UNTAGGED")
            lines.append(f"[{e.evidence_id}] {e.title}")
            lines.append(f"     Source Domain: {e.source_domain} | Stance: {stance_label} | Relevance: {e.relevance_score:.2f}")
            lines.append(f"     URL: {e.url}")
            lines.append(f"     Snippet: {e.snippet}")
            lines.append("")

        return "\n".join(lines)


# Global pool singleton
evidence_pool = EvidencePool()
