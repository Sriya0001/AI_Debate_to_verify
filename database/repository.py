import json
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import config
from database.models import Base, AuditRecord, ClaimRecord, EvidenceRecord, AuditTrailRecord
from models.verdict import AuditSessionResult, FinalVerdict, VerdictType
from models.claims import AtomicClaim


class DatabaseRepository:
    """
    Persistence repository supporting local SQLite or PostgreSQL.
    """
    def __init__(self, db_url: Optional[str] = None):
        url = db_url or config.database_url
        self.engine = create_engine(url, connect_args={"check_same_thread": False} if "sqlite" in url else {})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def save_session(
        self,
        session_result: AuditSessionResult,
        evidence_items: List[Any] = None,
    ) -> str:
        """
        Saves a completed verification session, its claims, evidence, and audit logs.
        """
        session = self.SessionLocal()
        try:
            audit_rec = AuditRecord(
                session_id=session_result.session_id,
                input_text=session_result.input_text,
                mode=session_result.mode,
                aggregate_verdict=session_result.aggregate_verdict.value,
                aggregate_confidence=session_result.aggregate_confidence,
                summary=session_result.summary,
                execution_time_seconds=session_result.execution_time_seconds,
                total_searches=session_result.total_searches,
                total_llm_calls=session_result.total_llm_calls,
                estimated_tokens=session_result.estimated_tokens,
            )
            session.add(audit_rec)

            # Save claims
            for v in session_result.verdicts:
                claim_rec = ClaimRecord(
                    session_id=session_result.session_id,
                    claim_id=v.claim_id,
                    claim_text=v.claim_text,
                    verdict=v.verdict.value,
                    confidence=v.confidence,
                    reasoning=v.reasoning,
                    key_evidence=json.dumps(v.key_evidence),
                    contradicting_evidence=json.dumps(v.contradicting_evidence),
                    why_not_other_verdicts=v.why_not_other_verdicts,
                    debate_rounds=v.debate_rounds,
                    consensus_iterations=v.consensus_iterations,
                )
                session.add(claim_rec)

            # Save evidence
            if evidence_items:
                for e in evidence_items:
                    ev_rec = EvidenceRecord(
                        session_id=session_result.session_id,
                        evidence_id=e.evidence_id,
                        claim_id=e.claim_id,
                        url=e.url,
                        title=e.title,
                        snippet=e.snippet,
                        source_domain=e.source_domain,
                        supports_claim=e.supports_claim,
                        relevance_score=e.relevance_score,
                    )
                    session.add(ev_rec)

            # Save audit trail
            for log in session_result.audit_trail:
                tr_rec = AuditTrailRecord(
                    session_id=session_result.session_id,
                    log_entry=log,
                )
                session.add(tr_rec)

            session.commit()
            return session_result.session_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.SessionLocal()
        try:
            audit = session.query(AuditRecord).filter_by(session_id=session_id).first()
            if not audit:
                return None

            claims = session.query(ClaimRecord).filter_by(session_id=session_id).all()
            evidence = session.query(EvidenceRecord).filter_by(session_id=session_id).all()
            trail = session.query(AuditTrailRecord).filter_by(session_id=session_id).all()

            return {
                "session_id": audit.session_id,
                "input_text": audit.input_text,
                "mode": audit.mode,
                "aggregate_verdict": audit.aggregate_verdict,
                "aggregate_confidence": audit.aggregate_confidence,
                "summary": audit.summary,
                "execution_time_seconds": audit.execution_time_seconds,
                "total_searches": audit.total_searches,
                "total_llm_calls": audit.total_llm_calls,
                "created_at": audit.created_at.isoformat() if audit.created_at else None,
                "claims": [
                    {
                        "claim_id": c.claim_id,
                        "claim_text": c.claim_text,
                        "verdict": c.verdict,
                        "confidence": c.confidence,
                        "reasoning": c.reasoning,
                        "key_evidence": json.loads(c.key_evidence or "[]"),
                        "contradicting_evidence": json.loads(c.contradicting_evidence or "[]"),
                        "why_not_other_verdicts": c.why_not_other_verdicts,
                        "debate_rounds": c.debate_rounds,
                        "consensus_iterations": c.consensus_iterations,
                    }
                    for c in claims
                ],
                "evidence": [
                    {
                        "evidence_id": e.evidence_id,
                        "claim_id": e.claim_id,
                        "url": e.url,
                        "title": e.title,
                        "snippet": e.snippet,
                        "source_domain": e.source_domain,
                        "supports_claim": e.supports_claim,
                        "relevance_score": e.relevance_score,
                    }
                    for e in evidence
                ],
                "audit_trail": [t.log_entry for t in trail],
            }
        finally:
            session.close()

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        session = self.SessionLocal()
        try:
            records = session.query(AuditRecord).order_by(AuditRecord.created_at.desc()).limit(limit).all()
            return [
                {
                    "session_id": r.session_id,
                    "input_text": r.input_text[:80] + ("..." if len(r.input_text) > 80 else ""),
                    "aggregate_verdict": r.aggregate_verdict,
                    "aggregate_confidence": r.aggregate_confidence,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ]
        finally:
            session.close()


repository = DatabaseRepository()
