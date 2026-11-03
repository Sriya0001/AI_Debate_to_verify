from sqlalchemy import Column, String, Float, Integer, Boolean, Text, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()


class AuditRecord(Base):
    __tablename__ = "audit_records"

    session_id = Column(String(64), primary_key=True)
    input_text = Column(Text, nullable=False)
    mode = Column(String(32), default="veritas")
    aggregate_verdict = Column(String(32), nullable=False)
    aggregate_confidence = Column(Float, nullable=False)
    summary = Column(Text, nullable=True)
    execution_time_seconds = Column(Float, default=0.0)
    total_searches = Column(Integer, default=0)
    total_llm_calls = Column(Integer, default=0)
    estimated_tokens = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ClaimRecord(Base):
    __tablename__ = "claim_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    claim_id = Column(String(32), nullable=False)
    claim_text = Column(Text, nullable=False)
    verdict = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    key_evidence = Column(Text, nullable=True) # JSON list
    contradicting_evidence = Column(Text, nullable=True) # JSON list
    why_not_other_verdicts = Column(Text, nullable=True)
    debate_rounds = Column(Integer, default=1)
    consensus_iterations = Column(Integer, default=0)


class EvidenceRecord(Base):
    __tablename__ = "evidence_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    evidence_id = Column(String(32), nullable=False)
    claim_id = Column(String(32), nullable=False)
    url = Column(Text, nullable=False)
    title = Column(Text, nullable=True)
    snippet = Column(Text, nullable=True)
    source_domain = Column(String(128), nullable=True)
    supports_claim = Column(Boolean, nullable=True)
    relevance_score = Column(Float, default=0.85)


class AuditTrailRecord(Base):
    __tablename__ = "audit_trail_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    log_entry = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
