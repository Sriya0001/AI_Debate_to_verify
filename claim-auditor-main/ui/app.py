import streamlit as st
import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000"

# Page config

st.set_page_config(
    page_title="ClaimAuditor",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS

USE_CUSTOM_CSS = True

if USE_CUSTOM_CSS:
    css = (Path(__file__).parent / "style.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# Header

st.markdown("""
<div class="ca-header">
    <div class="ca-wordmark">⚖ ClaimAuditor</div>
    <div class="ca-tagline">Multi-agent adversarial fact verification</div>
</div>
""", unsafe_allow_html=True)


# Helpers

RULING_COLORS = {
    "supported": "#4ade80",
    "refuted":   "#f87171",
    "uncertain": "#fbbf24",
    "escalated": "#a78bfa",
}

def confidence_bar(confidence: float, ruling: str) -> str:
    pct  = int(confidence * 100)
    color = RULING_COLORS.get(ruling, "#666")
    return (
        f'<div class="conf-row">'
        f'<div class="conf-bar-bg">'
        f'<div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
        f'<span class="conf-label">{pct}%</span>'
        f'</div>'
    )

def verdict_badge(ruling: str) -> str:
    return f'<span class="verdict-badge badge-{ruling}">{ruling}</span>'

def call_api(text: str, source_label: str, max_rounds: int) -> dict:
    resp = requests.post(
        f"{API_URL}/audit",
        json={
            "text":         text,
            "source_label": source_label or None,
            "max_rounds":   max_rounds,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


# Sidebar — settings

with st.sidebar:
    st.markdown("### Settings")
    max_rounds = st.slider(
        "Max debate rounds per claim",
        min_value=1, max_value=5, value=2,
        help="More rounds = deeper debate, more tokens and latency."
    )
    source_label = st.text_input(
        "Source label (optional)",
        placeholder="e.g. GPT-4o response, RAG answer",
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#444;line-height:1.8;">
    AGENTS<br>
    ─ Decomposer<br>
    ─ Advocate<br>
    ─ Skeptic<br>
    ─ Arbitrator<br>
    ─ Reporter
    </div>
    """, unsafe_allow_html=True)


# Main input

col1, col2 = st.columns([3, 1])
with col1:
    input_text = st.text_area(
        "Text to audit",
        height=160,
        placeholder="Paste any LLM-generated text, article excerpt, or factual claim here...",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run_audit = st.button("Run Audit", use_container_width=True)

    if st.session_state.get("last_result"):
        if st.button("Clear", use_container_width=True):
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_transcript", None)
            st.rerun()


# Run

if run_audit:
    if not input_text.strip():
        st.warning("Paste some text to audit first.")
    else:
        with st.spinner("Agents are deliberating..."):
            try:
                result = call_api(input_text.strip(), source_label, max_rounds)
                st.session_state["last_result"]     = result
                st.session_state["last_input_text"] = input_text.strip()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is `uvicorn api.main:app` running on port 8000?")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# Results

if result := st.session_state.get("last_result"):

    st.markdown("---")

    # Escalation banner
    if result.get("escalate_to_human"):
        st.markdown("""
        <div class="escalation-banner">
            ⚠ One or more claims require human review — confidence below threshold
            or genuine contestation detected between agents.
        </div>
        """, unsafe_allow_html=True)

    # Aggregate block
    agg_verdict    = result["aggregate_verdict"]
    agg_confidence = result["aggregate_confidence"]
    total_claims   = result["total_claims"]

    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"""
        <div class="agg-block">
            <div class="agg-label">Overall verdict — {total_claims} claim{'s' if total_claims != 1 else ''} audited</div>
            {verdict_badge(agg_verdict)}
            {confidence_bar(agg_confidence, agg_verdict)}
            <div class="agg-summary">{result.get("audit_summary", "")}</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        # Claim ruling breakdown counts
        counts = {"supported": 0, "refuted": 0, "uncertain": 0, "escalated": 0}
        for v in result["claim_verdicts"]:
            r = v["ruling"]
            counts[r] = counts.get(r, 0) + 1

        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        for ruling, count in counts.items():
            if count:
                color = RULING_COLORS[ruling]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                                 color:{color};letter-spacing:0.1em;text-transform:uppercase;
                                 min-width:80px;">{ruling}</span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                                 color:{color};">{count}</span>
                </div>
                """, unsafe_allow_html=True)

    # Per-claim cards
    st.markdown("#### Claim breakdown")

    for verdict in result["claim_verdicts"]:
        ruling     = verdict["ruling"]
        confidence = verdict["confidence"]
        claim_text = verdict["claim_text"]
        reasoning  = verdict["reasoning"]
        rounds     = verdict["debate_rounds"]

        st.markdown(f"""
        <div class="claim-card {ruling}">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;
                        gap:1rem;margin-bottom:0.6rem;">
                <div class="claim-text">{claim_text}</div>
                <div style="flex-shrink:0">{verdict_badge(ruling)}</div>
            </div>
            {confidence_bar(confidence, ruling)}
            <div class="claim-reasoning">{reasoning}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                        color:#333;margin-top:0.6rem;">
                {rounds} debate round{'s' if rounds != 1 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Debate transcript
    # Fetching transcript requires a second call or storing it in session.
    # For now we surface a note — the full transcript lives in the graph state
    # and can be exposed by adding a /audit/transcript endpoint later.
    with st.expander("About this audit"):
        st.markdown(f"""
        **Model**: Decomposer / Advocate / Skeptic / Reporter → `claude-haiku-4-5-20251001`
        **Model**: Arbitrator → `claude-sonnet-4-6`
        **Max debate rounds**: {max_rounds}  
        **Source**: {result.get('source_label') or 'not specified'}  
        **Total claims audited**: {total_claims}
        """)
        st.markdown("""
        *To expose the full debate transcript, add a `/audit/transcript` endpoint  
        that returns `state["debate_transcript"]` grouped by claim.*
        """)