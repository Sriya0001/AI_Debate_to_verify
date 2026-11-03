import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from models.evidence import SourceScore


# Authority Hierarchy Definitions
HIERARCHY_LEVELS = {
    1: "Official Government / Regulatory Repository",
    2: "Primary Institutional / Organization Official Source",
    3: "Peer-Reviewed Academic & Scientific Publication",
    4: "Major Established Independent News Agency",
    5: "Established Reference / Fact-Checking Archive",
    6: "Secondary Blog / Social Forum / Unverified Domain",
}


# Known domain mappings
GOVERNMENT_DOMAINS = {
    "gov", "mil", "sec.gov", "fda.gov", "who.int", "un.org", "europa.eu",
    "cdc.gov", "nasa.gov", "nih.gov", "justice.gov", "state.gov", "gov.uk"
}

ACADEMIC_DOMAINS = {
    "edu", "nature.com", "science.org", "thelancet.com", "cell.com", "nejm.org",
    "arxiv.org", "pnas.org", "jstor.org", "springer.com", "sciencedirect.com"
}

PRIMARY_INSTITUTIONAL_DOMAINS = {
    "toureiffel.paris", "cern.ch", "esa.int", "iso.org", "w3.org", "ietf.org"
}

MAJOR_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "wsj.com", "washingtonpost.com", "bloomberg.com", "ft.com", "theguardian.com",
    "afp.com", "economist.com", "pbs.org", "npr.org"
}

REFERENCE_DOMAINS = {
    "britannica.com", "wikipedia.org", "snopes.com", "politifact.com",
    "factcheck.org", "investopedia.com", "history.com"
}

LOW_AUTHORITY_DOMAINS = {
    "medium.com", "reddit.com", "twitter.com", "x.com", "facebook.com",
    "quora.com", "substack.com", "wordpress.com", "blogspot.com"
}


def analyze_source_credibility(url: str, domain: Optional[str] = None) -> SourceScore:
    """
    Evaluates source credibility based on domain authority, source type,
    independence, and primary vs. secondary status with explicit rationale.
    """
    if not domain:
        domain = urlparse(url).netloc.lower().replace("www.", "")

    domain_lower = domain.lower()

    # Level 1: Government / Regulatory
    if any(domain_lower.endswith(g) or domain_lower == g for g in GOVERNMENT_DOMAINS):
        return SourceScore(
            source_domain=domain,
            url=url,
            authority_level=1,
            authority_name=HIERARCHY_LEVELS[1],
            credibility_score=0.96,
            is_independent=True,
            is_primary=True,
            recency_factor=1.0,
            rationale="Official statutory or regulatory filing with legal accountability and primary evidentiary authority.",
        )

    # Level 2: Primary Institutional Official
    if any(domain_lower.endswith(p) or domain_lower == p for p in PRIMARY_INSTITUTIONAL_DOMAINS):
        return SourceScore(
            source_domain=domain,
            url=url,
            authority_level=2,
            authority_name=HIERARCHY_LEVELS[2],
            credibility_score=0.92,
            is_independent=True,
            is_primary=True,
            recency_factor=0.98,
            rationale="Official first-party institutional publication directly responsible for the entity or event described.",
        )

    # Level 3: Peer-Reviewed Scientific & Academic
    if any(domain_lower.endswith(a) or domain_lower == a for a in ACADEMIC_DOMAINS):
        return SourceScore(
            source_domain=domain,
            url=url,
            authority_level=3,
            authority_name=HIERARCHY_LEVELS[3],
            credibility_score=0.94,
            is_independent=True,
            is_primary=False,
            recency_factor=0.95,
            rationale="Peer-reviewed scholarly literature subject to blinded editorial scrutiny and methodology verification.",
        )

    # Level 4: Major Established News
    if any(domain_lower.endswith(m) or domain_lower == m for m in MAJOR_NEWS_DOMAINS):
        return SourceScore(
            source_domain=domain,
            url=url,
            authority_level=4,
            authority_name=HIERARCHY_LEVELS[4],
            credibility_score=0.84,
            is_independent=True,
            is_primary=False,
            recency_factor=0.95,
            rationale="Major independent wire service or journalistic outlet adhering to formal editorial standards and corrections policy.",
        )

    # Level 5: Established Reference / Secondary
    if any(domain_lower.endswith(r) or domain_lower == r for r in REFERENCE_DOMAINS):
        return SourceScore(
            source_domain=domain,
            url=url,
            authority_level=5,
            authority_name=HIERARCHY_LEVELS[5],
            credibility_score=0.72,
            is_independent=True,
            is_primary=False,
            recency_factor=0.90,
            rationale="Curated secondary reference encyclopedia or fact-checking repository with cross-citations.",
        )

    # Level 6: Low Authority / Forum / Blog
    if any(domain_lower.endswith(l) or domain_lower == l for l in LOW_AUTHORITY_DOMAINS):
        return SourceScore(
            source_domain=domain,
            url=url,
            authority_level=6,
            authority_name=HIERARCHY_LEVELS[6],
            credibility_score=0.30,
            is_independent=False,
            is_primary=False,
            recency_factor=0.80,
            rationale="User-generated or unvetted blogging platform without editorial oversight; requires primary corroboration.",
        )

    # Unrecognized / Generic domain default
    return SourceScore(
        source_domain=domain,
        url=url,
        authority_level=5,
        authority_name="General Web Source",
        credibility_score=0.60,
        is_independent=True,
        is_primary=False,
        recency_factor=0.85,
        rationale="Uncategorized web publication; moderate evidentiary weight pending corroboration by higher-tier sources.",
    )
