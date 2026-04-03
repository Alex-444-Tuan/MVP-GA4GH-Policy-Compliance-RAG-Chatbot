"""Seed the Neo4j knowledge graph with policy nodes, requirements, and relationships.

Run once after starting Neo4j:
    python scripts/seed_knowledge_graph.py

Requires:
    - Neo4j running (see docker-compose.yml)
    - ANTHROPIC_API_KEY and OPENAI_API_KEY set in .env
    - data/policies/ga4gh_framework.md
    - data/policies/ga4gh_daa_clauses.md
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import configure_logging, get_settings

configure_logging()
logger = logging.getLogger(__name__)

settings = get_settings()
DATA_DIR = Path(__file__).parent.parent / "data"

# ── Requirement definitions (from spec) ───────────────────────────────────────

REQUIREMENTS: list[dict] = [
    {
        "id": "REQ-01",
        "description": "Data use must be limited to the stated research purpose",
        "severity": "CRITICAL",
        "keywords": ["research purpose", "approved purpose", "data use", "permitted use", "prohibited use", "purpose limitation"],
        "daa_clause_category": "Purposes of Use",
        "daa_clause_number": "2",
        "group": "RG-GOVERNANCE",
    },
    {
        "id": "REQ-02",
        "description": "Principal investigator and all team members must be identified",
        "severity": "CRITICAL",
        "keywords": ["principal investigator", "PI", "research team", "authorized personnel", "team members"],
        "daa_clause_category": "Parties and Definitions",
        "daa_clause_number": "1",
        "group": "RG-GOVERNANCE",
    },
    {
        "id": "REQ-03",
        "description": "Institutional affiliation and authority must be established",
        "severity": "MAJOR",
        "keywords": ["institutional affiliation", "institution", "institutional representative", "authority", "legal entity"],
        "daa_clause_category": "Parties and Definitions",
        "daa_clause_number": "1",
        "group": "RG-GOVERNANCE",
    },
    {
        "id": "REQ-04",
        "description": "IRB or ethics committee approval must be documented",
        "severity": "CRITICAL",
        "keywords": ["IRB", "ethics committee", "institutional review board", "ethics approval", "ethics protocol", "research ethics"],
        "daa_clause_category": "Ethics Oversight",
        "daa_clause_number": "2",
        "group": "RG-ETHICS",
    },
    {
        "id": "REQ-05",
        "description": "Data security measures must be specified (encryption, access control, physical security)",
        "severity": "CRITICAL",
        "keywords": ["encryption", "access control", "physical security", "MFA", "data security", "secure storage", "RBAC", "AES"],
        "daa_clause_category": "Data Security Standards",
        "daa_clause_number": "11",
        "group": "RG-SECURITY",
    },
    {
        "id": "REQ-06",
        "description": "Data breach notification procedures must be defined (detection, timeline, contacts)",
        "severity": "CRITICAL",
        "keywords": ["breach notification", "data breach", "unauthorized disclosure", "breach detection", "incident response", "breach contact"],
        "daa_clause_category": "Data Breach Notification",
        "daa_clause_number": "12",
        "group": "RG-SECURITY",
    },
    {
        "id": "REQ-07",
        "description": "Data destruction or return plan must be stated for project end/termination",
        "severity": "CRITICAL",
        "keywords": ["data destruction", "data deletion", "data return", "termination", "end of project", "secure deletion", "data disposal"],
        "daa_clause_category": "Term and Termination",
        "daa_clause_number": "10",
        "group": "RG-GOVERNANCE",
    },
    {
        "id": "REQ-08",
        "description": "Intellectual property rights for derived works must be addressed",
        "severity": "CRITICAL",
        "keywords": ["intellectual property", "IP rights", "derived works", "licensing", "patents", "copyright", "IP claims", "downstream discoveries"],
        "daa_clause_category": "Intellectual Property",
        "daa_clause_number": "4",
        "group": "RG-LEGAL",
    },
    {
        "id": "REQ-09",
        "description": "Consent basis for the data must be confirmed compatible with proposed use",
        "severity": "MAJOR",
        "keywords": ["informed consent", "consent basis", "consent compatibility", "consent documentation", "re-consent", "consent restrictions"],
        "daa_clause_category": "Consent Compliance",
        "daa_clause_number": "2",
        "group": "RG-ETHICS",
    },
    {
        "id": "REQ-10",
        "description": "Re-identification attempts must be explicitly prohibited",
        "severity": "MAJOR",
        "keywords": ["re-identification", "reidentification", "anonymization", "de-identification", "participant privacy", "linkage attack", "stigmatization"],
        "daa_clause_category": "Permitted and Prohibited Uses",
        "daa_clause_number": "8",
        "group": "RG-PRIVACY",
    },
    {
        "id": "REQ-11",
        "description": "Third-party sharing and sub-licensing restrictions must be stated",
        "severity": "MAJOR",
        "keywords": ["third party", "data transfer", "outbound transfer", "sub-licensing", "external collaborator", "transfer restrictions", "data disclosure"],
        "daa_clause_category": "Transfer Restrictions",
        "daa_clause_number": "5",
        "group": "RG-GOVERNANCE",
    },
    {
        "id": "REQ-12",
        "description": "Compliance monitoring and audit provisions must be acknowledged",
        "severity": "MAJOR",
        "keywords": ["compliance monitoring", "audit", "reporting requirements", "progress report", "annual report", "oversight", "compliance obligations"],
        "daa_clause_category": "Compliance and Oversight",
        "daa_clause_number": "3",
        "group": "RG-GOVERNANCE",
    },
    {
        "id": "REQ-13",
        "description": "Sanctions for non-compliance must be acknowledged",
        "severity": "MINOR",
        "keywords": ["sanctions", "non-compliance", "termination", "liability", "penalties", "consequences", "legal action", "access revocation"],
        "daa_clause_category": "Liability and Indemnification",
        "daa_clause_number": "14",
        "group": "RG-LEGAL",
    },
    {
        "id": "REQ-14",
        "description": "Data source attribution and recognition practices must be stated",
        "severity": "MINOR",
        "keywords": ["attribution", "recognition", "citation", "acknowledgement", "dataset citation", "DOI", "attribution statement"],
        "daa_clause_category": "Recognition and Attribution",
        "daa_clause_number": "9",
        "group": "RG-GOVERNANCE",
    },
    {
        "id": "REQ-15",
        "description": "Governing law and jurisdiction must be specified",
        "severity": "MINOR",
        "keywords": ["governing law", "jurisdiction", "choice of law", "dispute resolution", "applicable law", "legal framework", "courts"],
        "daa_clause_category": "Governing Law",
        "daa_clause_number": "15",
        "group": "RG-LEGAL",
    },
]

REQUIREMENT_GROUPS: list[dict] = [
    {"id": "RG-GOVERNANCE", "name": "Data Governance", "description": "Requirements related to data use purpose, team identification, and reporting"},
    {"id": "RG-SECURITY", "name": "Data Security", "description": "Requirements related to technical and physical security measures and breach response"},
    {"id": "RG-ETHICS", "name": "Ethics and Consent", "description": "Requirements related to ethics oversight, IRB approval, and consent compatibility"},
    {"id": "RG-PRIVACY", "name": "Privacy Protection", "description": "Requirements related to re-identification prohibition and participant privacy"},
    {"id": "RG-LEGAL", "name": "Legal and IP", "description": "Requirements related to intellectual property, liability, and governing law"},
]

# ── Requirement ↔ Requirement relationships (cross-cutting dependencies) ──────

RELATED_REQUIREMENTS: list[tuple[str, str]] = [
    ("REQ-01", "REQ-09"),   # Purpose of use ↔ consent compatibility
    ("REQ-02", "REQ-03"),   # PI identification ↔ institutional affiliation
    ("REQ-05", "REQ-06"),   # Security measures ↔ breach notification
    ("REQ-07", "REQ-05"),   # Data destruction ↔ security
    ("REQ-10", "REQ-01"),   # Re-identification ↔ purpose limitation
    ("REQ-11", "REQ-05"),   # Transfer restrictions ↔ security
    ("REQ-12", "REQ-13"),   # Audit provisions ↔ sanctions
    ("REQ-08", "REQ-14"),   # IP rights ↔ attribution
]


# ── Chunking helpers ──────────────────────────────────────────────────────────


def chunk_framework_text(text: str) -> list[dict]:
    """Split GA4GH Framework markdown into semantically coherent chunks by section.

    Returns list of dicts with keys: id, section_title, text, chunk_index.
    """
    # Split on markdown headers (## or ###)
    section_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    splits = list(section_pattern.finditer(text))

    chunks = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        section_title = match.group(2).strip()
        body = text[start:end].strip()

        # Skip very short sections (acknowledgements, appendix refs, etc.)
        if len(body) < 100:
            continue

        # If a section is very long (> 800 tokens ≈ 3200 chars), split further
        if len(body) > 3200:
            sub_chunks = _split_long_section(body, section_title, chunk_index_start=len(chunks))
            chunks.extend(sub_chunks)
        else:
            chunks.append({
                "id": f"fw_chunk_{len(chunks):03d}",
                "policy_id": "ga4gh_framework_v1",
                "section_title": section_title,
                "chunk_index": len(chunks),
                "text": body,
            })

    return chunks


def _split_long_section(text: str, section_title: str, chunk_index_start: int) -> list[dict]:
    """Split a long section by paragraph boundaries into ~350-token chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_parts: list[str] = []
    current_len = 0
    target_chars = 1400  # ~350 tokens

    for para in paragraphs:
        if current_len + len(para) > target_chars and current_parts:
            idx = chunk_index_start + len(chunks)
            chunks.append({
                "id": f"fw_chunk_{idx:03d}",
                "policy_id": "ga4gh_framework_v1",
                "section_title": section_title,
                "chunk_index": idx,
                "text": "\n\n".join(current_parts),
            })
            current_parts = [para]
            current_len = len(para)
        else:
            current_parts.append(para)
            current_len += len(para)

    if current_parts:
        idx = chunk_index_start + len(chunks)
        chunks.append({
            "id": f"fw_chunk_{idx:03d}",
            "policy_id": "ga4gh_framework_v1",
            "section_title": section_title,
            "chunk_index": idx,
            "text": "\n\n".join(current_parts),
        })
    return chunks


def chunk_daa_clauses(text: str) -> list[dict]:
    """Split DAA clauses markdown into one chunk per numbered clause.

    Returns list of dicts with keys: id, clause_category, clause_number, text.
    """
    clause_pattern = re.compile(
        r"^##\s+Clause\s+(\d+):\s+(.+)$", re.MULTILINE
    )
    splits = list(clause_pattern.finditer(text))

    chunks = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        clause_number = match.group(1)
        clause_category = match.group(2).strip()
        body = text[start:end].strip()

        chunks.append({
            "id": f"daa_clause_{int(clause_number):02d}",
            "policy_form_id": "ga4gh_daa_clauses_v1",
            "clause_number": clause_number,
            "clause_category": clause_category,
            "text": body,
        })

    return chunks


# ── OpenAI embedding generation ───────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_embeddings(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Generate embeddings for a batch of texts using text-embedding-3-large.

    Returns list of 3072-dimensional embedding vectors.
    """
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        dimensions=settings.embedding_dims,
    )
    return [item.embedding for item in response.data]


def embed_chunks_in_batches(
    chunks: list[dict],
    client: OpenAI,
    batch_size: int = 20,
    text_key: str = "text",
) -> list[dict]:
    """Add 'embedding' field to each chunk dict. Processes in batches."""
    total = len(chunks)
    logger.info("Generating embeddings for %d chunks (batch_size=%d)...", total, batch_size)

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c[text_key] for c in batch]
        embeddings = generate_embeddings(texts, client)
        for chunk, emb in zip(batch, embeddings):
            chunk["embedding"] = emb
        logger.info("  Embedded %d/%d chunks", min(i + batch_size, total), total)
        time.sleep(0.5)  # rate-limit courtesy pause

    return chunks


# ── Neo4j seeding ─────────────────────────────────────────────────────────────


def seed_graph(driver: GraphDatabase.driver, framework_chunks: list[dict], daa_chunks: list[dict]) -> None:
    """Create all nodes and relationships in Neo4j."""
    with driver.session() as session:
        _create_constraints(session)
        _create_policy_nodes(session)
        _create_policy_chunks(session, framework_chunks)
        _create_daa_chunks(session, daa_chunks)
        _create_requirement_groups(session)
        _create_requirements(session)
        _create_chunk_relationships(session, framework_chunks)
        _create_requirement_chunk_relationships(session, daa_chunks)
        _create_related_requirements(session)
        logger.info("Knowledge graph seeding complete.")


def _create_constraints(session) -> None:
    """Create uniqueness constraints for all node types."""
    constraints = [
        "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (p:Policy) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT policy_form_id IF NOT EXISTS FOR (pf:PolicyForm) REQUIRE pf.id IS UNIQUE",
        "CREATE CONSTRAINT policy_chunk_id IF NOT EXISTS FOR (pc:PolicyChunk) REQUIRE pc.id IS UNIQUE",
        "CREATE CONSTRAINT policy_form_chunk_id IF NOT EXISTS FOR (pfc:PolicyFormChunk) REQUIRE pfc.id IS UNIQUE",
        "CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT requirement_group_id IF NOT EXISTS FOR (rg:RequirementGroup) REQUIRE rg.id IS UNIQUE",
    ]
    for cypher in constraints:
        session.run(cypher)
    logger.info("Constraints created.")


def _create_policy_nodes(session) -> None:
    """Create Policy and PolicyForm root nodes."""
    session.run(
        """
        MERGE (p:Policy {id: $id})
        SET p.name = $name, p.version = $version, p.source_url = $source_url
        """,
        id="ga4gh_framework_v1",
        name="GA4GH Framework for Responsible Sharing of Genomic and Health-Related Data",
        version="1.0",
        source_url="https://www.ga4gh.org/genomic-data-toolkit/regulatory-ethics-toolkit/framework-for-responsible-sharing-of-genomic-and-health-related-data/",
    )
    session.run(
        """
        MERGE (pf:PolicyForm {id: $id})
        SET pf.name = $name, pf.version = $version, pf.source_url = $source_url
        """,
        id="ga4gh_daa_clauses_v1",
        name="GA4GH Model Data Access Agreement (DAA) Clauses",
        version="1.0",
        source_url="https://www.ga4gh.org/genomic-data-toolkit/regulatory-ethics-toolkit/model-data-access-agreement/",
    )
    logger.info("Policy and PolicyForm nodes created.")


def _create_policy_chunks(session, chunks: list[dict]) -> None:
    """Create PolicyChunk nodes and BELONGS_TO relationships."""
    for chunk in chunks:
        session.run(
            """
            MERGE (pc:PolicyChunk {id: $id})
            SET pc.text = $text,
                pc.section_title = $section_title,
                pc.chunk_index = $chunk_index,
                pc.embedding = $embedding
            WITH pc
            MATCH (p:Policy {id: $policy_id})
            MERGE (pc)-[:BELONGS_TO]->(p)
            """,
            id=chunk["id"],
            text=chunk["text"],
            section_title=chunk["section_title"],
            chunk_index=chunk["chunk_index"],
            embedding=chunk.get("embedding", []),
            policy_id=chunk["policy_id"],
        )
    logger.info("Created %d PolicyChunk nodes.", len(chunks))


def _create_daa_chunks(session, chunks: list[dict]) -> None:
    """Create PolicyFormChunk nodes and BELONGS_TO relationships."""
    for chunk in chunks:
        session.run(
            """
            MERGE (pfc:PolicyFormChunk {id: $id})
            SET pfc.text = $text,
                pfc.clause_category = $clause_category,
                pfc.clause_number = $clause_number,
                pfc.embedding = $embedding
            WITH pfc
            MATCH (pf:PolicyForm {id: $policy_form_id})
            MERGE (pfc)-[:BELONGS_TO]->(pf)
            """,
            id=chunk["id"],
            text=chunk["text"],
            clause_category=chunk["clause_category"],
            clause_number=chunk["clause_number"],
            embedding=chunk.get("embedding", []),
            policy_form_id=chunk["policy_form_id"],
        )
    logger.info("Created %d PolicyFormChunk nodes.", len(chunks))


def _create_requirement_groups(session) -> None:
    """Create RequirementGroup nodes."""
    for group in REQUIREMENT_GROUPS:
        session.run(
            """
            MERGE (rg:RequirementGroup {id: $id})
            SET rg.name = $name, rg.description = $description
            """,
            id=group["id"],
            name=group["name"],
            description=group["description"],
        )
    logger.info("Created %d RequirementGroup nodes.", len(REQUIREMENT_GROUPS))


def _create_requirements(session) -> None:
    """Create Requirement nodes and MEMBER_OF relationships."""
    for req in REQUIREMENTS:
        session.run(
            """
            MERGE (r:Requirement {id: $id})
            SET r.description = $description,
                r.severity = $severity,
                r.keywords = $keywords,
                r.daa_clause_category = $daa_clause_category,
                r.daa_clause_number = $daa_clause_number
            WITH r
            MATCH (rg:RequirementGroup {id: $group_id})
            MERGE (r)-[:MEMBER_OF]->(rg)
            """,
            id=req["id"],
            description=req["description"],
            severity=req["severity"],
            keywords=req["keywords"],
            daa_clause_category=req["daa_clause_category"],
            daa_clause_number=req["daa_clause_number"],
            group_id=req["group"],
        )
    logger.info("Created %d Requirement nodes.", len(REQUIREMENTS))


def _create_chunk_relationships(session, chunks: list[dict]) -> None:
    """Create sequential NEXT relationships between PolicyChunks."""
    sorted_chunks = sorted(chunks, key=lambda c: c["chunk_index"])
    for i in range(len(sorted_chunks) - 1):
        session.run(
            """
            MATCH (a:PolicyChunk {id: $id_a}), (b:PolicyChunk {id: $id_b})
            MERGE (a)-[:NEXT]->(b)
            """,
            id_a=sorted_chunks[i]["id"],
            id_b=sorted_chunks[i + 1]["id"],
        )
    logger.info("Created %d NEXT relationships.", len(sorted_chunks) - 1)


def _create_requirement_chunk_relationships(session, daa_chunks: list[dict]) -> None:
    """Create CHECKS and REMEDIATED_BY relationships.

    - (Requirement)-[:CHECKS {relevance_weight}]->(PolicyChunk)
    - (Requirement)-[:REMEDIATED_BY {specificity}]->(PolicyFormChunk)

    Mapping: each Requirement maps to one or more framework PolicyChunks
    (by section relevance) and exactly one PolicyFormChunk (the DAA clause).
    """
    # Map from requirement ID to relevant framework section titles
    req_to_framework_sections: dict[str, list[str]] = {
        "REQ-01": ["Purposes of Use", "Core Elements of Responsible Data Sharing", "Transparency"],
        "REQ-02": ["Parties and Definitions", "Research Team", "Application"],
        "REQ-03": ["Parties and Definitions", "Application", "Purpose and Interpretation"],
        "REQ-04": ["Ethics Oversight and Consent Requirements", "Core Elements of Responsible Data Sharing"],
        "REQ-05": ["Data Quality and Security", "Data Security Standards"],
        "REQ-06": ["Accountability", "Data Breach Notification", "Data Quality and Security"],
        "REQ-07": ["Term and Termination", "Data Destruction", "Accountability"],
        "REQ-08": ["Intellectual Property Requirements", "Recognition and Attribution"],
        "REQ-09": ["Ethics Oversight and Consent Requirements", "Privacy, Data Protection and Confidentiality"],
        "REQ-10": ["Privacy, Data Protection and Confidentiality", "Re-identification and Harm"],
        "REQ-11": ["Outbound Data Transfers", "Accountability", "Transparency"],
        "REQ-12": ["Reporting and Monitoring of Use and Access", "Accountability", "Compliance and Oversight"],
        "REQ-13": ["Liability", "Accountability", "Governing Law, Jurisdiction, and Compliance"],
        "REQ-14": ["Recognition and Attribution", "Scientific Publication"],
        "REQ-15": ["Governing Law, Jurisdiction, and Compliance", "Implementation Mechanisms and Amendments"],
    }

    # Relevance weights by requirement–section match quality
    relevance_weights = {
        "exact": 1.0,
        "high": 0.85,
        "medium": 0.65,
    }

    # Create CHECKS relationships (Requirement → PolicyChunk)
    for req in REQUIREMENTS:
        req_id = req["id"]
        target_sections = req_to_framework_sections.get(req_id, [])

        for i, section in enumerate(target_sections):
            weight = relevance_weights["exact"] if i == 0 else relevance_weights["high"] if i == 1 else relevance_weights["medium"]
            session.run(
                """
                MATCH (r:Requirement {id: $req_id})
                MATCH (pc:PolicyChunk)
                WHERE pc.section_title CONTAINS $section
                MERGE (r)-[c:CHECKS]->(pc)
                SET c.relevance_weight = $weight
                """,
                req_id=req_id,
                section=section,
                weight=weight,
            )

        # Create REMEDIATED_BY relationship (Requirement → PolicyFormChunk)
        daa_clause_num = req["daa_clause_number"]
        daa_chunk_id = f"daa_clause_{int(daa_clause_num):02d}"
        session.run(
            """
            MATCH (r:Requirement {id: $req_id})
            MATCH (pfc:PolicyFormChunk {id: $pfc_id})
            MERGE (r)-[rem:REMEDIATED_BY]->(pfc)
            SET rem.specificity = 1.0
            """,
            req_id=req_id,
            pfc_id=daa_chunk_id,
        )

    logger.info("Created CHECKS and REMEDIATED_BY relationships.")


def _create_related_requirements(session) -> None:
    """Create RELATED_TO edges between cross-cutting requirements."""
    for req_a, req_b in RELATED_REQUIREMENTS:
        session.run(
            """
            MATCH (a:Requirement {id: $id_a}), (b:Requirement {id: $id_b})
            MERGE (a)-[:RELATED_TO]->(b)
            MERGE (b)-[:RELATED_TO]->(a)
            """,
            id_a=req_a,
            id_b=req_b,
        )
    logger.info("Created %d RELATED_TO relationships.", len(RELATED_REQUIREMENTS))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for seeding the knowledge graph."""
    logger.info("=== Seeding GA4GH Knowledge Graph ===")

    # Load policy text files
    framework_path = DATA_DIR / "policies" / "ga4gh_framework.md"
    daa_path = DATA_DIR / "policies" / "ga4gh_daa_clauses.md"

    logger.info("Loading policy files...")
    framework_text = framework_path.read_text(encoding="utf-8")
    daa_text = daa_path.read_text(encoding="utf-8")

    # Chunk policy texts
    logger.info("Chunking framework text...")
    framework_chunks = chunk_framework_text(framework_text)
    logger.info("  Framework: %d chunks", len(framework_chunks))

    logger.info("Chunking DAA clauses...")
    daa_chunks = chunk_daa_clauses(daa_text)
    logger.info("  DAA Clauses: %d chunks", len(daa_chunks))

    # Generate embeddings
    openai_client = OpenAI(api_key=settings.openai_api_key)
    framework_chunks = embed_chunks_in_batches(framework_chunks, openai_client)
    daa_chunks = embed_chunks_in_batches(daa_chunks, openai_client)

    # Seed Neo4j
    logger.info("Connecting to Neo4j at %s...", settings.neo4j_uri)
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        driver.verify_connectivity()
        logger.info("Neo4j connection verified.")
        seed_graph(driver, framework_chunks, daa_chunks)
    finally:
        driver.close()

    logger.info("=== Knowledge graph seeding complete ===")
    logger.info("Framework chunks: %d", len(framework_chunks))
    logger.info("DAA clause chunks: %d", len(daa_chunks))
    logger.info("Requirements: %d", len(REQUIREMENTS))
    logger.info("Requirement groups: %d", len(REQUIREMENT_GROUPS))


if __name__ == "__main__":
    main()
