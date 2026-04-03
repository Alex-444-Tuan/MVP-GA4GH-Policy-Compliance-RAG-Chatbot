"""Neo4j graph traversal: PolicyChunk → Requirements → PolicyFormChunks.

The traversal pattern:
    (PolicyChunk)<-[:CHECKS]-(Requirement)-[:REMEDIATED_BY]->(PolicyFormChunk)

Starting from the top-K semantic search hits (PolicyChunk IDs), we traverse
the graph to find connected Requirements and their remediation clauses.
"""

from __future__ import annotations

import logging

from neo4j import AsyncDriver, AsyncGraphDatabase

from src.config import get_settings
from src.models.schemas import PolicyFormChunkResult, RequirementNode, Severity

logger = logging.getLogger(__name__)
settings = get_settings()


async def graph_traversal(
    chunk_ids: list[str],
    driver: AsyncDriver,
) -> tuple[list[RequirementNode], list[PolicyFormChunkResult]]:
    """Traverse the Neo4j graph from PolicyChunk hits to Requirements and DAA clauses.

    For each PolicyChunk ID in chunk_ids:
        MATCH (pc)<-[c:CHECKS]-(r:Requirement)-[rem:REMEDIATED_BY]->(pfc:PolicyFormChunk)
        RETURN r, pfc, c.relevance_weight

    Deduplicates requirements and form chunks. Re-ranks form chunks by
    maximum relevance_weight across all paths that reach them.

    Args:
        chunk_ids: PolicyChunk IDs from the top-K semantic search results.
        driver: Async Neo4j driver instance.

    Returns:
        Tuple of (requirements, form_chunks) — deduplicated and sorted by relevance.
    """
    if not chunk_ids:
        return [], []

    cypher = """
        MATCH (pc:PolicyChunk {id: $chunk_id})<-[c:CHECKS]-(r:Requirement)-[rem:REMEDIATED_BY]->(pfc:PolicyFormChunk)
        RETURN
            r.id AS req_id,
            r.description AS req_description,
            r.severity AS req_severity,
            r.keywords AS req_keywords,
            r.daa_clause_category AS req_daa_category,
            r.daa_clause_number AS req_daa_number,
            pfc.id AS pfc_id,
            pfc.clause_category AS pfc_category,
            pfc.clause_number AS pfc_number,
            pfc.text AS pfc_text,
            c.relevance_weight AS relevance_weight
        ORDER BY c.relevance_weight DESC
    """

    req_map: dict[str, RequirementNode] = {}
    pfc_weight_map: dict[str, float] = {}
    pfc_map: dict[str, PolicyFormChunkResult] = {}

    async with driver.session() as session:
        for chunk_id in chunk_ids[:3]:  # Only top-3 chunks for graph traversal per spec
            try:
                result = await session.run(cypher, chunk_id=chunk_id)
                records = await result.data()
            except Exception as e:
                logger.error("Graph traversal failed for chunk %s: %s", chunk_id, e)
                continue

            for record in records:
                req_id = record["req_id"]
                pfc_id = record["pfc_id"]
                relevance = float(record.get("relevance_weight", 0.5))

                # Collect unique requirements
                if req_id not in req_map:
                    req_map[req_id] = RequirementNode(
                        id=req_id,
                        description=record["req_description"],
                        severity=Severity(record["req_severity"]),
                        keywords=list(record.get("req_keywords") or []),
                        daa_clause_category=record["req_daa_category"],
                        daa_clause_number=record.get("req_daa_number"),
                    )

                # Track max relevance weight per form chunk
                if pfc_id not in pfc_weight_map or relevance > pfc_weight_map[pfc_id]:
                    pfc_weight_map[pfc_id] = relevance
                    pfc_map[pfc_id] = PolicyFormChunkResult(
                        chunk_id=pfc_id,
                        clause_category=record["pfc_category"],
                        clause_number=str(record["pfc_number"]),
                        text=record["pfc_text"],
                        relevance_weight=relevance,
                    )

    requirements = list(req_map.values())
    form_chunks = sorted(pfc_map.values(), key=lambda x: x.relevance_weight, reverse=True)

    logger.debug(
        "Graph traversal from %d chunks → %d requirements, %d DAA clauses",
        len(chunk_ids), len(requirements), len(form_chunks),
    )
    return requirements, form_chunks


async def get_all_requirements(driver: AsyncDriver) -> list[RequirementNode]:
    """Fetch all 15 requirements from Neo4j. Used as fallback when retrieval finds nothing.

    Args:
        driver: Async Neo4j driver instance.

    Returns:
        All Requirement nodes from the graph.
    """
    cypher = """
        MATCH (r:Requirement)
        RETURN r.id AS id, r.description AS description, r.severity AS severity,
               r.keywords AS keywords, r.daa_clause_category AS daa_clause_category,
               r.daa_clause_number AS daa_clause_number
        ORDER BY r.id
    """
    async with driver.session() as session:
        result = await session.run(cypher)
        records = await result.data()

    return [
        RequirementNode(
            id=r["id"],
            description=r["description"],
            severity=Severity(r["severity"]),
            keywords=list(r.get("keywords") or []),
            daa_clause_category=r["daa_clause_category"],
            daa_clause_number=r.get("daa_clause_number"),
        )
        for r in records
    ]


def create_async_driver(uri: str, user: str, password: str) -> AsyncDriver:
    """Create a Neo4j async driver. Caller is responsible for closing it.

    Args:
        uri: Neo4j bolt URI.
        user: Neo4j username.
        password: Neo4j password.

    Returns:
        AsyncDriver instance.
    """
    return AsyncGraphDatabase.driver(uri, auth=(user, password))
