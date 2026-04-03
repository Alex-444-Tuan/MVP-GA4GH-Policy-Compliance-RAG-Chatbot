"""FastAPI dependency injection for database connections and API clients.

All dependencies are designed to be injected via FastAPI's Depends() system.
Database connections are pooled and reused across requests.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

import anthropic
import asyncpg
import openai
from fastapi import Request
from neo4j import AsyncDriver

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── PostgreSQL pool ───────────────────────────────────────────────────────────


async def get_pg_conn(request: Request) -> AsyncGenerator[asyncpg.Connection, None]:
    """Yield a PostgreSQL connection from the application connection pool.

    The pool is initialized in the FastAPI lifespan and stored in app.state.
    """
    pool: asyncpg.Pool = request.app.state.pg_pool
    async with pool.acquire() as conn:
        yield conn


# ── Neo4j driver ──────────────────────────────────────────────────────────────


def get_neo4j_driver(request: Request) -> AsyncDriver:
    """Return the Neo4j async driver from application state.

    The driver is initialized in the FastAPI lifespan.
    """
    return request.app.state.neo4j_driver


# ── Anthropic client ──────────────────────────────────────────────────────────


def get_anthropic_client(request: Request) -> anthropic.Anthropic:
    """Return the shared Anthropic client from application state."""
    return request.app.state.anthropic_client


# ── OpenAI client ─────────────────────────────────────────────────────────────


def get_openai_client(request: Request) -> openai.OpenAI:
    """Return the shared OpenAI client from application state."""
    return request.app.state.openai_client
