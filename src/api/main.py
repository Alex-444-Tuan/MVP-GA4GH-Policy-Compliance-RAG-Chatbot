"""FastAPI application factory with lifespan management.

Initializes database connections and API clients on startup,
tears them down cleanly on shutdown.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import anthropic
import asyncpg
import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neo4j import AsyncGraphDatabase

from src.api.routes import router
from src.config import configure_logging, get_settings

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and clean up all shared resources."""
    logger.info("Starting up GA4GH Compliance API...")

    # PostgreSQL connection pool
    dsn = settings.postgres_dsn.replace("postgresql+asyncpg://", "postgresql://")
    app.state.pg_pool = await asyncpg.create_pool(
        dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    logger.info("PostgreSQL pool initialized.")

    # Neo4j async driver
    app.state.neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_pool_size=10,
    )
    logger.info("Neo4j driver initialized.")

    # API clients (stateless — shared across requests)
    app.state.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    app.state.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
    logger.info("LLM clients initialized.")

    yield

    # Cleanup
    logger.info("Shutting down...")
    await app.state.pg_pool.close()
    await app.state.neo4j_driver.close()
    logger.info("Connections closed.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="GA4GH Policy Compliance API",
        description="GraphRAG-based compliance checking for genomic data-use letters",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
