"""Application configuration loaded from environment variables via pydantic-settings."""

import logging
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings. Loaded once at startup via get_settings()."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM APIs ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password123")

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_dsn: str = Field(
        default="postgresql+asyncpg://rag_user:rag_password@localhost:5432/ga4gh_rag"
    )

    # ── Retrieval config ──────────────────────────────────────────────────────
    rrf_lexical_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, gt=0)
    retrieval_top_k: int = Field(default=5, gt=0)

    # ── Model config ──────────────────────────────────────────────────────────
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_dims: int = Field(default=3072)
    analysis_model: str = Field(default="claude-sonnet-4-6")
    preprocessing_model: str = Field(default="claude-haiku-4-5-20251001")

    # ── App config ────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    ui_port: int = Field(default=8501)
    log_level: str = Field(default="INFO")
    api_base_url: str = Field(default="http://localhost:8000")

    # ── LLM call config ───────────────────────────────────────────────────────
    llm_timeout_seconds: int = Field(default=60)
    llm_max_retries: int = Field(default=3)

    @field_validator("rrf_lexical_weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Ensure lexical weight is in [0, 1]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("rrf_lexical_weight must be between 0.0 and 1.0")
        return v

    @property
    def rrf_semantic_weight(self) -> float:
        """Semantic weight is complement of lexical weight."""
        return 1.0 - self.rrf_lexical_weight


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
