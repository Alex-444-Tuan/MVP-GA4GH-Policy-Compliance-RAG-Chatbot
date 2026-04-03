# Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Use conda or pyenv |
| PostgreSQL | 14+ | Local install or Docker |
| pgvector | 0.5+ | PostgreSQL extension |
| Neo4j | 5.x Community | Local install or Docker |
| Docker + Compose | Latest | Optional but recommended |

---

## 1. Clone and install dependencies

```bash
cd MVP_Project
pip install -r requirements.txt
```

---

## 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```env
ANTHROPIC_API_KEY=sk-ant-...     # Required: Claude analysis + keyword extraction
OPENAI_API_KEY=sk-...            # Required: text-embedding-3-large (3072 dims)
```

Everything else can stay at the defaults for local development.

---

## 3. Start databases

### Option A — Docker (recommended)

```bash
docker compose up -d neo4j postgres
```

Wait ~20 seconds for health checks to pass:

```bash
docker compose ps   # both should show "healthy"
```

### Option B — Local PostgreSQL

If you have a local PostgreSQL running on port 5432:

```bash
# Create user and database
psql postgres -c "CREATE USER rag_user WITH PASSWORD 'rag_password';"
psql postgres -c "CREATE DATABASE ga4gh_rag OWNER rag_user;"

# Install pgvector extension (requires superuser)
psql -d ga4gh_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

> **pgvector not found?** Install it first:
> ```bash
> brew install pgvector
> ```
> Then restart PostgreSQL and repeat the `CREATE EXTENSION` command.

### Option B — Local Neo4j

Download [Neo4j Desktop](https://neo4j.com/download/) or install via Homebrew:

```bash
brew install neo4j
neo4j start
```

Default credentials: `neo4j` / `neo4j` (you'll be prompted to change on first login).
Update `NEO4J_PASSWORD` in your `.env` accordingly.

---

## 4. Seed the databases (one-time)

This step parses the policy documents, generates embeddings (~$0.30 in OpenAI costs), and populates both databases.

```bash
# Seed Neo4j knowledge graph (nodes, relationships, embeddings)
python scripts/seed_knowledge_graph.py

# Seed PostgreSQL (tsvector + pgvector columns)
python scripts/seed_postgres.py
```

Expected output for each:
```
=== Seeding complete ===
Framework chunks: 20
DAA clause chunks: 15
Requirements: 15
```

> You only need to run these once. Re-running is safe (uses MERGE / ON CONFLICT).

---

## 5. Run the application

Open **two terminals**:

**Terminal 1 — API server:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 — Streamlit UI:**
```bash
streamlit run src/ui/streamlit_app.py --server.port 8501
```

Open your browser at **http://localhost:8501**

The API docs (Swagger) are at **http://localhost:8000/docs**

---

## 6. Verify everything works

```bash
# Unit tests (no databases required)
pytest tests/ -m "not integration" -v

# Integration tests (requires live DBs + seeded data)
pytest tests/ -v

# Evaluation suite (requires API server running)
python scripts/run_evaluation.py
```

---

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `role "rag_user" does not exist` | Local Postgres, no user created | Run Step 3 Option B commands |
| `permission denied to create extension` | Not a superuser | Run `CREATE EXTENSION` as your system user: `psql -d ga4gh_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"` |
| `column cannot have more than 2000 dimensions for ivfflat index` | pgvector ivfflat limit | Already fixed in `seed_postgres.py` — sequential scan is used instead |
| `Cannot connect to API at http://localhost:8000` | API server not running | Start with `uvicorn src.api.main:app --reload --port 8000` |
| `ModuleNotFoundError: No module named 'neo4j'` | Dependencies not installed | `pip install -r requirements.txt` |
| `ServiceUnavailable: Failed to establish connection to Neo4j` | Neo4j not running | `docker compose up -d neo4j` or start Neo4j Desktop |
