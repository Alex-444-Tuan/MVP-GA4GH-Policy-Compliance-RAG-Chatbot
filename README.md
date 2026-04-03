# GA4GH Policy Compliance RAG Chatbot

A GraphRAG-powered chatbot that validates researcher data-use letters against GA4GH genomic data sharing policies. Upload a letter, get a structured gap report, and receive actionable remediation with fill-in-ready DAA clause templates.

---

## What it does

Researchers requesting access to genomic datasets must submit a data-use letter demonstrating compliance with the [GA4GH Framework for Responsible Sharing of Genomic and Health-Related Data](https://www.ga4gh.org/genomic-data-toolkit/regulatory-ethics-toolkit/framework-for-responsible-sharing-of-genomic-and-health-related-data/) and the [GA4GH Model Data Access Agreement (DAA) Clauses](https://www.ga4gh.org/genomic-data-toolkit/regulatory-ethics-toolkit/model-data-access-agreement/).

This tool:

1. **Ingests** the letter (PDF, DOCX, or plain text)
2. **Retrieves** relevant policy requirements using hybrid search (lexical + semantic + graph traversal)
3. **Analyzes** each of the 15 GA4GH DAA requirements — `FULLY_MET`, `PARTIALLY_MET`, or `NOT_MET`
4. **Produces** a structured gap report with severity ratings (`CRITICAL`, `MAJOR`, `MINOR`)
5. **Generates** actionable remediation — specific DAA clause templates with auto-filled and manual fields
6. **Supports** conversational follow-up to help researchers fix their letters

---

## Architecture

```
Letter (PDF/DOCX/TXT)
        │
        ▼
  Document Parser
  (PyMuPDF / python-docx)
        │
        ▼
  Two-Pass Chunker
  (structural headers → semantic sentence split)
        │
        ▼
┌───────────────────────────────────────────┐
│           Hybrid Retrieval                │
│                                           │
│  ┌─────────────┐   ┌──────────────────┐  │
│  │ Lexical     │   │ Semantic Search  │  │
│  │ (tsvector)  │   │ (pgvector cosine)│  │
│  └──────┬──────┘   └────────┬─────────┘  │
│         └────────┬──────────┘            │
│              RRF Fusion                  │
│         (configurable α/β weights)       │
│              │                           │
│         ┌────▼────────┐                  │
│         │ Graph Search│                  │
│         │ (Neo4j)     │                  │
│         │ Chunk→Req→  │                  │
│         │ DAA Clause  │                  │
│         └─────────────┘                  │
└───────────────────────────────────────────┘
        │
        ▼
  Gap Analysis (Claude Sonnet)
  Per-chunk → aggregate 15 requirements
        │
        ▼
  Coherence Check (Claude Sonnet)
  Cross-chunk contradiction detection
        │
        ▼
  Remediation (Claude Sonnet)
  DAA clause templates with auto-fill
        │
        ▼
  Gap Report + Verdict
  VALID / INVALID_FIXABLE / INVALID_MAJOR_REVISION
```

### Knowledge Graph Schema (Neo4j)

```
(Policy) ←─[:BELONGS_TO]─ (PolicyChunk) ─[:NEXT]→ (PolicyChunk)
                                │
                         [:CHECKS] ↑
                                │
                         (Requirement) ─[:MEMBER_OF]→ (RequirementGroup)
                                │
                     [:REMEDIATED_BY] ↓
                                │
(PolicyForm) ←─[:BELONGS_TO]─ (PolicyFormChunk)
```

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI + uvicorn |
| Frontend UI | Streamlit |
| Knowledge graph | Neo4j 5.x Community |
| Vector + lexical search | PostgreSQL 14 + pgvector |
| Embeddings | OpenAI `text-embedding-3-large` (3072 dims) |
| LLM analysis | Anthropic `claude-sonnet-4-6` |
| LLM preprocessing | Anthropic `claude-haiku-4-5-20251001` |
| Document parsing | PyMuPDF (`fitz`), `python-docx` |
| PDF export | fpdf2 |

---

## The 15 Compliance Requirements

| ID | Description | Severity | DAA Clause |
|----|-------------|----------|------------|
| REQ-01 | Data use limited to stated research purpose | CRITICAL | Clause 2: Purposes of Use |
| REQ-02 | Principal investigator and team members identified | CRITICAL | Clause 1: Definitions |
| REQ-03 | Institutional affiliation and authority established | MAJOR | Clause 1: Definitions |
| REQ-04 | IRB / ethics committee approval documented | CRITICAL | Clause 2: Ethics Oversight |
| REQ-05 | Data security measures specified | CRITICAL | Clause 11: Data Security |
| REQ-06 | Data breach notification procedures defined | CRITICAL | Clause 12: Breach Notification |
| REQ-07 | Data destruction / return plan stated | CRITICAL | Clause 10: Data Destruction |
| REQ-08 | Intellectual property rights for derived works addressed | CRITICAL | Clause 4: IP Requirements |
| REQ-09 | Consent basis confirmed compatible with proposed use | MAJOR | Clause 2: Purposes of Use |
| REQ-10 | Re-identification attempts explicitly prohibited | MAJOR | Clause 8: Re-identification |
| REQ-11 | Third-party sharing and sub-licensing restrictions stated | MAJOR | Clause 5: Outbound Transfers |
| REQ-12 | Compliance monitoring and audit provisions acknowledged | MAJOR | Clause 3: Reporting & Monitoring |
| REQ-13 | Sanctions for non-compliance acknowledged | MINOR | Clause 14: Liability |
| REQ-14 | Data source attribution and recognition stated | MINOR | Clause 9: Scientific Publication |
| REQ-15 | Governing law and jurisdiction specified | MINOR | Clause 15: Duration |

---

## Verdict logic

| Verdict | Condition |
|---------|-----------|
| `VALID` | Zero `NOT_MET` requirements |
| `INVALID_FIXABLE` | Has `NOT_MET` requirements, but none are `CRITICAL` |
| `INVALID_MAJOR_REVISION` | Any `CRITICAL` requirement is `NOT_MET` |

---

## Quick start

```bash
cp .env.example .env          # Add your API keys
docker compose up -d          # Start Neo4j + PostgreSQL
pip install -r requirements.txt
python scripts/seed_knowledge_graph.py
python scripts/seed_postgres.py

# Terminal 1
uvicorn src.api.main:app --reload --port 8000

# Terminal 2
streamlit run src/ui/streamlit_app.py --server.port 8501
```

See [SETUP.md](SETUP.md) for detailed setup instructions and troubleshooting.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Upload letter file (PDF/DOCX/TXT) → gap report |
| `POST` | `/analyze/text` | Submit letter as plain JSON text → gap report |
| `GET` | `/session/{id}` | Retrieve stored analysis session |
| `POST` | `/followup` | Conversational follow-up on a session |
| `GET` | `/health` | Health check |

Interactive API docs: **http://localhost:8000/docs**

---

## Project structure

```
├── data/
│   ├── policies/          # GA4GH Framework + DAA Clauses (markdown)
│   ├── test_letters/      # Sample letters for testing and evaluation
│   └── lexicon/           # Domain keyword lexicon for retrieval boosting
├── scripts/
│   ├── seed_knowledge_graph.py   # Populate Neo4j
│   ├── seed_postgres.py          # Populate PostgreSQL
│   └── run_evaluation.py         # Precision/recall evaluation
├── src/
│   ├── config.py                 # Settings via pydantic-settings
│   ├── ingestion/                # Document parsing + chunking
│   ├── retrieval/                # Lexical, semantic, graph, RRF fusion
│   ├── analysis/                 # Gap detection, coherence, remediation
│   ├── models/                   # Pydantic schemas + SQLAlchemy models
│   ├── api/                      # FastAPI app, routes, dependencies
│   └── ui/                       # Streamlit frontend
├── tests/                        # Unit + integration + e2e tests
└── evaluation/                   # Ground truth annotations + metrics
```

---

## Running tests

```bash
# Unit tests only (no databases needed)
pytest tests/ -m "not integration" -v

# All tests (requires running DBs + seeded data)
pytest tests/ -v

# Evaluation against annotated test cases
python scripts/run_evaluation.py
```

---

## Policy sources

- [GA4GH Framework for Responsible Sharing of Genomic and Health-Related Data](https://www.ga4gh.org/genomic-data-toolkit/regulatory-ethics-toolkit/framework-for-responsible-sharing-of-genomic-and-health-related-data/) (v1.0, reaffirmed 2019)
- [GA4GH Model Data Access Agreement Clauses](https://www.ga4gh.org/genomic-data-toolkit/regulatory-ethics-toolkit/model-data-access-agreement/) (approved November 2024)
