# AIRA — AI Research Assistant

AIRA analyzes a research-paper PDF for integrity and reproducibility signals. Upload a
paper and it extracts the text (with an OCR fallback for scanned PDFs), verifies every
reference against [Crossref](https://www.crossref.org/) with confidence scoring, extracts
statistical indicators (p-values, sample sizes, effect sizes, confidence intervals),
computes a heuristic open-science transparency score (0–100, letter grade A–F), and
persists the result to a local library where papers can be grouped into projects and
compared by semantic similarity ([SPECTER](https://github.com/allenai/specter) embeddings)
and related work ([OpenAlex](https://openalex.org/)). It is a single-user, local-first
tool aimed at researchers and engineers who want a fast triage signal on a paper's
citation integrity and transparency — not a substitute for peer review.

> The integrity score is a **heuristic and is not empirically validated.** It is a
> preliminary signal, framed as transparency markers — never a verdict on a paper or its
> authors. See [Limitations](#limitations--current-state).

## Features

Everything below is implemented and running today:

- **PDF text extraction** via `pypdf`, with an **OCR fallback** (Tesseract + Poppler) for
  scanned documents.
- **Reference verification against Crossref** — DOI lookup and fuzzy/semantic title
  matching, each match confidence-scored and labelled `verified` / `suspicious` /
  `not_found`.
- **In-text citation detection** — numeric (`[12]`) and author–year (`(Smith, 2020)`)
  markers.
- **Statistics extraction** — p-values, sample sizes, effect sizes, and confidence
  intervals, with heuristic red flags.
- **Heuristic integrity score** — 0–100 with an A–F grade, derived from open-science
  transparency markers (limitations, conflict-of-interest and ethics statements,
  pre-registration, power analysis, open data/code, replication).
- **Persistent library** — ingested papers are deduplicated by SHA-256 and stored in
  SQLite, with the full report, normalized references, and an embedding vector.
- **Projects** — group papers into projects (many-to-many).
- **Semantic similarity** — cosine similarity over stored SPECTER vectors, scoped to the
  library or a single project.
- **Related work** — OpenAlex `related_works` for a paper's DOI (cached, polite pool).

### Interfaces

- **React + Vite frontend** (`frontend/`) — the primary UI.
- **Streamlit app** (`app_v2.py`) — a secondary interface that calls the same API. Still
  functional but **frozen** (no further investment).
- **`app_legacy.py`** — the original standalone Streamlit prototype (OpenAI-dependent,
  pre-API), superseded by `app_v2.py` and the React frontend; kept for reference.

## Architecture

```
User → React frontend (Vite)  :5173 ──┐
User → Streamlit (app_v2.py)  :8501 ──┼──▶ FastAPI (app/ package, :8000) ──▶ Crossref API
                                      │     via backend.py (3-line shim) ──▶ OpenAlex API
                                      └────────────────────────────────────▶ SQLite (data/aira.db) + PDFs (data/pdfs/)
```

`backend.py` is a three-line shim (`from app.main import app`) so `uvicorn backend:app`,
CI, and the Streamlit app all keep working after the code was split into the `app/`
package (`main.py` for lifespan + CORS, `config.py`, `routers/`, `services/`, `schemas/`,
`db/`, `utils/`).

### Ingest request path

`POST /api/papers` runs the full pipeline:

1. **Hash & dedup** — SHA-256 of the PDF bytes; an existing hash short-circuits and
   returns the stored paper (`200`), otherwise ingest proceeds (`201`).
2. **Store** — the PDF is written to `data/pdfs/<sha256>.pdf`.
3. **Extract** — native text extraction (or OCR fallback), plus section detection.
4. **Analyze** — references, in-text citations, statistics, and the integrity score run
   concurrently via `asyncio.gather`; reference verification calls Crossref (bounded by a
   semaphore).
5. **Embed** — a SPECTER vector is computed off the event loop (`asyncio.to_thread`).
6. **Persist** — `Paper`, `AnalysisReport`, normalized `Reference` rows, and the
   `Embedding` are written in one transaction.

### Data model (6 tables)

| Table | Model | Notes |
|-------|-------|-------|
| `papers` | `Paper` | SHA-256 (unique/dedup), title, authors, year, DOI, `pdf_path`, full text, sections, counts, extraction method |
| `analysis_reports` | `AnalysisReport` | full report JSON + promoted `integrity_score`/`integrity_grade` columns + `analyzer_version` |
| `references` | `Reference` | normalized per-reference rows (raw text, DOI, matched title, status, confidence) |
| `projects` | `Project` | name, description |
| `project_papers` | `ProjectPaper` | **many-to-many join** between `projects` and `papers` (composite PK, cascade delete) |
| `embeddings` | `Embedding` | SPECTER vector stored as `LargeBinary`, unique per `(paper_id, kind)` |

Schema is managed by Alembic (`alembic/versions/cd28c00f66a0_initial_schema.py`).

### Startup sequence

- **Backend lifespan** (`app/main.py`): opens a shared `httpx.AsyncClient` and, when
  `USE_SEMANTIC_MATCHING=1` (default), loads the SPECTER model once at startup.
- **Migrations are applied separately** — `alembic upgrade head` runs on Docker container
  start (see the `Dockerfile` `CMD`) or is run manually in local development. They are
  *not* run inside the lifespan hook.

## Tech Stack

Backend versions are pinned in `requirements.lock.txt` / `requirements-dev.lock.txt`
(compiled with pip-tools; CI installs from the lockfiles). Frontend uses caret ranges in
`frontend/package.json`.

| Layer | Key dependencies |
|-------|------------------|
| **Runtime** | Python 3.11 |
| **API** | FastAPI 0.128.0, Uvicorn 0.29.0, Pydantic 2.7.1, python-multipart 0.0.18 |
| **Persistence** | SQLAlchemy 2.0.51, Alembic 1.18.5, SQLite |
| **PDF / OCR** | pypdf 6.6.0, pytesseract 0.3.10, pdf2image 1.17.0, Pillow 10.3.0 (Tesseract + Poppler system binaries) |
| **ML / embeddings** | sentence-transformers 5.6.0 (`allenai-specter`, 768-dim), torch 2.12.1 (CPU), transformers 5.13.0, numpy 1.26.4 |
| **External data** | httpx 0.27.0 → Crossref, OpenAlex |
| **Frontend** | React 18.2, Vite 6.1, Tailwind 3.4, TypeScript 7.0, @tanstack/react-query 5.101, react-router-dom 7.12, ~40 Radix UI primitives (shadcn/ui) |
| **Secondary UI** | Streamlit 1.35.0 (`app_v2.py`) |
| **Optional AI** | openai 1.35.7 — key-gated; powers the GPT "Smart Summary" in `app_v2.py` only when `OPENAI_API_KEY` is set. The FastAPI backend does not use it. |
| **Tests** | pytest 9.1.1, pytest-asyncio 1.4.0 |

## Setup

### Prerequisites

Install the OCR/PDF system binaries once:

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
```

Copy `.env.example` to `.env`. The only variable that matters for a basic run is
`CROSSREF_EMAIL` (sent in the Crossref User-Agent header for API politeness).

### Local development

**Backend:**

```bash
python3.11 -m venv .venv --clear
source .venv/bin/activate
pip install -r requirements.lock.txt
alembic upgrade head
uvicorn backend:app --reload --port 8000
# verify:
curl http://localhost:8000/healthz
```

**Frontend (primary UI):**

```bash
cd frontend
npm install
npm run dev        # → http://localhost:5173
```

**Streamlit (secondary UI):**

```bash
source .venv/bin/activate
streamlit run app_v2.py    # → http://localhost:8501
```

### Docker (backend only)

```bash
docker compose up --build
```

The container runs `alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port
8000` on start. The first start downloads the SPECTER model (~500 MB) into
`./data/hf-cache`; the SQLite DB, stored PDFs, and model cache all persist via the single
`./data` bind mount. torch is CPU-pinned to the lockfile version.

> **Note:** the Docker setup is **not build-verified** — it was authored without a Docker
> daemon on the development machine.

## API Reference

Base URL: `http://localhost:8000`. **21 endpoints across 5 routers.**

**Health** (`app/routers/health.py`)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/healthz` | Health check |
| GET | `/version` | Dependency version info |

**PDF** (`app/routers/pdf.py`)

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/parse_pdf` | Extract text (native pypdf) |
| POST | `/api/parse_pdf_ocr` | Extract text via OCR (Tesseract) |

**Analysis** (`app/routers/analysis.py`) — stateless sub-analyses

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/verify_references` | Verify a reference list against Crossref |
| POST | `/api/find_intext_citations` | Extract in-text citation markers |
| POST | `/api/extract_statistics` | Extract p-values, sample sizes, effect sizes, CIs |
| POST | `/api/truthiness_score` | Compute the heuristic integrity score |
| POST | `/api/analyze_pdf` | Full stateless pipeline (combined) |

**Papers** (`app/routers/papers.py`)

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/papers` | Ingest a PDF (dedup + persist); `201` new / `200` duplicate |
| GET | `/api/papers` | List papers (paginated; optional `project_id` filter) |
| GET | `/api/papers/{id}` | Paper detail + stored integrity report |
| GET | `/api/papers/{id}/similar` | Cosine similarity over stored SPECTER vectors |
| GET | `/api/papers/{id}/related` | OpenAlex related works (cached) |

**Projects** (`app/routers/projects.py`)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/projects` | List projects with paper counts |
| POST | `/api/projects` | Create a project |
| GET | `/api/projects/{id}` | Get a project with its paper IDs |
| PUT | `/api/projects/{id}` | Update a project |
| DELETE | `/api/projects/{id}` | Delete a project (cascades membership) |
| POST | `/api/projects/{id}/papers` | Add a paper to a project |
| DELETE | `/api/projects/{id}/papers/{paper_id}` | Remove a paper from a project |

## Testing

```bash
pip install -r requirements-dev.lock.txt
pytest tests/ -v
```

**88 tests across 8 files:**

| File | Tests |
|------|-------|
| `test_backend.py` | 17 |
| `test_ingest.py` | 7 |
| `test_models.py` | 8 |
| `test_papers_read.py` | 16 |
| `test_projects.py` | 19 |
| `test_references.py` | 3 |
| `test_related.py` | 10 |
| `test_similar.py` | 8 |

Tests run against an in-memory SQLite database (`sqlite:///:memory:`) with FastAPI's
`TestClient`; external HTTP (Crossref/OpenAlex) is stubbed, so the suite needs no network.

CI (`.github/workflows/test.yml`) runs on **push and pull requests to `main`**: Python
3.11, installs the Tesseract/Poppler system packages and `requirements-dev.lock.txt`, then
runs `pytest tests/ -v`.

## Design Decisions

- **Brute-force numpy cosine over BLOB-stored SPECTER vectors — not FAISS/sqlite-vec.**
  At library scale (10²–10³ papers × 768-dim), a dot product over vectors loaded from
  SQLite is milliseconds and dependency-free. The documented upgrade path is
  sqlite-vec/FAISS past roughly 10k papers — a deliberate choice, not an oversight.
- **SQLite + SQLAlchemy + Alembic.** Zero-setup single-file persistence for reviewers,
  with a cheap path to Postgres if it's ever needed. Alembic from day one because the
  schema keeps evolving.
- **Local-first, keyless core.** The core intelligence (SPECTER embeddings, Crossref
  verification) runs locally with no API key. Any LLM-backed features are optional and
  activate only when a key is configured.
- **`analyzer_version` on every report.** Reports are immutable and stamped with the
  analyzer version so results stay reproducible as the heuristics change.
- **Package split with a shim.** The backend was split from a single file into the `app/`
  package, keeping `backend.py` as a three-line shim so `uvicorn backend:app`, CI, and the
  Streamlit app didn't have to change.

## Limitations & Current State

Two caveats change how you should read the tool's output:

1. **The integrity score is a heuristic and is not empirically validated.** It is a
   preliminary signal, not peer review — presented as open-science transparency markers,
   never a verdict on a paper or its authors.
2. **Title / author / year extraction is a crude first-line heuristic.** There is no
   GROBID or Crossref/OpenAlex metadata lookup; titles are approximate and authors/year
   are often left unpopulated at ingest.

Engineering state, in brief: no test-coverage measurement; single-user with no auth; the
add-to-project picker fetches the whole library (fine at current scale, won't scale past a
few hundred papers). The Streamlit path is frozen and `app_legacy.py` is archival; the
Docker setup is not build-verified.

This is a single-user, local-first tool and an actively developed portfolio project; a
frontend redesign is currently in progress. Longer-term, exploratory product thinking
lives in [VISION.md](VISION.md) — treat it as direction under evaluation, **not a
committed roadmap.**
