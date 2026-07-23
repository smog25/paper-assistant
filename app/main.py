import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CROSSREF_EMAIL, USE_SEMANTIC, ALLOWED_ORIGINS
from app.routers import health, pdf, analysis, papers, projects

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        timeout=20.0,
        headers={"User-Agent": f"ResearchAssistant/1.0 (mailto:{CROSSREF_EMAIL})"},
    )
    if USE_SEMANTIC:
        from app.services.embeddings import get_specter_model
        get_specter_model()
        logger.info("SPECTER model loaded")
    logger.info("Started ResearchAssistant backend")
    yield
    await app.state.http.aclose()
    logger.info("Shutdown ResearchAssistant backend")


app = FastAPI(title="Research Assistant API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # A wildcard origin cannot be combined with credentials — browsers
    # reject that response. Wildcard therefore means "public cookie-less
    # API"; explicit origins get credentials for future authed requests.
    allow_credentials="*" not in ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(pdf.router)
app.include_router(analysis.router)
app.include_router(papers.router)
app.include_router(projects.router)
