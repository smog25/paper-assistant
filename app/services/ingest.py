import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ANALYZER_VERSION, EMBEDDING_DTYPE, USE_SEMANTIC
from app.db.models import AnalysisReport, Embedding, Paper, Reference
from app.schemas.models import IntegrityReport, PaperIngestResponse, StatisticsResult
from app.services import citations as citations_svc
from app.services import integrity as integrity_svc
from app.services import references as references_svc
from app.services import statistics as statistics_svc
from app.services.pdf import detect_sections, extract_text_from_pdf_bytes
from app.utils.helpers import extract_first_doi

logger = logging.getLogger(__name__)

PDF_DIR = Path("data/pdfs")

# Non-title lines commonly found in PDF front matter before the real title.
_TITLE_SKIP_PREFIXES = ("doi", "10.", "arxiv", "http", "www.", "issn", "vol.", "©")
_TITLE_SKIP_RE = re.compile(
    r"(?i)\bjournal of\b|\bproceedings\b|\bvol\.?\s*\d|\bpp\.\s*\d|received:|accepted:"
)


def _extract_title(full_text: str, sections: dict, filename: str) -> str:
    """Crude title heuristic: first substantial line before the Abstract.

    Falls back to the uploaded filename. Known-imprecise; tracked in the
    CLAUDE.md backlog for a proper metadata lookup later.
    """
    head_end = min(sections.get("Abstract", 500), 1000)
    lines = full_text[:head_end].splitlines()
    # The slice can cut a line mid-word; a truncated title must never be stored.
    if head_end < len(full_text) and full_text[head_end] not in "\r\n" and lines:
        lines.pop()
    for line in lines:
        line = line.strip()
        if not 20 <= len(line) <= 300:
            continue
        if line.lower().startswith(_TITLE_SKIP_PREFIXES):
            continue
        if _TITLE_SKIP_RE.search(line):
            continue
        letters = sum(c.isalpha() for c in line)
        if letters < len(line) * 0.5:
            continue
        return line
    return Path(filename).stem


async def ingest_paper(
    db: Session,
    http: Optional[httpx.AsyncClient],
    pdf_bytes: bytes,
    filename: str,
    pdf_dir: Path = PDF_DIR,
) -> PaperIngestResponse:
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    # --- Dedup check ---
    existing = db.execute(select(Paper).where(Paper.sha256 == sha256)).scalar_one_or_none()
    if existing:
        report_row = db.execute(
            select(AnalysisReport)
            .where(AnalysisReport.paper_id == existing.id)
            .order_by(AnalysisReport.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        report = None
        if report_row:
            try:
                report = IntegrityReport.model_validate_json(report_row.report_json)
            except Exception:
                pass
        return PaperIngestResponse(
            paper_id=existing.id,
            sha256=sha256,
            was_duplicate=True,
            report=report,
        )

    # --- Save PDF ---
    pdf_dir.mkdir(parents=True, exist_ok=True)
    (pdf_dir / f"{sha256}.pdf").write_bytes(pdf_bytes)
    pdf_path = str(pdf_dir / f"{sha256}.pdf")

    # --- Extract text ---
    try:
        full_text, num_pages = extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as e:
        logger.error(f"PDF parse error during ingest: {e}")
        full_text, num_pages = "", 0

    sections = detect_sections(full_text)
    word_count = len(full_text.split())

    # --- Run all analysis concurrently ---
    if len(full_text) >= 100 and http is not None:
        cites, stats, truthiness, intext = await asyncio.gather(
            references_svc.verify_references(http, full_text),
            statistics_svc.extract_statistics(full_text),
            integrity_svc.calculate_truthiness(full_text),
            citations_svc.find_intext_citations(full_text),
        )
    elif len(full_text) >= 100:
        # http not available (e.g. tests without a live client)
        stats, truthiness, intext = await asyncio.gather(
            statistics_svc.extract_statistics(full_text),
            integrity_svc.calculate_truthiness(full_text),
            citations_svc.find_intext_citations(full_text),
        )
        cites = []
    else:
        from app.schemas.models import InTextCitations
        cites = []
        stats = StatisticsResult(
            p_values=[], sample_sizes=[], effect_sizes=[], cis=[],
            red_flags=[], summary="OCR required - no text extracted",
        )
        truthiness = await integrity_svc.calculate_truthiness("")
        intext = InTextCitations(numeric=[], author_year=[], total_count=0,
                                 explanation="No text extracted")

    # --- Build IntegrityReport ---
    verified = sum(1 for c in cites if c.status == "verified")
    suspicious = sum(1 for c in cites if c.status == "suspicious")
    not_found = sum(1 for c in cites if c.status == "not_found")

    report = IntegrityReport(
        pages=num_pages,
        word_count=word_count,
        sections_detected=list(sections.keys()),
        extraction_method="native" if len(full_text) >= 100 else "failed_needs_ocr",
        references_found=len(cites),
        references_verified=verified,
        references_suspicious=suspicious,
        references_not_found=not_found,
        citations=cites,
        statistics=stats,
        integrity_score=truthiness.score,
        integrity_grade=truthiness.grade,
        integrity_signals=truthiness.reasons,
        intext_total=intext.total_count,
        intext_numeric=len(intext.numeric),
        intext_author_year=len(intext.author_year),
        analyzed_at=datetime.utcnow().isoformat(),
        disclaimer="This is an automated heuristic analysis, not peer review.",
    )

    # --- Persist in a single transaction ---
    # Don't bisect a DOI at the head cut — extend to the end of the current token.
    doi_cut = 3000
    while doi_cut < len(full_text) and not full_text[doi_cut].isspace():
        doi_cut += 1

    paper = Paper(
        sha256=sha256,
        title=_extract_title(full_text, sections, filename),
        doi=extract_first_doi(full_text[:doi_cut]),
        pdf_path=pdf_path,
        full_text=full_text,
        sections=json.dumps(sections),
        word_count=word_count,
        extraction_method=report.extraction_method,
    )
    db.add(paper)
    db.flush()  # get paper.id before adding children

    db.add(AnalysisReport(
        paper_id=paper.id,
        report_json=report.model_dump_json(),
        integrity_score=report.integrity_score,
        integrity_grade=report.integrity_grade,
        analyzer_version=ANALYZER_VERSION,
    ))

    for cite in cites:
        db.add(Reference(
            paper_id=paper.id,
            raw_text=cite.raw_text,
            doi=cite.doi,
            matched_title=cite.title,
            status=cite.status,
            confidence=cite.confidence,
        ))

    await _try_store_embedding(db, paper.id, full_text)

    db.commit()
    db.refresh(paper)

    return PaperIngestResponse(
        paper_id=paper.id,
        sha256=sha256,
        was_duplicate=False,
        report=report,
    )


async def _try_store_embedding(db: Session, paper_id: int, full_text: str) -> None:
    if not USE_SEMANTIC:
        return
    try:
        from app.services.embeddings import get_specter_model
        model = get_specter_model()
        embed_text = full_text[:1000]
        # encode() is CPU-bound; off the event loop. The Session is not
        # thread-safe, so only the encode crosses the thread boundary.
        vec = (await asyncio.to_thread(model.encode, [embed_text], normalize_embeddings=True))[0]
        db.add(Embedding(
            paper_id=paper_id,
            kind="title_abstract",
            vector=vec.astype(EMBEDDING_DTYPE).tobytes(),
        ))
    except Exception as e:
        logger.warning(f"Embedding skipped for paper {paper_id}: {e}")
