import re
import logging
import asyncio
from typing import List, Optional

import httpx
import numpy as np

from app.config import CROSSREF_EMAIL, USE_SEMANTIC
from app.cache import CROSSREF_SEMAPHORE, reference_cache
from app.schemas.models import Citation
from app.utils.helpers import (
    extract_first_doi,
    get_cache_key,
    sleep_with_backoff,
    extract_title_from_ref,
    title_similarity,
    first_year,
)
from app.services.embeddings import get_specter_model

logger = logging.getLogger(__name__)


async def crossref_query(
    http: httpx.AsyncClient,
    query: str,
    use_title: bool = False,
    rows: int = 1,
) -> Optional[dict]:
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic" if not use_title else "query.title": query,
        "rows": rows,
        "mailto": CROSSREF_EMAIL,
    }
    for attempt in range(3):
        try:
            response = await http.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in (429, 500, 502, 503, 504):
                await sleep_with_backoff(attempt)
                continue
            else:
                logger.warning(f"Crossref returned {response.status_code}")
                break
        except httpx.TimeoutException:
            logger.warning(f"Crossref timeout on attempt {attempt + 1}")
            if attempt < 2:
                await sleep_with_backoff(attempt)
            else:
                break
    return None


def _title_sims(guessed_title: str, cand_titles: List[str]) -> List[float]:
    # SPECTER encode is CPU-bound and synchronous — callers on the async path
    # must invoke this via asyncio.to_thread so it can't stall the event loop.
    # Concurrency is naturally bounded by CROSSREF_SEMAPHORE at the call sites.
    if USE_SEMANTIC:
        model = get_specter_model()
        vecs = model.encode([guessed_title] + cand_titles, normalize_embeddings=True)
        return [float(np.dot(vecs[0], vecs[i + 1])) for i in range(len(cand_titles))]
    return [title_similarity(guessed_title, ct) for ct in cand_titles]


async def _verify_one(http: httpx.AsyncClient, reference: str) -> Citation:
    cache_key = get_cache_key(reference)
    if cache_key in reference_cache:
        return reference_cache[cache_key]

    async with CROSSREF_SEMAPHORE:
        clean_ref = re.sub(r"^\[?\d+\]?[\.\)]\s*", "", reference)[:200]

        year_match = re.search(r"\b(19|20)\d{2}\b", reference)
        author_matches = re.findall(
            r"([A-Z][a-z]+)(?:\s*,|\s+and\s+|\s+&\s+)", reference
        )[:5]

        doi_in_ref = extract_first_doi(reference)
        item = None
        base = 0.0
        title_sim_val = 0.0
        doi_ok = False

        if doi_in_ref:
            work_url = f"https://api.crossref.org/works/{doi_in_ref}"
            for attempt in range(2):
                try:
                    resp = await http.get(work_url, params={"mailto": CROSSREF_EMAIL})
                    if resp.status_code == 200:
                        msg = resp.json().get("message", {})
                        if msg:
                            item = msg
                            # A resolvable DOI alone must not verify the reference:
                            # works/{doi} responses carry no relevance score, and a
                            # typo'd DOI can resolve to an unrelated article. Title
                            # agreement has to carry the rest of the confidence.
                            base = 30.0
                            item_title = msg.get("title", [""])[0] if msg.get("title") else ""
                            guessed_title = extract_title_from_ref(clean_ref)
                            if item_title and guessed_title:
                                title_sim_val = (
                                    await asyncio.to_thread(_title_sims, guessed_title, [item_title])
                                )[0]
                            doi_ok = True
                        break
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        await sleep_with_backoff(attempt)
                        continue
                    else:
                        break
                except httpx.TimeoutException:
                    await sleep_with_backoff(attempt)

        if item is None:
            data = await crossref_query(http, clean_ref, rows=3)
            if data and data.get("message", {}).get("items"):
                guessed_title = extract_title_from_ref(clean_ref)
                candidates = data["message"]["items"]
                cand_titles = [c.get("title", [""])[0] if c.get("title") else "" for c in candidates]
                sims = await asyncio.to_thread(_title_sims, guessed_title, cand_titles)

                best = None
                best_total = -1.0
                best_base = 0.0
                best_sim = 0.0
                for cand, sim in zip(candidates, sims):
                    cand_base = float(cand.get("score", 0.0))
                    total = cand_base + 40.0 * sim
                    if total > best_total:
                        best_total = total
                        best = cand
                        best_base = cand_base
                        best_sim = sim
                if best is not None:
                    item = best
                    base = best_base
                    title_sim_val = best_sim

        if item is not None:
            boost = 0
            year_ok = False
            authors_ok = False
            if year_match:
                item_year = first_year(item)
                if item_year and item_year == year_match.group(0):
                    boost += 10
                    year_ok = True
            if author_matches and item.get("author"):
                crossref_authors = {a.get("family", "").lower() for a in item["author"] if a.get("family")}
                ref_authors = {a.lower() for a in author_matches}
                if crossref_authors & ref_authors:
                    boost += 10
                    authors_ok = True
            if doi_in_ref and item.get("DOI") and doi_in_ref.lower() == item["DOI"].lower():
                boost += 30
                doi_ok = True

            confidence = max(0.0, min(100.0, base + 40.0 * title_sim_val + boost))
            status = "verified" if confidence >= 75.0 else "suspicious" if confidence >= 45.0 else "not_found"
            explanation = (
                f"base:{base:.0f} title_sim:{title_sim_val:.2f} "
                f"year:{'✓' if year_ok else '—'} authors:{'✓' if authors_ok else '—'} doi:{'✓' if doi_ok else '—'}"
            )
            result = Citation(
                raw_text=reference[:200],
                normalized=(item.get("title", ["Unknown"])[0][:200] if item.get("title") else clean_ref[:200]),
                status=status,
                confidence=confidence / 100.0,
                doi=item.get("DOI"),
                title=item.get("title", [""])[0] if item.get("title") else None,
                authors=[a.get("family", "") for a in item.get("author", [])[:5]] if item.get("author") else None,
                explanation=explanation,
            )
        else:
            result = Citation(
                raw_text=reference[:200],
                normalized=clean_ref[:200],
                status="not_found",
                confidence=0.0,
                doi=None,
                title=None,
                authors=None,
                explanation="base:0 title_sim:0.00 year:— authors:— doi:—",
            )

        reference_cache[cache_key] = result
        return result


async def verify_references(http: httpx.AsyncClient, text: str) -> List[Citation]:
    from app.utils.helpers import parse_reference_list
    from app.services.pdf import extract_references_section

    ref_section = extract_references_section(text)
    if not ref_section:
        return []

    ref_list = parse_reference_list(ref_section)
    if not ref_list:
        return []

    results = await asyncio.gather(
        *[_verify_one(http, ref) for ref in ref_list[:20]],
        return_exceptions=True,
    )

    valid: List[Citation] = []
    for r in results:
        if isinstance(r, Citation):
            valid.append(r)
        elif isinstance(r, Exception):
            logger.exception("verify_references worker failed", exc_info=r)
    return valid
