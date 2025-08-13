# backend.py
"""
FastAPI backend for Research Assistant
Production-ready with async operations, caching, and proper error handling
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from datetime import datetime
import re
import httpx
import asyncio
import hashlib
import pypdf
import io
import random
import logging
import os
import sys
from cachetools import TTLCache
import pytesseract
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
from PIL import Image
import importlib.metadata

# Load environment
load_dotenv()

# Configuration from environment
MAX_PDF_SIZE = int(os.getenv("MAX_PDF_SIZE_MB", 10)) * 1024 * 1024
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 100))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", 20))
CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", "test@example.com")
CROSSREF_CONCURRENT = int(os.getenv("CROSSREF_CONCURRENT_REQUESTS", 3))
POPPLER_PATH = os.getenv("POPPLER_PATH")  # For Windows
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Security for OCR
Image.MAX_IMAGE_PIXELS = 30_000_000  # Prevent decompression bombs

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global semaphores
CROSSREF_SEMAPHORE = asyncio.Semaphore(CROSSREF_CONCURRENT)
OCR_SEMAPHORE = asyncio.Semaphore(2)  # Limit concurrent OCR operations

# TTL cache for reference verification (24 hours)
reference_cache = TTLCache(maxsize=5000, ttl=86400)

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    # Startup
    app.state.http = httpx.AsyncClient(
        timeout=20.0,
        headers={
            "User-Agent": f"ResearchAssistant/1.0 (mailto:{CROSSREF_EMAIL})"
        }
    )
    logger.info("Started ResearchAssistant backend")
    yield
    # Shutdown
    await app.state.http.aclose()
    logger.info("Shutdown ResearchAssistant backend")

# Initialize FastAPI
app = FastAPI(
    title="Research Assistant API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
origins = ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PYDANTIC MODELS ---

class TextBody(BaseModel):
    """Request body for text-based endpoints"""
    text: str = Field(..., min_length=1, max_length=5000000)

class PDFResponse(BaseModel):
    """Response for PDF parsing"""
    text: str
    pages: int
    sections: Dict[str, int]
    word_count: int
    extraction_method: str

class Citation(BaseModel):
    """Verified citation result"""
    raw_text: str
    normalized: str
    status: str
    confidence: float = Field(ge=0, le=1)
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    explanation: str

class InTextCitations(BaseModel):
    """In-text citation results"""
    numeric: List[str]
    author_year: List[str]
    total_count: int
    explanation: str

class StatisticsResult(BaseModel):
    """Statistical analysis results"""
    p_values: List[str]
    sample_sizes: List[str]
    effect_sizes: List[str]
    red_flags: List[str]
    summary: str

class TruthinessResult(BaseModel):
    """Paper trustworthiness analysis"""
    score: int = Field(ge=0, le=100)
    reasons: List[str]
    grade: str
    disclaimer: str = "This is a heuristic analysis, not peer review"

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str

class VersionResponse(BaseModel):
    """Version information"""
    app_version: str
    python_version: str
    dependencies: Dict[str, str]

# --- HELPER FUNCTIONS ---

async def sleep_with_backoff(attempt: int, max_delay: float = 5.0):
    """Exponential backoff with jitter"""
    delay = min(1.0 * (2 ** attempt) + random.random() * 0.5, max_delay)
    await asyncio.sleep(delay)

def get_cache_key(text: str) -> str:
    """Generate consistent cache key"""
    return hashlib.md5(text.encode()).hexdigest()

async def crossref_query(http: httpx.AsyncClient, query: str, use_title: bool = False) -> Optional[dict]:
    """Query Crossref with retries and proper parameters"""
    url = "https://api.crossref.org/works"
    
    params = {
        "query.bibliographic" if not use_title else "query.title": query,
        "rows": 1,
        "mailto": CROSSREF_EMAIL
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

def detect_sections(text: str) -> Dict[str, int]:
    """Identify major sections in paper"""
    sections = {}
    patterns = {
        'Abstract': r'\b(?:Abstract|ABSTRACT|Summary)\b',
        'Introduction': r'\b(?:Introduction|INTRODUCTION|Background)\b',
        'Methods': r'\b(?:Methods?|METHODS?|Methodology|Materials and Methods)\b',
        'Results': r'\b(?:Results?|RESULTS?|Findings)\b',
        'Discussion': r'\b(?:Discussion|DISCUSSION|Conclusions?)\b',
        'References': r'\b(?:References|REFERENCES|Bibliography|Works Cited)\b'
    }
    
    for name, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            sections[name] = match.start()
    
    return dict(sorted(sections.items(), key=lambda x: x[1]))

def extract_references_section(text: str) -> str:
    """Extract references section from paper"""
    # Improved regex: line-anchored, case-insensitive
    pattern = r'(?mi)^\s*(references|bibliography|works cited|literature cited)\s*:?\s*$'
    match = re.search(pattern, text)
    if match:
        return text[match.start():]
    return ""

def parse_reference_list(ref_text: str) -> List[str]:
    """Parse individual references from section"""
    if not ref_text:
        return []
    
    references = []
    
    # Try numbered format first
    numbered = re.split(r'\n\s*\[?\d+\]?[\.\)]\s+', ref_text)
    if len(numbered) > 5:
        references = [r.strip() for r in numbered if 20 < len(r.strip()) < 500]
    else:
        # Fallback to line-based parsing
        lines = ref_text.split('\n')
        current = ""
        for line in lines:
            line = line.strip()
            if re.match(r'^[A-Z]', line) and current and len(current) > 20:
                references.append(current)
                current = line
            elif line:
                current += " " + line if current else line
        if current and len(current) > 20:
            references.append(current)
    
    return references[:50]

def get_year_from_item(item: dict) -> Optional[str]:
    """Extract year from Crossref item with fallbacks"""
    date_fields = ['issued', 'published-online', 'published-print']
    for field in date_fields:
        if item.get(field) and item[field].get('date-parts'):
            year = item[field]['date-parts'][0][0]
            if year:
                return str(year)
    return None

# --- ENDPOINTS ---

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/version", response_model=VersionResponse)
async def version_info():
    """Version information endpoint"""
    deps = {}
    
    # Get actual installed versions
    packages = ['fastapi', 'httpx', 'pypdf', 'streamlit']
    for pkg in packages:
        try:
            deps[pkg] = importlib.metadata.version(pkg)
        except:
            deps[pkg] = "unknown"
    
    return VersionResponse(
        app_version="1.0.0",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        dependencies=deps
    )

@app.post("/api/parse_pdf", response_model=PDFResponse)
async def parse_pdf(file: UploadFile = File(...)):
    """Parse PDF with native text extraction"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    contents = await file.read()
    
    if not contents.startswith(b"%PDF"):
        raise HTTPException(400, "Invalid PDF file")
    
    if len(contents) > MAX_PDF_SIZE:
        raise HTTPException(400, f"PDF too large. Max: {MAX_PDF_SIZE/1024/1024:.1f}MB")
    
    try:
        pdf_file = io.BytesIO(contents)
        pdf_reader = pypdf.PdfReader(pdf_file)
        
        if len(pdf_reader.pages) > MAX_PDF_PAGES:
            raise HTTPException(400, f"Too many pages. Max: {MAX_PDF_PAGES}")
        
        full_text = ""
        for i, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            except Exception as e:
                logger.warning(f"Failed to extract page {i+1}: {e}")
                continue
        
        if len(full_text) < 100:
            return PDFResponse(
                text="",
                pages=len(pdf_reader.pages),
                sections={},
                word_count=0,
                extraction_method="failed_needs_ocr"
            )
        
        sections = detect_sections(full_text)
        
        return PDFResponse(
            text=full_text,
            pages=len(pdf_reader.pages),
            sections=sections,
            word_count=len(full_text.split()),
            extraction_method="native"
        )
        
    except Exception as e:
        logger.error(f"PDF parsing error: {e}")
        raise HTTPException(500, "Failed to process PDF")

@app.post("/api/parse_pdf_ocr", response_model=PDFResponse)
async def parse_pdf_ocr(file: UploadFile = File(...)):
    """Parse PDF using OCR (for scanned documents)"""
    
    contents = await file.read()
    
    if not contents.startswith(b"%PDF"):
        raise HTTPException(400, "Invalid PDF file")
    
    if len(contents) > MAX_PDF_SIZE:
        raise HTTPException(400, f"PDF too large for OCR")
    
    async with OCR_SEMAPHORE:  # Limit concurrent OCR
        try:
            # Convert to images with Windows support
            kwargs = {"dpi": 150, "thread_count": 2}
            if POPPLER_PATH:
                kwargs["poppler_path"] = POPPLER_PATH
            
            images = convert_from_bytes(contents, **kwargs)
            
            # Limit pages for OCR
            if len(images) > MAX_OCR_PAGES:
                images = images[:MAX_OCR_PAGES]
                logger.warning(f"Truncated OCR to {MAX_OCR_PAGES} pages")
            
            full_text = ""
            loop = asyncio.get_running_loop()
            
            for i, image in enumerate(images):
                page_text = await loop.run_in_executor(
                    None, 
                    pytesseract.image_to_string, 
                    image
                )
                full_text += f"\n--- Page {i+1} ---\n{page_text}"
            
            if len(full_text) < 100:
                raise HTTPException(400, "OCR failed to extract text")
            
            sections = detect_sections(full_text)
            
            return PDFResponse(
                text=full_text,
                pages=len(images),
                sections=sections,
                word_count=len(full_text.split()),
                extraction_method="ocr"
            )
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            raise HTTPException(500, "OCR processing failed")

@app.post("/api/verify_references", response_model=List[Citation])
async def verify_references(request: Request, body: TextBody):
    """Verify references using Crossref"""
    
    ref_section = extract_references_section(body.text)
    if not ref_section:
        return []
    
    ref_list = parse_reference_list(ref_section)
    if not ref_list:
        return []
    
    async def verify_one(reference: str) -> Citation:
        cache_key = get_cache_key(reference)
        if cache_key in reference_cache:
            return reference_cache[cache_key]
        
        async with CROSSREF_SEMAPHORE:
            clean_ref = re.sub(r'^\[?\d+\]?[\.\)]\s*', '', reference)[:200]
            
            # Extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', reference)
            
            # Better author extraction: look near commas or "and"
            author_pattern = r'([A-Z][a-z]+)(?:\s*,|\s+and\s+|\s+&\s+)'
            author_matches = re.findall(author_pattern, reference)[:5]
            
            data = await crossref_query(request.app.state.http, clean_ref)
            
            if data and data.get('message', {}).get('items'):
                item = data['message']['items'][0]
                score = item.get('score', 0)
                
                if 40 <= score <= 70:
                    title_match = re.match(r'^[^.]+', clean_ref)
                    if title_match:
                        title_data = await crossref_query(
                            request.app.state.http, 
                            title_match.group(0), 
                            use_title=True
                        )
                        if title_data and title_data.get('message', {}).get('items'):
                            title_item = title_data['message']['items'][0]
                            if title_item.get('score', 0) > score:
                                item = title_item
                                score = title_item.get('score', 0)
                
                confidence_boost = 0
                explanation_parts = []
                
                # Check year with fallbacks
                if year_match:
                    item_year = get_year_from_item(item)
                    if item_year and item_year == year_match.group(0):
                        confidence_boost += 10
                        explanation_parts.append("year match")
                
                # Check authors
                if author_matches and item.get('author'):
                    crossref_authors = {a.get('family', '').lower() for a in item['author']}
                    ref_authors = {a.lower() for a in author_matches}
                    if crossref_authors & ref_authors:
                        confidence_boost += 10
                        explanation_parts.append("author match")
                
                # Check DOI
                doi_in_ref = re.search(r'10\.\d{4,}/[-._;()/:\w]+', reference)
                if doi_in_ref and item.get('DOI'):
                    if doi_in_ref.group(0).lower() == item['DOI'].lower():
                        confidence_boost += 30
                        explanation_parts.append("DOI match")
                
                adjusted_score = min(score + confidence_boost, 100)
                
                explanation = f"Crossref score: {score:.0f}"
                if explanation_parts:
                    explanation += f" (boosted: {', '.join(explanation_parts)})"
                
                if adjusted_score > 70:
                    result = Citation(
                        raw_text=reference[:100],
                        normalized=item.get('title', ['Unknown'])[0][:100],
                        status="verified",
                        confidence=adjusted_score / 100,
                        doi=item.get('DOI'),
                        title=item.get('title', [''])[0],
                        authors=[a.get('family', '') for a in item.get('author', [])[:3]],
                        explanation=explanation
                    )
                elif adjusted_score > 40:
                    result = Citation(
                        raw_text=reference[:100],
                        normalized=clean_ref[:100],
                        status="suspicious",
                        confidence=adjusted_score / 100,
                        doi=item.get('DOI') if adjusted_score > 50 else None,
                        title=item.get('title', [''])[0] if adjusted_score > 50 else None,
                        authors=None,
                        explanation=f"Low confidence: {explanation}"
                    )
                else:
                    result = Citation(
                        raw_text=reference[:100],
                        normalized=clean_ref[:100],
                        status="not_found",
                        confidence=0.0,
                        doi=None,
                        title=None,
                        authors=None,
                        explanation="No reliable match found"
                    )
            else:
                result = Citation(
                    raw_text=reference[:100],
                    normalized=clean_ref[:100],
                    status="not_found",
                    confidence=0.0,
                    doi=None,
                    title=None,
                    authors=None,
                    explanation="Could not verify with Crossref"
                )
            
            reference_cache[cache_key] = result
            return result
    
    results = await asyncio.gather(
        *[verify_one(ref) for ref in ref_list[:20]],
        return_exceptions=True
    )
    
    # Log exceptions and filter
    valid_results = []
    for r in results:
        if isinstance(r, Citation):
            valid_results.append(r)
        elif isinstance(r, Exception):
            logger.exception("verify_references worker failed", exc_info=r)
    
    return valid_results

@app.post("/api/find_intext_citations", response_model=InTextCitations)
async def find_intext_citations(body: TextBody):
    """Extract in-text citation markers"""
    
    text = body.text
    
    numeric = []
    numeric_patterns = [
        r'\[(\d+)\]',
        r'\[(\d+[-–]\d+)\]',
        r'\[(\d+(?:,\s*\d+)+)\]',
    ]
    
    for pattern in numeric_patterns:
        matches = re.findall(pattern, text)
        numeric.extend(matches)
    
    author_year = []
    author_patterns = [
        r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s+(\d{4})\)',
        r'\(([A-Z][a-z]+\s+(?:&|and)\s+[A-Z][a-z]+),?\s+(\d{4})\)',
    ]
    
    for pattern in author_patterns:
        matches = re.findall(pattern, text)
        author_year.extend([f"{m[0]}, {m[1]}" for m in matches])
    
    numeric = list(set(numeric))[:100]
    author_year = list(set(author_year))[:100]
    
    total = len(numeric) + len(author_year)
    explanation = f"Found {len(numeric)} numeric and {len(author_year)} author-year citations"
    
    return InTextCitations(
        numeric=numeric,
        author_year=author_year,
        total_count=total,
        explanation=explanation
    )

@app.post("/api/extract_statistics", response_model=StatisticsResult)
async def extract_statistics(body: TextBody):
    """Extract and analyze statistical claims"""
    
    text = body.text
    
    # Normalize whitespace for better matching
    normalized = re.sub(r'\s+', ' ', text)
    
    # Extract statistics with better patterns
    p_values = re.findall(r'[pP]\s*[=<>≤≥]\s*0?\.\d+', normalized)
    p_values.extend(re.findall(r'[pP]\s*<\s*\.0\d+', normalized))  # p<.001 format
    p_values = p_values[:30]
    
    sample_sizes = re.findall(r'[nN]\s*=\s*\d+', normalized)[:30]
    
    effect_sizes = []
    effect_sizes.extend(re.findall(r"Cohen's\s*d\s*=\s*-?\d*\.\d+", normalized))
    effect_sizes.extend(re.findall(r'\br\s*=\s*-?\d*\.\d+', normalized))
    effect_sizes.extend(re.findall(r'η²\s*=\s*\d*\.\d+', normalized))
    effect_sizes = effect_sizes[:15]
    
    red_flags = []
    
    p_vals_near_05 = [p for p in p_values if any(x in p for x in ['0.04', '0.05', '0.06'])]
    if len(p_vals_near_05) > 3:
        red_flags.append(f"🚩 {len(p_vals_near_05)} p-values suspiciously close to 0.05")
    
    small_samples = []
    for size_str in sample_sizes[:10]:
        match = re.findall(r'\d+', size_str)
        if match:
            n = int(match[0])
            if n < 30:
                small_samples.append(size_str)
    
    if small_samples:
        red_flags.append(f"⚠️ Small sample sizes: {', '.join(small_samples[:3])}")
    
    if not re.search(r'\blimitation', text, re.IGNORECASE):
        red_flags.append("❌ No limitations section found")
    
    if not re.search(r'conflict.*interest|disclosure', text, re.IGNORECASE):
        red_flags.append("❌ No conflict of interest statement")
    
    if not re.search(r'ethic.*approv|IRB|institutional.*review', text, re.IGNORECASE):
        red_flags.append("⚠️ No ethics approval mentioned")
    
    if re.search(r'pre-?register', text, re.IGNORECASE):
        red_flags.append("✅ Study was pre-registered")
    
    if re.search(r'power\s+analysis', text, re.IGNORECASE):
        red_flags.append("✅ Power analysis conducted")
    
    if re.search(r'data.*available|github\.com|osf\.io|figshare', text, re.IGNORECASE):
        red_flags.append("✅ Data/code availability statement")
    
    if re.search(r'replicat', text, re.IGNORECASE):
        red_flags.append("✅ Discusses replication")
    
    summary = f"Found {len(p_values)} p-values, {len(sample_sizes)} sample sizes, {len(effect_sizes)} effect sizes"
    
    return StatisticsResult(
        p_values=p_values,
        sample_sizes=sample_sizes,
        effect_sizes=effect_sizes,
        red_flags=red_flags,
        summary=summary
    )

@app.post("/api/truthiness_score", response_model=TruthinessResult)
async def calculate_truthiness(body: TextBody, field: Optional[str] = Query(None)):
    """Calculate paper trustworthiness score"""
    
    text = body.text
    score = 100
    reasons = []
    
    stats_body = TextBody(text=text)
    stats = await extract_statistics(stats_body)
    
    flags = stats.red_flags
    
    has_no_limits = any("No limitations" in f for f in flags)
    has_no_coi = any("conflict of interest" in f.lower() for f in flags)
    has_no_ethics = any("No ethics" in f for f in flags)
    is_preregistered = any("pre-registered" in f.lower() for f in flags)
    has_power_analysis = any("Power analysis" in f for f in flags)
    has_open_data = any(
        ("data" in f.lower() and "available" in f.lower()) or 
        "github.com" in f.lower() or 
        "osf.io" in f.lower() 
        for f in flags
    )
    discusses_replication = any("replication" in f.lower() for f in flags)
    
    field_multipliers = {
        "psychology": 1.0,
        "clinical": 1.2,
        "cs": 0.8,
        "biology": 1.1
    }
    multiplier = field_multipliers.get(field, 1.0) if field else 1.0
    
    suspicious_p = [p for p in stats.p_values if any(x in p for x in ['0.04', '0.05', '0.06'])]
    if len(suspicious_p) > 2:
        penalty = int(15 * multiplier)
        score -= penalty
        reasons.append(f"Multiple p-values near threshold ({len(suspicious_p)} found)")
    
    small_n_count = 0
    for size_str in stats.sample_sizes[:10]:
        match = re.findall(r'\d+', size_str)
        if match:
            n = int(match[0])
            threshold = int(30 * multiplier)
            if n < threshold:
                small_n_count += 1
    
    if small_n_count > 0:
        score -= int(10 * multiplier)
        reasons.append(f"Small sample sizes detected ({small_n_count} instances)")
    
    if has_no_limits:
        score -= 20
        reasons.append("No limitations discussed (major concern)")
    
    if has_no_coi:
        score -= 10
        reasons.append("No conflict of interest statement")
    
    if has_no_ethics:
        score -= 5
        reasons.append("No ethics approval mentioned")
    
    if is_preregistered:
        score += 10
        reasons.append("✅ Pre-registered study")
    
    if has_power_analysis:
        score += 5
        reasons.append("✅ Power analysis conducted")
    
    if has_open_data:
        score += 10
        reasons.append("✅ Open data/code available")
    
    if discusses_replication:
        score += 5
        reasons.append("✅ Discusses replication")
    
    score = max(0, min(100, score))
    
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 55:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"
    
    return TruthinessResult(
        score=score,
        reasons=reasons,
        grade=grade,
        disclaimer="This is a heuristic analysis, not peer review. Use as a preliminary signal only."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)