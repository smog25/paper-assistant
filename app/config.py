import os
import logging
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MAX_PDF_SIZE = int(os.getenv("MAX_PDF_SIZE_MB", 10)) * 1024 * 1024
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 100))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", 20))
CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", "test@example.com")
CROSSREF_CONCURRENT = int(os.getenv("CROSSREF_CONCURRENT_REQUESTS", 3))
# One contact email by default; independently overridable for OpenAlex's polite pool
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", CROSSREF_EMAIL)
OPENALEX_CONCURRENT = int(os.getenv("OPENALEX_CONCURRENT_REQUESTS", 3))
POPPLER_PATH = os.getenv("POPPLER_PATH")
# Default to the known local frontends (React dev, alt dev port, Streamlit)
# rather than "*": a wildcard also forces credentials off in main.py, since
# browsers reject Access-Control-Allow-Origin:* on credentialed responses.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:8501",
    ).split(",")
    if o.strip()
]
USE_SEMANTIC = os.getenv("USE_SEMANTIC_MATCHING", "1") == "1"

ANALYZER_VERSION = "1.0.0"

import numpy as np
EMBEDDING_DTYPE = np.float32

# Prevent PIL decompression bombs from scanned PDFs
Image.MAX_IMAGE_PIXELS = 30_000_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
