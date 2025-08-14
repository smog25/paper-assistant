# run_instructions.md
# Research Assistant - Setup & Run Instructions

## System Dependencies (One-time install)

### macOS
brew install tesseract poppler

### Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils

### Windows
1. Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
2. Download Poppler: http://blog.alivate.com.au/poppler-windows/
3. Add both to PATH
4. Set POPPLER_PATH in .env file to your Poppler bin directory

## Python Setup

1. Use Python 3.11 (recommended):
python3 --version  # should show 3.11.x

2. Create virtual environment:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Create .env file from example:
cp .env.example .env
# Edit .env and add:
# - CROSSREF_EMAIL (REQUIRED)
# - OPENAI_API_KEY (optional, for AI summaries)
# - POPPLER_PATH (Windows only)

## Running the Application

### Terminal A - Start Backend:
uvicorn backend:app --reload --port 8000
# You should see: "Started ResearchAssistant backend"

### Terminal B - Start Frontend:
streamlit run app_v2.py
# Opens automatically at http://localhost:8501

## API Testing (Optional)

### Health Check:
curl http://localhost:8000/healthz

### Version Info:
curl http://localhost:8000/version

### Parse PDF:
curl -X POST -F "file=@paper.pdf" http://localhost:8000/api/parse_pdf

### Verify References:
curl -X POST http://localhost:8000/api/verify_references \
  -H "Content-Type: application/json" \
  -d '{"text": "Your paper text here..."}'

### Extract Statistics:
curl -X POST http://localhost:8000/api/extract_statistics \
  -H "Content-Type: application/json" \
  -d '{"text": "Your paper text with p-values..."}'

## Troubleshooting

- **Backend not connecting**: Ensure port 8000 is free
- **OCR not working**: Check tesseract: `tesseract --version`
- **Windows OCR issues**: Verify POPPLER_PATH in .env points to poppler/bin
- **Slow OCR**: Normal for large PDFs, limited to 20 pages by default
- **CrossRef rate limits**: Add valid email to .env
- **Different API port**: Set API_BASE in .env