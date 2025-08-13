# cleanup.md
# Cleanup Instructions

## Old Files to Archive

1. Rename your old app:
mv app.py app_legacy.py

2. Use new frontend:
- app_v2.py is now your main Streamlit app
- All logic has moved to backend.py

## What Changed

### Obsolete in app_legacy.py:
- Direct OpenAI calls → Now optional in app_v2.py
- PDF processing logic → Moved to backend.py
- Citation verification → Now via CrossRef API in backend
- Statistics extraction → Proper endpoint in backend
- All business logic → Separated into API

### New Architecture:
User → Streamlit (app_v2.py) → FastAPI (backend.py) → External APIs
                                       ↓
                              Can swap to Next.js later

### Benefits:
- Frontend agnostic (can swap to Next.js/React)
- Proper caching and rate limiting
- Better error handling
- OCR support with fallback
- Free core features (no AI costs)
- Production-ready patterns

## Migration Complete!
You're now using an API-first architecture ready for scale.

## Next Steps:
1. Test all features with a real PDF
2. Verify citations work without OpenAI
3. Try OCR with a scanned document
4. Check cost tracking in sidebar