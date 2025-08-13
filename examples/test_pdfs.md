# examples/test_pdfs.md
# Test PDFs for Research Assistant

## Computer Science Papers
- [Attention Is All You Need (Transformer paper)](https://arxiv.org/pdf/1706.03762.pdf)
  - Good for: Technical citations, complex formatting
  - Expected: Mix of verified and suspicious citations

## Biology Papers
- [Any recent bioRxiv paper](https://www.biorxiv.org/)
  - Good for: P-values, sample sizes, statistical tests
  - Expected: Multiple red flags if not pre-registered

## Psychology Papers
- [Papers from PsyArXiv](https://psyarxiv.com/)
  - Good for: Small sample sizes, p-value clustering
  - Expected: Varies widely in truthiness score

## Scanned PDF Test
- [Internet Archive Books](https://archive.org/)
  - Download any old book/paper PDF
  - Tests OCR functionality
  - Expected: "Run OCR" button appears

## Quick Test Workflow

1. **Upload PDF** → Check extraction method (native vs needs OCR)
2. **Verify Citations** → Should show verified/suspicious/not found
3. **Extract Statistics** → Should find p-values and sample sizes
4. **Truthiness Score** → Should give grade with reasons
5. **In-Text Citations** → Should list numeric and author-year

## Expected Results

### Good Paper Signs:
- ✅ Pre-registered study
- ✅ Data availability statement
- ✅ Power analysis conducted
- ✅ Discusses limitations

### Red Flag Signs:
- 🚩 Multiple p-values near 0.05
- ⚠️ Small sample sizes (n<30)
- ❌ No conflict of interest statement
- ❌ No limitations section