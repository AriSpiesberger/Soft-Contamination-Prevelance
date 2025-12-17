# Embedding Analysis Findings

**Date:** 2025-12-17
**Analysis of:** my_run contamination analysis results

---

## EMBEDDING COVERAGE ISSUE

### What We Found:
- **Total embeddings saved:** 1.84 TB (~310M rows across 944 files)
- **Major issue:** Incomplete coverage across all data sources

### Coverage by Source:

| Source | Actual Size | Expected Size | Coverage |
|--------|-------------|---------------|----------|
| common_crawl | 1714 GB | 3150 GB | **54%** |
| stack_edu | 86 GB | 526 GB | **16%** ⚠️ |
| finemath | 44 GB | 96 GB | **46%** |
| olmocr_science_pdfs | 43 GB | 84 GB | **51%** |
| dolma1_7 | 1.1 GB | 4.2 GB | **26%** |
| rpj | **0 GB** | 9.1 GB | **MISSING** ❌ |

### Key Issues:
1. **stack_edu**: Only 16% coverage - severely underrepresented
2. **rpj**: Completely missing - no embeddings saved at all
3. **common_crawl**: Only half of expected data
4. Most likely cause: **Pipeline was interrupted or failed partway through**

---

## CONTAMINATION ANALYSIS RESULTS

### Completed Benchmarks:

| Benchmark | Mode | Items | Status |
|-----------|------|-------|--------|
| MuSR | input | 756 | ✅ Complete |
| MuSR | output | 756 | ✅ Complete |
| MuSR | input_output | 756 | ✅ Complete |
| MBPP | input | 257 | ✅ Complete |
| MBPP | output | 257 | ✅ Complete |
| MBPP | input_output | 257 | Partial (files missing) |

### Analysis Scope:
- **Corpus files analyzed:** 944 parquet files
- **Total corpus size:** 310M embedded paragraphs/documents
- **Top-K matches saved:** 100 per benchmark item

### Data Stored:
1. **Checkpoint files** (`checkpoints_fast/`):
   - `*_sims.pt`: Top 100 similarity scores per benchmark item
   - `*_idxs.pt`: Global indices of top 100 matches
   - `*_state.pkl`: Mapping of global indices → parquet files

2. **Results files** (`results_fast/`):
   - `*_matches.json`: Resolved matches with file paths and local indices
   - `*_sims.npy`: NumPy arrays of similarity scores
   - `*_dist.png`: Distribution plots

---

## CSV EXTRACTION STATUS

### In Progress:
Creating CSV files with:
- Benchmark ID and full text
- Rank (1-100)
- Similarity score
- Corpus file name
- Corpus text content
- Global and local indices

### Output Location:
`my_run/top_matches_csvs/`

Files being generated:
- `musr_input_top100.csv`
- `musr_output_top100.csv`
- `musr_input_output_top100.csv`
- `mbpp_input_top100.csv`
- `mbpp_output_top100.csv`

---

## QUESTIONS & ANSWERS

**Q: Did we save all the embeddings?**
A: No - only ~54% of common_crawl, 16% of stack_edu, and rpj is completely missing.

**Q: Do we have all similarity comparisons or just top 100?**
A: Only top 100 matches per benchmark item. Full pairwise comparisons would be ~300B values (too large to store).

**Q: Can we extract the actual text for matches?**
A: Yes - extraction in progress using checkpoint data to map global indices → parquet files → actual text.

---

## NEXT STEPS

### Immediate:
1. ✅ Extract top 100 matches to CSV with full text (in progress)
2. Review CSV files to assess contamination severity

### To Fix Embedding Coverage:
1. Check production pipeline logs to find where it failed
2. Resume embedding generation for incomplete sources:
   - rpj (completely missing)
   - stack_edu (84% missing)
   - common_crawl (46% missing)
3. Re-run contamination analysis on complete dataset

### Production Pipeline Investigation:
- Check S3 bucket for partial uploads
- Review production.py process logs
- Identify failure point (OOM, network, timeout?)
- Restart from checkpoint if possible
