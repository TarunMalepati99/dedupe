# dedupe-lab: deciding whether two crawled pages are the same, with no ground truth

A small, honest prototype of one decision from a search-index pipeline: collapse duplicate pages without collapsing pages a reader would want separately.

**Idea.** On a template-heavy or thin page, most words are site furniture, so whole-page similarity says "duplicate" for two unrelated pages on one host and "different" for the same article on two hosts. Fix: learn each host's template from the unlabeled crawl (lines that appear on a large share of that host's pages), strip it, then compare main content with Jaccard OR containment-with-length-guard. MinHash-LSH generates candidates so it does not need all-pairs comparison.

**Benchmark (constructed, real text).** 94 documents: Python standard-library docs (pydoc) and Debian licence texts, wrapped in 3 synthetic site templates (nav, cookie banner, footer, volatile date/counter lines). 1,014 labelled pairs across 10 kinds, split dev/test by document (no document in both). Labels follow POLICY.md, written before measuring. Both systems get the same treatment: threshold tuned on dev, scored on test.

**Results (held-out test, 507 pairs, see RESULTS.txt).**

| System | Precision | Recall | F1 |
|---|---|---|---|
| Raw-page Jaccard @0.5 (naive default) | 0.728 | 0.775 | 0.751 |
| Raw-page Jaccard, dev-tuned (0.3) | 0.762 | 0.983 | 0.858 |
| Boilerplate-stripped (this repo) | 1.000 | 1.000 | 1.000 |

The gain is concentrated in thin pages on a shared template: the dev-tuned baseline gets 15/86 of those distinct pairs right, the stripped version 86/86. LSH (64 hashes, 16 bands) kept 223/231 (96.5%) of labelled duplicate test pairs while examining 1.1% of all page pairs.

**Read this before trusting the numbers.**
- The benchmark is small, synthetic in its templates, and separable; a perfect score here says the failure mode is fixed on this data, not that the method scores 1.000 on the web. It is a proof of mechanism, not a production claim.
- Not tested: paraphrases, translations, pages differing in one field (price/version), real crawl noise, and real template drift. Digit-collapsing in line normalisation means a content line that differs only by a number is treated as template if it repeats across a host (a known failure mode).
- Template learning assumes the host is known and the host has enough pages; the boilerplate model was fit on all pages (unlabeled, including test pages), which uses no labels but is transductive.
- LSH missed 8 of 231 duplicate pairs at these settings; tuning bands/rows trades recall against candidate count.

**Run.** `python3 build_benchmark.py && python3 evaluate.py && python3 lsh_check.py && python3 tests.py` (Python 3.11, standard library only).
