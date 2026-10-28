# Development/Test Split and Evaluation Guide

## Frozen data and memberships

The frozen qrels contain **294 positive pairs: 259 Grade 2 and 35 Grade 1**. Grade 2 is relevant; Grade 1 is partially relevant. Keep these labels unchanged.

- **Effective development:** 188 queries, 251 pairs (223 Grade 2, 28 Grade 1). Start with original train plus validation membership, then remove QID 181 from all experiment inputs and evaluation.
- **Locked test:** 34 queries, 42 pairs (35 Grade 2, 7 Grade 1). Keep all 34 assignments fixed; do not use test outcomes for any development decision.
- **Unusable source QIDs:** 4, 13, 43, 47, 108, 146, 165, 195, 198.
- **Quarantine:** QID 181 is a near-duplicate of locked test QID 184. Its qrel remains in the full frozen qrels, but QID 181 is excluded from every experiment input/evaluation view.
- **Embedding check:** QID 113/118 (0.954491) and QID 134/135 (0.915173) are both wholly within development.
- Every effective development and test query has at least one positive qrel.

## Applying the split

Use `Frozen_Split_and_Leakage_Audit_Manifest.json` as the assignment and exclusion control. Match original source QIDs exactly; do not assign by row position, regenerate IDs, or change labels. Effective development is the union of original train and validation QIDs minus QID 181. Test is the original 34 test QIDs. Keep the test membership locked.

`Final_Relevance_Judgments_294_Pairs.csv` is the full authoritative frozen qrels. Derive split-specific evaluation views by filtering that file on QID membership. These views do not replace, relabel, or modify the frozen qrels; QID 181's pair is retained in the full file but not scored. Verify positive coverage after filtering.

## Evaluation rules

Use nDCG@10 as the primary graded metric with gains `2^grade - 1` (Grade 2 gain 3, Grade 1 gain 1). Do not treat Grade 1 as fully relevant. Also report Grade 2-only MRR@10 and Recall@5/10, counts of Grade 1 and Grade 2 retrieved separately, and any-positive Hit@5/10 with its binary definition stated. Macro-average by query; include per-game, task-type, and query-form breakdowns. Report paired query-level comparisons and confidence intervals. Tune only on development.

The corpus is shared across query partitions as expected in retrieval. Shared relevant documents/snippets are a limitation to disclose, not automatic query leakage. Never provide answers, reference snippets, qrels, or `retrieved_docs` to retrieval/ranking models.
