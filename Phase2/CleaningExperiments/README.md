# Query Cleaning Experiments

Notebook `notebooks/04_query_cleaning_c0_c1_c2.ipynb` compares English queries under the same provisional MiniLM encoder, document embeddings, exact FAISS index, top-100 retrieval, development QIDs, qrels, and metric code.

- **C0:** original question, with only encoder-required formatting.
- **C1:** C0 plus documented Unicode, whitespace, and clearly present markup normalization.
- **C2:** C1 plus only user-reviewed, defensible typo/entity-alias entries. The proposal inventory must be shown to the user before any correction is applied. It may remain blocked pending the user's choice if no valid corrections are found.

Only original development questions may be inspected for C2 candidates. Never use answers, reference snippets, qrels, or test questions to make corrections. QID 181 is excluded. Do not game-filter or rerank; reuse the same document vectors and index across all conditions. Report protocol metrics with Grade 1 and Grade 2 preserved as distinct grades, plus per-game and per-task breakdowns and paired query differences.

Versioned cleaned-query files, ranked runs, per-query metrics, experiment manifests, and reports belong in `outputs/` and `reports/`. Results are provisional development results, not test results and not proof that cleaning helps.

## Current run status

The full question-only review covered all 188 effective development questions using their original question text. It found no clearly defensible typo or entity-alias correction; `outputs/c2_correction_inventory_proposed_v1.csv` therefore contains only the column headers. C0 and C1 have been executed and saved. Since C1 made no changes on this dataset, the C0 and C1 rankings and metrics are identical. C2 has not been treated as a scored condition or used to apply transformations; obtain the user's direction on whether to define C2 as an explicit no-op or provide a correction inventory before continuing.
