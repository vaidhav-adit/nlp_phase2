# ChronoPlay-QA Team Handoff v1

This folder is the experiment handoff for four teammates working in parallel. It packages copies of the frozen **ChronoPlay-QA Claude Verified v1.1** corpus and corrected qrels, the controlling research protocol and split manifest, the original project presentation, and the development-only BM25 reference run. The source project files were not modified.

## Authoritative files

- Frozen corpus: `02_FROZEN_DATA/Corpus_4192_Gaming_Documents.csv` (4,192 rows; stable corpus IDs).
- Original query export and QID metadata: `02_FROZEN_DATA/Original_Queries_232_with_Usability_Fields.csv` (232 source rows; preserve `qid` as provided).
- Frozen corrected qrels: `02_FROZEN_DATA/Final_Relevance_Judgments_294_Pairs.csv` (294 positive pairs: 259 Grade 2, 35 Grade 1). This is the authoritative judgment file. The old `ChronoPlay_qrels_FINAL.csv`, `CURRENT_qrels_usable.csv`, and split-specific qrels are not included as substitutes.
- Split and leakage control: `03_SPLITS_AND_EVALUATION/Frozen_Split_and_Leakage_Audit_Manifest.json`.
- Experimental rules and four-phase protocol: `01_READ_FIRST/Research_Protocol_and_Experimental_Rules.md`.
- Reference evidence mapping: `02_FROZEN_DATA/Reference_to_Corpus_Document_Mapping.csv`. It maps source references to corpus documents; it is not a qrels file.
- Baseline output: the three files under `04_BASELINE_RESULTS/`.
- Presentation: `01_READ_FIRST/Project_Overview_and_Four_Phases.pdf`, an exact copy of the original PDF.

No separate dataset export manifest or standalone data dictionary was found among the project files. The older `ChronoPlay_SPLIT_MANIFEST.json` is an original split manifest, not an export manifest, and points at older qrels; it is intentionally omitted to avoid conflicting authority. Column headers in the original query, corpus, mapping, and qrels exports are preserved in their respective copies.

## Shared populations

There are 223 usable source queries. Nine unusable source QIDs are 4, 13, 43, 47, 108, 146, 165, 195, and 198. For experiments, combine the original train and validation assignments and exclude QID 181: effective development is **188 queries and 251 qrel pairs** (223 Grade 2, 28 Grade 1). The **34-query test set remains locked** and has 42 qrel pairs (35 Grade 2, 7 Grade 1). Every effective development and test query has at least one positive qrel. QID 181 is omitted from all experiment inputs and evaluation because it is a near-duplicate of locked test QID 184; its judgment remains in the frozen qrels.

The embedding-similar pairs QID 113/118 and QID 134/135 are both within development. The split manifest is the control file for original assignments, exclusions, duplicate checks, and locked test membership. Apply it by original QID; never regenerate QIDs or infer membership from row order. Use the full frozen qrels as the immutable judgment source, then create development/test evaluation views by filtering on the manifest assignments and excluding QID 181 from development. A split-specific evaluation view is not a replacement or edit of the full qrels.

## First steps for the four teammates

1. Each teammate reads this README, the complete protocol, and the split/evaluation guide before running anything.
2. Verify local copies against the hashes recorded in `HANDOFF_SHA256_MANIFEST.json` and preserve these files read-only.
3. Keep a working directory outside the frozen-data subfolder. Use only effective development queries for experiments, selection, tuning, and fine-tuning.
4. Begin the independent phase work in parallel as defined in the protocol: Documents, Queries, Retrieval, and Synthesis. Phase 1’s BM25 run is a shared reference baseline, not a prerequisite for others to start.
5. Log each configuration, change one main variable per ablation, and save ranked runs, metrics, per-game/task breakdowns, latency, compute cost, limitations, and output paths. Compare controlled development results before integrating components.

## Non-negotiable team rules

1. All four phases run in parallel; nobody waits for Phase 1 to finish before starting independent work.
2. Everyone follows the same frozen corpus, source QIDs, qrels, effective split policy, and metric definitions.
3. Use development data for all experimentation, model selection, tuning, and fine-tuning.
4. Never inspect test performance to select models, prompts, preprocessing, thresholds, or methods.
5. Never feed reference answers, reference snippets, qrels, or `retrieved_docs` into retrieval/ranking inputs.
6. Change one main variable per ablation and log all configurations and results.
7. For Phase 2, label generated Hindi/Romanized Hindi queries accurately; do not claim they represent naturally typed Hinglish.
8. For Phase 4, compare synthesis methods using identical retrieved evidence and a fixed context budget.
9. Record metrics, per-game/task breakdowns, latency, compute cost, limitations, and saved outputs.
10. Integrate components only after independent controlled comparisons. Freeze the chosen system before the final locked-test evaluation.

## Copy verification checklist

- [x] 4,192-document frozen corpus copied and source/copy SHA-256 matched.
- [x] 232-query source export copied with source QIDs and metadata intact; SHA-256 matched.
- [x] Corrected 294-pair qrels copied; grade counts verified as 259 Grade 2 and 35 Grade 1; SHA-256 matched.
- [x] Reference mapping copied; SHA-256 matched.
- [x] Corrected split manifest and full protocol copied; SHA-256 matched.
- [x] Original presentation PDF copied; SHA-256 matched.
- [x] BM25 run, metrics/settings, and group metrics copied; SHA-256 matched.
- [x] BM25 run has 788,096 rows (188 × 4,192); all document IDs exist in the frozen corpus.
- [x] No locked-test run or test performance, source notebook, review queues, temporary scripts, or debug logs included.
