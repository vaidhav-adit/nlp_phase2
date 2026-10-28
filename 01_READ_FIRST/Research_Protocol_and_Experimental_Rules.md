# ChronoPlay-QA Research Protocol v1

**Status: READY TO EXPERIMENT.** The embedding duplicate gate passed. This protocol defines the four-phase plan and the first development-only baseline.

## Dataset and evaluation population

Use the frozen **ChronoPlay-QA Claude Verified v1.1** corpus and judgments. The corpus has 4,192 English documents. The source dataset has 232 query-answer pairs, of which 223 are usable. Excluded source QIDs are 4, 13, 43, 47, 108, 146, 165, 195, and 198. Query IDs refer to the pinned source revision `leoner24/ChronoPlay-QA@de418597b67fb8a671e5a85219c7c5ef9d9f7406`; do not remap them by row order in another revision.

The frozen qrels contain 294 positive query-document pairs: Grade 2 means relevant and Grade 1 means partially relevant. One candidate remains unresolved (QID 223 × `1074_Game Modes`) and is not a positive qrel. The original qrels and frozen files remain unchanged.

Limitations: negative candidates were not all individually reviewed; the original split seed is recorded as 42 but its generation procedure is unavailable and cannot be reproduced; one candidate remains unresolved. The labels are sparse and reference-derived, and the corpus is imbalanced across games.

## Split decision and leakage

The existing files assign 156 train, 33 validation, and 34 test queries. Keep the 34 original test QIDs locked. Use train plus validation as the provisional development set, with **QID 181 quarantined from training, tuning, and evaluation**. This leaves 188 development queries and 34 test queries. The effective qrel view is 251 development pairs and 42 test pairs; QID 181's Grade 2 pair remains in the frozen 294-pair qrels but is excluded from experiment inputs and metrics. Existing split files are preserved as provenance.

The split audit found no cross-boundary exact question or answer duplicates. Using the notebook's normalization and word unigram/bigram TF-IDF checks, there were no cross-boundary question pairs at 0.80 or answer pairs at 0.85; the largest cross-boundary similarities were 0.4075 for questions and 0.7068 for answers. Manual comparison found a clear near-duplicate pair the lexical thresholds missed: QID 181 is in train and QID 184 is in test. Their questions ask how to remove an unauthorized linked PUBG MOBILE login; their answers and reference snippet are nearly identical, and both cite the same Grade 2 document (`2024-08-20-Account Protection Features`). Quarantining QID 181 preserves test membership and prevents this known query-level leak.

The notebook's embedding check was rerun with its configured model, `sentence-transformers/all-MiniLM-L6-v2`, on the 223 usable questions. The two pairs at cosine similarity ≥0.90 are QID 113/118 (0.954491) and QID 134/135 (0.915173). Both pairs are wholly within development; neither crosses the locked test boundary. Along with the lexical-threshold check and the known QID 181/184 near-duplicate, no remaining cross-boundary duplicate was found. The gate passes. Keep the one development-side quarantine, QID 181, and keep all 34 test QIDs unchanged. Do not change the frozen qrels.

Shared evidence is a limitation, not by itself leakage: 23 distinct reference snippets are shared across 40 effective development-test query pairs (19 test queries affected; 36 development queries involved). The qrels share 25 relevant document IDs across the boundary; 31 of 34 test queries have at least one positive document also relevant to a development query. Retrieval uses the same corpus for every query, as it should. Never feed references, answers, qrels, or `retrieved_docs` into the retriever or ranker. Use only the question, and optionally explicit `game_name` when the experiment tests that routing condition.

## Metrics and qrel rules

Primary retrieval metric: **nDCG@10**, using graded gains of 3 for Grade 2 and 1 for Grade 1 (`2^grade - 1`). Do not collapse Grade 1 into full relevance for graded scoring. Also report Grade 2-only Recall@5/10 and MRR@10, Grade 1 and Grade 2 counts retrieved separately, and any-positive Hit@5/10 with its binary-any-positive definition stated. Macro-average by query; report results by game, task type, and query form. Use paired query-level comparisons with confidence intervals; do not tune on test results.

Every effective development and test query has at least one positive qrel. The effective development set has 188 queries / 251 positive pairs (223 Grade 2, 28 Grade 1); test has 34 queries / 42 pairs (35 Grade 2, 7 Grade 1). The original 294-pair qrel file remains authoritative. Exclude QID 181 only when constructing experiment inputs and reported evaluation metrics.

## Four-phase plan and controls

**Fixed across phases:** frozen corpus and qrels; exact QID-to-source mapping; the effective development/test policy above; one change per ablation; no test-driven model, prompt, threshold, or preprocessing selection; retain raw text and stable document IDs. Record code/version, model checkpoint, tokenizer, hardware, random seed, index settings, latency, and token/compute cost. Report the selected development configuration once on the locked test set.

### 1. Documents

Start from raw title and contents. Compare title-only, contents-only, and title-plus-contents. Compare raw text with conservative normalization (Unicode, whitespace, markup only); avoid stemming, stop-word removal, and aggressive lemmatization as default processing. Establish sparse BM25/TF-IDF baselines, then compare dense encoders (all-MiniLM-L6-v2, all-mpnet-base-v2, and E5-base) with the same query set and evaluation. Chunk only if the tokenizer audit shows material truncation, preserving source-document IDs. Test contrastive adaptation only after the frozen pretrained baseline, using development data only. Compare clustering methods using silhouette/stability and cluster diagnostics, but select clusters only if they improve downstream retrieval over flat search.

### 2. Queries

Baseline with the original English question. Create Devanagari Hindi with IndicTrans2 and Romanized Hindi with IndicXlit; preserve game, item, character, error-code, number, and version strings. Validate meaning and entity preservation before retrieval. Keep every translation/transliteration attached to its source QID and in the same split. Label these as machine-translated/transliterated queries; do not claim they represent naturally typed Hinglish. Compare query cleaning, expansion, translation, and multilingual encoders without changing the corpus or qrels.

### 3. Retrieval

Establish BM25 and BGE-M3 dense baselines, then hybrid reciprocal-rank fusion with `k=60`. Ablate explicit game-name routing, a cross-encoder reranker over the top 50, and MMR (`lambda=0.7`) to return five distinct documents. Keep retrieval cutoffs and the evaluation metrics fixed. Route only when the query names a game; report routing failures and recall loss. The deck's earlier 223-query scores are pilot results, not the result of this frozen protocol.

### 4. Synthesis

Baseline: retrieve-then-read using the same top five distinct documents and a fixed context-token budget. Compare one synthesis change at a time: redundancy filtering, contextual compression, context reordering, corrective retrieval, and structured/dense answer formatting. Before Phase 4, freeze and report the generation model/checkpoint, prompt, decoding settings, maximum output length, and context budget; do not select these on test. Measure answer correctness against the reference, claim-level evidence support/citations, key-fact coverage/completeness, redundancy, token use, latency, and extra calls. Use the same retrieved set when isolating synthesis methods.

## Cross-phase selection and integrated evaluation

Choose each phase's component using development results and the stated metrics, with cost and stability reported alongside quality. Carry a component forward only when its gain is repeatable and attributable to the controlled change. Evaluate the integrated pipeline with selected components on the same frozen corpus, query set, qrels, and locked test QIDs. Compare it with the fixed BM25 and retrieve-then-read baselines, and report end-to-end retrieval, answer quality, grounding, coverage, and cost. Keep stage-level ablations so an end-to-end gain can be attributed.

## Immediate first experiment

The gate has cleared. Run a development-only BM25 reference run over all 4,192 documents using raw `title + contents`, original English questions, no game filter, and no answer/reference/context fields as input. Use the 188-query effective development set and its 251 qrels. Use Okapi BM25 with k1=1.2 and b=0.75, Robertson-Sparck Jones IDF `ln(1 + (N-df+0.5)/(df+0.5))`, Unicode casefolded word unigrams matched by `(?u)\b\w\w+\b`, no stop-word removal or stemming, and deterministic document-ID ascending tie-breaking. Rank all 4,192 documents per query. Required outputs: a ranked run file with QID, document ID, rank, and score; nDCG@10, Grade 2-only MRR@10 and Recall@5/10, Grade 1/2 retrieval counts, any-positive Hit@5/10; per-game and per-task metrics; index/build and query latency; exact code, tokenizer, and BM25 settings. Do not score or inspect test results during this experiment.
