# BM25 Development Baseline Interpretation

This is the protocol-specified development-only reference run. It is not a locked-test result and does not establish improvement or statistical significance.

| Metric | Development result |
|---|---:|
| nDCG@10 (graded) | 0.6932 |
| Grade 2 MRR@10 | 0.6469 |
| Grade 2 Recall@5 | 0.7634 |
| Grade 2 Recall@10 | 0.8325 |
| Any-positive Hit@5 | 0.8085 |
| Any-positive Hit@10 | 0.8723 |
| Mean query latency | 6.68 ms |
| p95 query latency | 13.09 ms |
| Index-build time | 0.497 s |

BM25 ranked all 4,192 documents for each of 188 effective development queries. It used raw title plus newline plus contents, original English questions, and no game filter. Answers, references, and `retrieved_docs` were not model inputs. The exact parameters, tokenizer, hashes, runtime, and latency fields are in `BM25_Development_Metrics_and_Settings.json`.

Grade 1 is partially relevant and remains distinct in nDCG gains. Grade 2-only metrics exclude Grade 1 judgments; any-positive hit treats either positive grade as a hit. Review the per-game/task breakdown in `BM25_Development_Per_Game_and_Task_Metrics.csv` alongside the macro scores.
