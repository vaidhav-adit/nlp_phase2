# ChronoPlay-QA Team Handoff & Phase 2 Pipeline

This repository contains the verified, frozen **ChronoPlay-QA** retrieval benchmark dataset, baseline evaluation artifacts, and the Phase 2 research pipeline.

## Repository Overview

- **`01_READ_FIRST/`**: Team handoff instructions, complete research protocol, non-negotiable experimental rules, and project presentation.
- **`02_FROZEN_DATA/`**: Verified, immutable benchmark datasets:
  - `Corpus_4192_Gaming_Documents.csv`: 4,192 gaming documents.
  - `Original_Queries_232_with_Usability_Fields.csv`: 232 original queries with usability flags.
  - `Final_Relevance_Judgments_294_Pairs.csv`: 294 frozen relevance judgments (259 Grade 2, 35 Grade 1).
  - `Reference_to_Corpus_Document_Mapping.csv`: Reference-to-corpus document mapping.
- **`03_SPLITS_AND_EVALUATION/`**: Split guide and `Frozen_Split_and_Leakage_Audit_Manifest.json` (Seed 42, 188 development queries, 34 locked test queries).
- **`04_BASELINE_RESULTS/`**: BM25 development reference runs and metrics.
- **`Phase2/`**: Phase 2 Query Representation and Processing:
  - `DocumentBuilding/`: Corpus ingestion, token analysis, MiniLM document embeddings, and FAISS index generation.
  - `CleaningExperiments/`: Query cleaning conditions (C0, C1, C2), ranked runs, and metrics comparison.
  - `requirements.txt`: Environment dependencies.
- **`HANDOFF_SHA256_MANIFEST.json`**: Provenance hashes for all frozen data files.

## Getting Started

1. Read [`01_READ_FIRST/README_Team_Handoff.md`](01_READ_FIRST/README_Team_Handoff.md) and [`01_READ_FIRST/Research_Protocol_and_Experimental_Rules.md`](01_READ_FIRST/Research_Protocol_and_Experimental_Rules.md).
2. For Phase 2 experiments, install dependencies:
   ```bash
   pip install -r Phase2/requirements.txt
   ```
3. Follow the notebook workflow detailed in [`Phase2/README.md`](Phase2/README.md).
