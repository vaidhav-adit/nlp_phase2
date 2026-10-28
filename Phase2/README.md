# ChronoPlay-QA Phase 2

Phase 2 studies query representation and processing. The current scope is Plan A's reusable document ingestion/dense retrieval reference pipeline and English query cleaning conditions C0, C1, and C2. The selected MiniLM encoder is a provisional Phase 2 reference, not a final project-wide encoder choice.

## Layout

- `DocumentBuilding/`: ingestion, token/truncation inspection, document embeddings, FAISS index, manifest, and independent validation notebooks.
- `CleaningExperiments/`: C0/C1/C2 development-only cleaning comparison, ranked runs, metrics, and report.
- `MANAGER_INSTRUCTIONS.md`: approval, provenance, notebook, and no-overwrite rules.
- `requirements.txt`: pinned Python package environment for this stage.

## Environment

The `NLP/` directory is the Python 3.12 virtual environment and contains the `NLP` kernel (`Python (NLP)`). Install with `NLP/bin/python -m pip install -r Phase2/requirements.txt`. All notebooks must select kernel `NLP`. The current host reports MPS unavailable, so the reference implementation uses CPU.

## Notebook order

1. `DocumentBuilding/notebooks/01_corpus_inspection_and_ingestion.ipynb`
2. `DocumentBuilding/notebooks/02_document_embeddings_and_faiss.ipynb`
3. `DocumentBuilding/notebooks/03_document_index_validation.ipynb`
4. `CleaningExperiments/notebooks/04_query_cleaning_c0_c1_c2.ipynb`

Do not run step 4 beyond C0/C1 and the C2 proposal gate until the user reviews proposed C2 corrections. If no defensible C2 entries are identified, obtain the user's choice before scoring an empty C2 condition.

## Reuse and outputs

The frozen corpus, source query export, and qrels remain in `../02_FROZEN_DATA/` and are read in place. The reusable dense index is stored under `DocumentBuilding/embeddings/` and `DocumentBuilding/indexes/`; its mapping and configuration are under `DocumentBuilding/manifests/`. Cleaning runs and reports are under `CleaningExperiments/outputs/` and `reports/`. Existing handoff outputs are read-only inputs; never overwrite them.

Future Plan A approaches (related-term expansion, Query2doc, generated variations/fine-tuning, contrastive fine-tuning, and query decomposition) are not implemented in this initial scope. Later encoder choices must use separate named configurations, preserving this MiniLM run.
