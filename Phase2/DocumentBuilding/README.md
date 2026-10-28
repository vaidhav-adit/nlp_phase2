# Document Building

This stage creates one retrieval unit per frozen corpus document using `Title: {title}\nContent: {contents}`. It preserves source IDs and applies only NFC Unicode normalization and deterministic newline normalization; it does not filter by game, chunk, summarize, stem, lemmatize, or remove stop words.

The provisional encoder is `sentence-transformers/all-MiniLM-L6-v2`, pinned at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. CPU is used because MPS is unavailable in the current runtime. Notebook 01 records the actual tokenizer/model limits and identifies documents whose inputs exceed that limit; it does not silently change the model limit or chunk documents.

Notebook 02 generates one L2-normalized vector per document in batches, writes a FAISS `IndexFlatIP`, a deterministic position-to-document-ID mapping, and a configuration manifest. Notebook 03 independently reloads and checks corpus/index count, IDs, vector dimensions and norms, manifest agreement, and deterministic smoke-search results.

Artifacts are written under `data/`, `embeddings/`, `indexes/`, `manifests/`, and `logs/`. The original corpus stays in `../02_FROZEN_DATA/`; notebooks read it in place. The index is a provisional representation for Phase 2, not a claim of optimality. If Phase 1 later selects another encoder or document representation, make a separate versioned configuration; do not replace this one.
