"""Compare MiniLM document input caps 256 and 512 on development queries only.

The 256-token artifacts are immutable inputs. The 512-token vectors, FAISS
index, runs, metrics, and manifest use new versioned filenames.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import unicodedata
import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import sentence_transformers
import torch
import transformers
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=None,
                    help="Handoff folder containing 02_FROZEN_DATA (auto-detected from this script by default).")
parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
parser.add_argument("--batch-size", type=int, default=None,
                    help="Inference batch size. Default: 8 on CUDA, 4 otherwise.")
args = parser.parse_args()
if args.root is not None:
    ROOT = args.root.expanduser().resolve()
else:
    ROOT = next((p for p in Path(__file__).resolve().parents if (p / "02_FROZEN_DATA").is_dir()), None)
if ROOT is None or not (ROOT / "02_FROZEN_DATA").is_dir():
    raise FileNotFoundError("Set --root to the ChronoPlay handoff folder containing 02_FROZEN_DATA.")
BUILD = ROOT / "Phase2" / "DocumentBuilding"
OUT = BUILD / "outputs"
EMBED = BUILD / "embeddings"
INDEX_DIR = BUILD / "indexes"
MAP_DIR = BUILD / "manifests"
for folder in (OUT, EMBED, INDEX_DIR, MAP_DIR):
    folder.mkdir(parents=True, exist_ok=True)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CORPUS_PATH = ROOT / "02_FROZEN_DATA" / "Corpus_4192_Gaming_Documents.csv"
QUERY_PATH = ROOT / "02_FROZEN_DATA" / "Original_Queries_232_with_Usability_Fields.csv"
QRELS_PATH = ROOT / "02_FROZEN_DATA" / "Final_Relevance_Judgments_294_Pairs.csv"
DEV_QIDS_PATH = MAP_DIR / "effective_development_qids.json"
DOC_MANIFEST_PATH = BUILD / "data" / "chronoplay_document_manifest_v1.csv"
BASELINE_INDEX_PATH = INDEX_DIR / "chronoplay_minilm_faiss_index_v1.faiss"
BASELINE_MAP_PATH = MAP_DIR / "chronoplay_document_id_mapping_v1.csv"
BASELINE_CONFIG_PATH = MAP_DIR / "chronoplay_minilm_document_embedding_manifest_v1.json"
HF_HOME = ROOT / "NLP" / "hf_cache"
os.environ["HF_HOME"] = str(HF_HOME)
HF_HOME.mkdir(parents=True, exist_ok=True)

if args.device == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = args.device
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU.")
if DEVICE == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("MPS was requested but PyTorch cannot see Apple Metal support.")
BATCH_SIZE = args.batch_size if args.batch_size is not None else (8 if DEVICE == "cuda" else 4)
if BATCH_SIZE < 1:
    raise ValueError("--batch-size must be positive.")

EMBED_PATH = EMBED / "chronoplay_minilm_document_embeddings_512_v1.npy"
INDEX_PATH = INDEX_DIR / "chronoplay_minilm_faiss_index_512_v1.faiss"
MAP_PATH = MAP_DIR / "chronoplay_minilm_document_id_mapping_512_v1.csv"
MANIFEST_PATH = MAP_DIR / "chronoplay_minilm_document_embedding_manifest_512_v1.json"
RUN_PATH = OUT / "dense_ranked_run_doccap256_vs512_dev_v1.csv"
QUERY_METRICS_PATH = OUT / "metrics_by_query_doccap256_vs512_dev_v1.csv"
METRICS_PATH = OUT / "metrics_comparison_doccap256_vs512_dev_v1.csv"
GROUP_PATH = OUT / "metrics_by_game_task_doccap256_vs512_dev_v1.csv"
DIFF_PATH = OUT / "paired_differences_doccap512_minus256_dev_v1.csv"
REPORT_PATH = OUT / "experiment_report_doccap256_vs512_dev_v1.json"
OUTPUTS = (EMBED_PATH, INDEX_PATH, MAP_PATH, MANIFEST_PATH, RUN_PATH,
           QUERY_METRICS_PATH, METRICS_PATH, GROUP_PATH, DIFF_PATH, REPORT_PATH)
existing = [str(p) for p in OUTPUTS if p.exists()]
if existing:
    raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(existing))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if DEVICE == "cpu":
    torch.set_num_threads(1)
faiss.omp_set_num_threads(1)
corpus = pd.read_csv(CORPUS_PATH, usecols=["id", "title", "contents"], dtype={"id": "string"})
doc_manifest = pd.read_csv(DOC_MANIFEST_PATH, dtype={"id": "string"})
assert len(corpus) == len(doc_manifest) == 4192
assert corpus.id.tolist() == doc_manifest.id.tolist()

def norm_newlines(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n").replace("\r", "\n")

doc_texts = [f"Title: {norm_newlines(t)}\nContent: {norm_newlines(c)}"
             for t, c in zip(corpus.title, corpus.contents)]
doc_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in doc_texts]
assert doc_hashes == doc_manifest.document_text_sha256.tolist()

dev_info = json.loads(DEV_QIDS_PATH.read_text())
dev_qids = set(map(str, dev_info["qids"]))
assert len(dev_qids) == 188 and "181" not in dev_qids and dev_info.get("locked_test_read") is False
queries = pd.read_csv(QUERY_PATH, usecols=["qid", "question", "game_name", "task_type"], dtype={"qid": "string"})
queries = queries.loc[queries.qid.isin(dev_qids)].copy()
assert set(queries.qid) == dev_qids and len(queries) == 188
assert queries.question.notna().all()

base_cfg = json.loads(BASELINE_CONFIG_PATH.read_text())
assert base_cfg["model_revision"] == MODEL_REVISION and base_cfg["model_max_seq_length"] == 256
base_mapping = pd.read_csv(BASELINE_MAP_PATH, dtype={"document_id": "string"})
assert base_mapping.document_id.tolist() == corpus.id.astype(str).tolist()
base_index = faiss.read_index(str(BASELINE_INDEX_PATH))
assert base_index.ntotal == 4192 and base_index.d == 384

snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
                                  cache_dir=str(HF_HOME), local_files_only=False,
                                  allow_patterns=["*.json", "*.txt", "*.safetensors"]))
assert snapshot.name == MODEL_REVISION
model = SentenceTransformer(str(snapshot), device=DEVICE)
assert model.max_seq_length == 256 and model.get_sentence_embedding_dimension() == 384

# Keep query encoding identical to the existing 256-token baseline.
model.max_seq_length = 256
query_start = time.perf_counter()
query_vectors = np.asarray(model.encode(
    queries.question.astype(str).tolist(), batch_size=BATCH_SIZE, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True, device=DEVICE), dtype=np.float32)
query_encode_seconds = time.perf_counter() - query_start
assert query_vectors.shape == (188, 384) and np.isfinite(query_vectors).all()

# Measure full input lengths and truncation at the two candidate document caps.
lengths = []
for start in range(0, len(doc_texts), 128):
    batch = model.tokenizer(doc_texts[start:start + 128], add_special_tokens=True,
                            truncation=False, return_attention_mask=False)
    lengths.extend(map(len, batch["input_ids"]))
assert len(lengths) == 4192
truncation = {}
for cap in (256, 512):
    dropped = [max(0, n - cap) for n in lengths]
    truncation[str(cap)] = {
        "documents_truncated": int(sum(x > 0 for x in dropped)),
        "percent_documents_truncated": 100 * sum(x > 0 for x in dropped) / len(dropped),
        "total_wordpieces_discarded": int(sum(dropped)),
        "maximum_wordpieces_discarded_in_one_document": int(max(dropped)),
        "input_cap_includes_special_tokens": True,
    }

# Set the separate experimental document input cap; do not modify the 256 artifacts.
model.max_seq_length = 512
assert model.max_seq_length == 512
tokenizer_cap = model.tokenizer.model_max_length
model_config_max = model[0].auto_model.config.max_position_embeddings
assert tokenizer_cap == 512 and model_config_max == 512

embed_start = time.perf_counter()
raw = model.encode(doc_texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                   convert_to_numpy=True, normalize_embeddings=False, device=DEVICE)
raw = np.asarray(raw, dtype=np.float32)
assert raw.shape == (4192, 384) and np.isfinite(raw).all()
norms = np.linalg.norm(raw, axis=1, keepdims=True)
assert np.all(norms > 0)
embeddings = np.ascontiguousarray(raw / norms, dtype=np.float32)
embedding_seconds = time.perf_counter() - embed_start
assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)

index = faiss.IndexFlatIP(384)
index.add(embeddings)
assert index.ntotal == 4192
mapping = pd.DataFrame({"vector_position": np.arange(4192, dtype=np.int64),
                        "document_id": corpus.id.astype(str)})

qrels = pd.read_csv(QRELS_PATH, usecols=["query_id", "corpus_doc_id", "relevance"],
                    dtype={"query_id": "string", "corpus_doc_id": "string"})
qrels = qrels.loc[qrels.query_id.isin(dev_qids)].copy()
qrels.relevance = qrels.relevance.astype(int)
assert len(qrels) == 251 and set(qrels.query_id) == dev_qids
qrels_by_qid = {str(qid): dict(zip(g.corpus_doc_id, g.relevance.astype(int)))
                for qid, g in qrels.groupby("query_id")}
assert set(qrels_by_qid) == dev_qids and all(qrels_by_qid.values())

def retrieve(which: str, faiss_index: faiss.Index) -> pd.DataFrame:
    started = time.perf_counter()
    scores, positions = faiss_index.search(query_vectors, 100)
    search_seconds = time.perf_counter() - started
    rows = []
    for qi, qid in enumerate(queries.qid.astype(str)):
        for rank, (score, pos) in enumerate(zip(scores[qi], positions[qi]), start=1):
            rows.append({"condition": which, "query_id": qid,
                         "document_id": str(mapping.iloc[int(pos)].document_id),
                         "rank": rank, "score": float(score)})
    return pd.DataFrame(rows), search_seconds

base_run, base_search_seconds = retrieve("doc_cap_256", base_index)
run_512, search512_seconds = retrieve("doc_cap_512", index)
run_df = pd.concat([base_run, run_512], ignore_index=True)
assert len(run_df) == 2 * 188 * 100

def metrics_for(ranked_ids: list[str], grades: dict[str, int]) -> dict[str, float | bool]:
    top10 = ranked_ids[:10]
    dcg = sum(((2 ** grades.get(doc, 0)) - 1) / np.log2(rank + 1)
              for rank, doc in enumerate(top10, 1) if grades.get(doc, 0) > 0)
    ideal = sorted(grades.values(), reverse=True)[:10]
    idcg = sum(((2 ** grade) - 1) / np.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    grade2 = {doc for doc, grade in grades.items() if grade == 2}
    any_positive = set(grades)
    first2 = next((rank for rank, doc in enumerate(ranked_ids[:10], 1) if doc in grade2), None)
    return {
        "ndcg10": dcg / idcg if idcg else np.nan,
        "g2_mrr10": (1 / first2 if first2 else (np.nan if not grade2 else 0.0)),
        "g2_recall5": len(set(ranked_ids[:5]) & grade2) / len(grade2) if grade2 else np.nan,
        "g2_recall10": len(set(ranked_ids[:10]) & grade2) / len(grade2) if grade2 else np.nan,
        "any_hit5": float(bool(set(ranked_ids[:5]) & any_positive)),
        "any_hit10": float(bool(set(ranked_ids[:10]) & any_positive)),
        "g1_retrieved10": sum(grades.get(doc) == 1 for doc in ranked_ids[:10]),
        "g2_retrieved10": sum(grades.get(doc) == 2 for doc in ranked_ids[:10]),
        "has_grade2_qrel": bool(grade2),
    }

meta = queries.set_index("qid")[["game_name", "task_type"]]
per_query = []
for condition, group in run_df.groupby("condition", sort=True):
    for qid, ranked in group.groupby("query_id", sort=False):
        ranked = ranked.sort_values("rank")
        per_query.append({"condition": condition, "query_id": str(qid),
                          **metrics_for(ranked.document_id.tolist(), qrels_by_qid[str(qid)])})
per_query_df = pd.DataFrame(per_query)
per_query_df["game_name"] = per_query_df.query_id.map(meta.game_name)
per_query_df["task_type"] = per_query_df.query_id.map(meta.task_type)
metric_cols = ["ndcg10", "g2_mrr10", "g2_recall5", "g2_recall10", "any_hit5", "any_hit10",
               "g1_retrieved10", "g2_retrieved10"]
overall = []
for condition, group in per_query_df.groupby("condition"):
    row = {"condition": condition, "n_queries": len(group),
           "n_queries_without_grade2_qrel": int((~group.has_grade2_qrel).sum())}
    for col in metric_cols:
        vals = group[col].fillna(0) if col in ("g2_mrr10", "g2_recall5", "g2_recall10") else group[col]
        row[col] = float(vals.mean())
        if col in ("g2_mrr10", "g2_recall5", "g2_recall10"):
            row[col + "_eligible_only"] = float(group.loc[group.has_grade2_qrel, col].mean())
    overall.append(row)
metrics_df = pd.DataFrame(overall)

group_rows = []
for field in ("game_name", "task_type"):
    for group_value, subset in per_query_df.groupby(field):
        for condition, group in subset.groupby("condition"):
            row = {"group_type": field, "group": group_value, "condition": condition, "n_queries": len(group)}
            for col in metric_cols:
                vals = group[col].fillna(0) if col in ("g2_mrr10", "g2_recall5", "g2_recall10") else group[col]
                row[col] = float(vals.mean())
            group_rows.append(row)
group_df = pd.DataFrame(group_rows)

wide = per_query_df.pivot(index="query_id", columns="condition", values=metric_cols)
rng = np.random.default_rng(42)
diff_rows = []
for metric in metric_cols:
    fill = metric in ("g2_mrr10", "g2_recall5", "g2_recall10")
    a = wide[(metric, "doc_cap_256")].fillna(0).to_numpy() if fill else wide[(metric, "doc_cap_256")].to_numpy()
    b = wide[(metric, "doc_cap_512")].fillna(0).to_numpy() if fill else wide[(metric, "doc_cap_512")].to_numpy()
    diffs = b - a
    indices = rng.integers(0, len(diffs), size=(10000, len(diffs)))
    boot = diffs[indices].mean(axis=1)
    diff_rows.append({"comparison": "doc_cap_512-doc_cap_256", "metric": metric,
                      "n_queries": len(diffs), "mean_difference": float(diffs.mean()),
                      "ci95_low": float(np.quantile(boot, 0.025)),
                      "ci95_high": float(np.quantile(boot, 0.975)),
                      "bootstrap_seed": 42, "bootstrap_replicates": 10000})
diff_df = pd.DataFrame(diff_rows)

pd.DataFrame({"vector_position": np.arange(4192, dtype=np.int64),
              "document_id": corpus.id.astype(str)}).to_csv(MAP_PATH, index=False)
np.save(EMBED_PATH, embeddings)
run_df.to_csv(RUN_PATH, index=False)
per_query_df.to_csv(QUERY_METRICS_PATH, index=False)
metrics_df.to_csv(METRICS_PATH, index=False)
group_df.to_csv(GROUP_PATH, index=False)
diff_df.to_csv(DIFF_PATH, index=False)

manifest = {
    "status": "PASS",
    "experiment": "development-only MiniLM document sequence cap 256 versus 512",
    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "dataset_version": "ChronoPlay-QA Claude Verified v1.1",
    "device": DEVICE, "batch_size": BATCH_SIZE, "corpus_count": 4192, "embedding_dimension": 384,
    "document_representation": "Title: {title}\\nContent: {contents}; NFC and newline normalization only",
    "chunking": "none; one vector per source document",
    "comparison_control": "same model weights/revision, corpus text, source order, normalized IP scoring, dev queries, query vectors at max length 256, top-100, qrels, metrics, and FAISS IndexFlatIP; only document max_seq_length differs",
    "document_caps": {"baseline": 256, "candidate": 512},
    "model_repository_configured_cap": 256,
    "architecture_max_position_embeddings": int(model_config_max),
    "tokenizer_model_max_length": int(tokenizer_cap),
    "training_sequence_length_from_model_card": 128,
    "truncation_statistics": truncation,
    "development_query_count": 188, "development_qrel_count": 251,
    "excluded_qids": ["181"], "locked_test_evaluated": False,
    "grade_metrics": "nDCG@10 gains 3 for Grade 2 and 1 for Grade 1; Grade-2-only MRR/Recall; any-positive Hit; Grade-2 metrics macro-as-zero and eligible-only; macro by query",
    "latency_seconds": {"shared_query_encoding_256": query_encode_seconds,
                        "document_embedding_512": embedding_seconds,
                        "search_256_index": base_search_seconds,
                        "search_512_index": search512_seconds},
    "bootstrap": {"paired": "512 minus 256", "seed": 42, "replicates": 10000},
    "source_sha256": {"corpus": sha256(CORPUS_PATH), "qrels": sha256(QRELS_PATH),
                      "development_qids_manifest": sha256(DEV_QIDS_PATH)},
    "outputs": {},
}
faiss.write_index(index, str(INDEX_PATH))
for key, path in {
    "embeddings_512": EMBED_PATH, "faiss_index_512": INDEX_PATH,
    "mapping_512": MAP_PATH, "ranked_run": RUN_PATH,
    "per_query_metrics": QUERY_METRICS_PATH, "overall_metrics": METRICS_PATH,
    "game_task_metrics": GROUP_PATH, "paired_differences": DIFF_PATH,
}.items():
    manifest["outputs"][key] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
report = {"status": "PASS", "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
          "overall_metrics": metrics_df.to_dict(orient="records"),
          "paired_differences": diff_df.to_dict(orient="records"),
          "truncation_statistics": truncation,
          "latency_seconds": manifest["latency_seconds"],
          "outputs": [str(p.relative_to(ROOT)) for p in OUTPUTS]}
REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(metrics_df.to_string(index=False))
print(diff_df.to_string(index=False))
print(json.dumps(truncation, indent=2))
print("REPORT", REPORT_PATH.relative_to(ROOT))
