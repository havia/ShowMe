"""
index.py  —  Multimodal Video Search Engine
============================================
Reads the raw JSON output from ingest.py, chunks the text,
computes text embeddings, and stores everything in ChromaDB.

Run:
    python index.py --raw data/index/raw/ --db data/index/chroma/
"""

import argparse
import json
from pathlib import Path

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

TEXT_MODEL_NAME  = "paraphrase-multilingual-mpnet-base-v2"  # multilingual, 768-dim, supports Hebrew
CHUNK_WINDOW_SEC = 15.0                  # cap chunk duration
CHUNK_MAX_CHARS  = 500                   # cap chunk length in characters
CLIP_DIM         = 512                   # CLIP ViT-B/32 output size
TEXT_DIM         = 768                   # mpnet-base output size

# Collection names inside ChromaDB
COLL_TEXT   = "text_chunks"    # ASR + OCR text with text embeddings
COLL_VISUAL = "visual_frames"  # CLIP frame embeddings


# ─────────────────────────────────────────────
# Chunking helpers
# ─────────────────────────────────────────────

def chunk_asr(segments: list[dict], window_sec: float = CHUNK_WINDOW_SEC) -> list[dict]:
    """
    Merge consecutive ASR segments whose combined span ≤ window_sec
    into single chunks. Each chunk keeps start/end timestamps.

    Input:  [{ "start", "end", "text" }, …]
    Output: [{ "start", "end", "text", "source" }, …]
    """
    if not segments:
        return []

    chunks = []
    current = dict(segments[0])
    current["source"] = "asr"

    for seg in segments[1:]:
        span_after_merge = seg["end"] - current["start"]
        chars_after_merge = len(current["text"]) + 1 + len(seg["text"])
        if span_after_merge <= window_sec and chars_after_merge <= CHUNK_MAX_CHARS:
            # Extend the current chunk
            current["end"]   = seg["end"]
            current["text"] += " " + seg["text"]
        else:
            chunks.append(current)
            current = {"start": seg["start"], "end": seg["end"],
                       "text": seg["text"], "source": "asr"}

    chunks.append(current)
    return chunks


def chunk_ocr(records: list[dict]) -> list[dict]:
    """
    OCR records are already timestamped per slide.
    Deduplicate consecutive identical texts (same slide shown across frames).

    Input:  [{ "timestamp", "text" }, …]
    Output: [{ "start", "end", "text", "source" }, …]
    """
    if not records:
        return []

    chunks = []
    prev_text = None
    current = None

    for rec in records:
        text = rec["text"].strip()
        if text == prev_text:
            # Same slide — extend the end timestamp
            if current:
                current["end"] = rec["timestamp"]
        else:
            if current:
                chunks.append(current)
            current = {
                "start":  rec["timestamp"],
                "end":    rec["timestamp"],
                "text":   text,
                "source": "ocr",
            }
            prev_text = text

    if current:
        chunks.append(current)

    return chunks


# ─────────────────────────────────────────────
# Indexing
# ─────────────────────────────────────────────

def build_text_index(
    raw_dir: Path,
    text_coll: chromadb.Collection,
    text_model: SentenceTransformer,
):
    """
    For each video in raw_dir:
      1. Load asr.json + ocr.json
      2. Chunk
      3. Embed with sentence-transformers
      4. Upsert into ChromaDB text collection
    """
    video_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    print(f"[text index] Processing {len(video_dirs)} video(s) …")

    for vdir in video_dirs:
        video_id = vdir.name

        # Load raw data
        asr_path = vdir / "asr.json"
        ocr_path = vdir / "ocr.json"

        asr_segments = json.loads(asr_path.read_text(encoding="utf-8")) if asr_path.exists() else []
        ocr_records  = json.loads(ocr_path.read_text(encoding="utf-8")) if ocr_path.exists() else []

        # Chunk
        asr_chunks = chunk_asr(asr_segments)
        ocr_chunks = chunk_ocr(ocr_records)
        all_chunks = asr_chunks + ocr_chunks

        if not all_chunks:
            print(f"  [{video_id}] No text chunks — skipping.")
            continue

        texts = [c["text"] for c in all_chunks]

        # Embed (batch for efficiency)
        print(f"  [{video_id}] Embedding {len(texts)} text chunks …")
        embeddings = text_model.encode(
            texts, batch_size=64, show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        # Build ChromaDB records
        ids       = [f"{video_id}__{i}" for i in range(len(all_chunks))]
        metadatas = [
            {
                "video_id": video_id,
                "start":    c["start"],
                "end":      c["end"],
                "source":   c["source"],   # "asr" or "ocr"
            }
            for c in all_chunks
        ]

        # Upsert (safe to re-run)
        text_coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  [{video_id}] ✓ {len(ids)} text chunks indexed.")


def build_visual_index(
    raw_dir: Path,
    visual_coll: chromadb.Collection,
):
    """
    For each video in raw_dir:
      1. Load clip.json  (embeddings already computed by ingest.py)
      2. Upsert into ChromaDB visual collection
    """
    video_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    print(f"[visual index] Processing {len(video_dirs)} video(s) …")

    for vdir in video_dirs:
        video_id = vdir.name
        clip_path = vdir / "clip.json"

        if not clip_path.exists():
            print(f"  [{video_id}] No clip.json — skipping.")
            continue

        clip_records = json.loads(clip_path.read_text(encoding="utf-8"))
        print(f"  [{video_id}] Indexing {len(clip_records)} frames …")

        ids        = [f"{video_id}_frame_{r['timestamp']}" for r in clip_records]
        embeddings = [r["embedding"] for r in clip_records]
        metadatas  = [{"video_id": video_id, "timestamp": r["timestamp"]} for r in clip_records]
        # ChromaDB requires documents — store timestamp as a string placeholder
        documents  = [f"frame at {r['timestamp']}s" for r in clip_records]

        # Upsert in batches of 1000 (ChromaDB default limit)
        batch = 1000
        for start in tqdm(range(0, len(ids), batch), desc=f"  [{video_id}] batches"):
            visual_coll.upsert(
                ids=ids[start:start+batch],
                embeddings=embeddings[start:start+batch],
                documents=documents[start:start+batch],
                metadatas=metadatas[start:start+batch],
            )

        print(f"  [{video_id}] ✓ {len(ids)} frames indexed.")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build vector index from ingested videos.")
    parser.add_argument(
        "--raw",  type=Path, default=Path("data/index/raw"),
        help="Raw JSON directory from ingest.py",
    )
    parser.add_argument(
        "--db",   type=Path, default=Path("data/index/chroma"),
        help="ChromaDB storage path",
    )
    args = parser.parse_args()

    # ── ChromaDB client (local persistent storage) ────────────────
    args.db.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.db))

    # Get or create collections
    # embedding_function=None → we supply our own vectors
    text_coll   = client.get_or_create_collection(
        COLL_TEXT,
        metadata={"hnsw:space": "cosine"},
    )
    visual_coll = client.get_or_create_collection(
        COLL_VISUAL,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Text embedding model ──────────────────────────────────────
    print(f"[index] Loading text model: {TEXT_MODEL_NAME} …")
    text_model = SentenceTransformer(TEXT_MODEL_NAME)

    # ── Build indexes ─────────────────────────────────────────────
    build_text_index(args.raw, text_coll, text_model)
    build_visual_index(args.raw, visual_coll)

    print(f"\n✓ Indexing complete.")
    print(f"  Text chunks  : {text_coll.count()}")
    print(f"  Visual frames: {visual_coll.count()}")
    print(f"  DB path      : {args.db}")


if __name__ == "__main__":
    main()
