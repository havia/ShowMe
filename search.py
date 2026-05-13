"""
search.py  —  Multimodal Video Search Engine
=============================================
Encodes a user query, searches both text and visual indexes,
fuses and reranks the results, and returns timestamped hits.

Can be used as a module (from app.py) or run standalone:
    python search.py --query "RAG retrieval augmented generation" --db data/index/chroma/
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


# ─────────────────────────────────────────────
# Config  (must match index.py)
# ─────────────────────────────────────────────

TEXT_MODEL_NAME  = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME  = "openai/clip-vit-base-patch32"
COLL_TEXT        = "text_chunks"
COLL_VISUAL      = "visual_frames"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_TEXT       = 10   # text candidates per query
TOP_K_VISUAL     = 5    # visual candidates per query
TEXT_WEIGHT      = 0.6  # weight for text score in fusion
VISUAL_WEIGHT    = 0.4  # weight for visual score in fusion
DEDUP_WINDOW_SEC = 3.0  # suppress results within N sec of a higher-scored hit


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass(order=True)
class SearchResult:
    score:    float          # fused relevance score (higher = better)
    video_id: str = field(compare=False)
    start:    float = field(compare=False)   # timestamp in seconds
    end:      float = field(compare=False)
    text:     str   = field(compare=False)   # snippet shown to user
    source:   str   = field(compare=False)   # "asr" | "ocr" | "visual"


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

class SearchEngine:
    """
    Stateful search engine — load once, query many times.
    Used by app.py.
    """

    def __init__(self, db_path: str | Path):
        db_path = Path(db_path)
        print(f"[search] Connecting to ChromaDB at {db_path} …")
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.text_coll   = self.client.get_collection(COLL_TEXT)
        self.visual_coll = self.client.get_collection(COLL_VISUAL)

        print(f"[search] Loading text model …")
        self.text_model = SentenceTransformer(TEXT_MODEL_NAME)

        print(f"[search] Loading CLIP …")
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        self.clip_model.eval()

        print("[search] Ready.")

    # ── Query encoders ────────────────────────────────────────────

    def encode_text(self, query: str) -> list[float]:
        return self.text_model.encode(
            [query], normalize_embeddings=True
        )[0].tolist()

    def encode_query_clip(self, query: str) -> list[float]:
        """Encode a text query with CLIP's text encoder for cross-modal search."""
        inputs = self.clip_processor(text=[query], return_tensors="pt",
                                     padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            emb = self.clip_model.get_text_features(**inputs)
            if not isinstance(emb, torch.Tensor):
                if hasattr(emb, "text_embeds") and emb.text_embeds is not None:
                    emb = emb.text_embeds
                elif hasattr(emb, "pooler_output") and emb.pooler_output is not None:
                    emb = emb.pooler_output
                else:
                    raise TypeError(f"Unexpected CLIP output type: {type(emb)!r}")
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().cpu().tolist()

    # ── Individual modality searches ──────────────────────────────

    def _search_text(self, query_vec: list[float], video_filter: str | None, n: int) -> list[dict]:
        where = {"video_id": video_filter} if video_filter else None
        results = self.text_coll.query(
            query_embeddings=[query_vec],
            n_results=min(n, self.text_coll.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns cosine *distance* (0=identical, 2=opposite)
            # Convert to similarity score in [0, 1]
            score = 1.0 - dist / 2.0
            hits.append({
                "score":    score,
                "video_id": meta["video_id"],
                "start":    meta["start"],
                "end":      meta["end"],
                "text":     doc,
                "source":   meta.get("source", "asr"),
            })
        return hits

    def _search_visual(self, query_vec: list[float], video_filter: str | None, n: int) -> list[dict]:
        where = {"video_id": video_filter} if video_filter else None
        results = self.visual_coll.query(
            query_embeddings=[query_vec],
            n_results=min(n, self.visual_coll.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist / 2.0
            ts = meta["timestamp"]
            hits.append({
                "score":    score,
                "video_id": meta["video_id"],
                "start":    ts,
                "end":      ts + 1.0,
                "text":     f"[visual match @ {ts:.1f}s]",
                "source":   "visual",
            })
        return hits

    # ── Fusion + deduplication ────────────────────────────────────

    @staticmethod
    def _fuse_and_rank(text_hits: list[dict], visual_hits: list[dict]) -> list[SearchResult]:
        """
        Simple weighted score fusion.
        Hits from the same video within DEDUP_WINDOW_SEC of a higher-scored hit
        are removed to avoid showing the same moment multiple times.
        """
        combined: list[SearchResult] = []

        for h in text_hits:
            combined.append(SearchResult(
                score    = h["score"] * TEXT_WEIGHT,
                video_id = h["video_id"],
                start    = h["start"],
                end      = h["end"],
                text     = h["text"],
                source   = h["source"],
            ))

        for h in visual_hits:
            combined.append(SearchResult(
                score    = h["score"] * VISUAL_WEIGHT,
                video_id = h["video_id"],
                start    = h["start"],
                end      = h["end"],
                text     = h["text"],
                source   = h["source"],
            ))

        # Merge hits that are very close in time (same moment, different sources)
        merged: list[SearchResult] = []
        combined.sort(key=lambda x: -x.score)

        for candidate in combined:
            duplicate = False
            for kept in merged:
                if (kept.video_id == candidate.video_id and
                        abs(kept.start - candidate.start) < DEDUP_WINDOW_SEC):
                    # Add scores (reward multi-modal agreement)
                    kept.score += candidate.score * 0.3
                    duplicate = True
                    break
            if not duplicate:
                merged.append(candidate)

        merged.sort(key=lambda x: -x.score)
        return merged

    # ── Public search API ─────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        video_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Search across all indexed videos (or one specific video).

        Args:
            query:        natural language search phrase
            top_k:        number of results to return
            video_filter: optional video_id to restrict search scope

        Returns:
            Sorted list of SearchResult (best first).
        """
        text_vec   = self.encode_text(query)
        visual_vec = self.encode_query_clip(query)

        text_hits   = self._search_text(text_vec,   video_filter, TOP_K_TEXT)
        visual_hits = self._search_visual(visual_vec, video_filter, TOP_K_VISUAL)

        results = self._fuse_and_rank(text_hits, visual_hits)
        return results[:top_k]


# ─────────────────────────────────────────────
# CLI entry point (quick test)
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query the video search engine.")
    parser.add_argument("--query",  type=str,  required=True)
    parser.add_argument("--db",     type=Path, default=Path("data/index/chroma"))
    parser.add_argument("--top_k",  type=int,  default=5)
    parser.add_argument("--video",  type=str,  default=None,
                        help="Restrict to a specific video_id")
    args = parser.parse_args()

    engine  = SearchEngine(args.db)
    results = engine.search(args.query, top_k=args.top_k, video_filter=args.video)

    print(f"\nTop {len(results)} results for: \"{args.query}\"\n")
    print(f"{'#':<3} {'Score':>6}  {'Video':<20} {'Time':>8}  {'Src':<7}  Snippet")
    print("─" * 80)
    for i, r in enumerate(results, 1):
        ts = f"{int(r.start//60):02d}:{int(r.start%60):02d}"
        snippet = r.text[:55].replace("\n", " ")
        print(f"{i:<3} {r.score:>6.3f}  {r.video_id:<20} {ts:>8}  {r.source:<7}  {snippet}")


if __name__ == "__main__":
    main()
