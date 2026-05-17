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
import os
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from dotenv import load_dotenv

load_dotenv()

try:
    from anthropic import Anthropic
except ModuleNotFoundError:
    Anthropic = None  # generation is optional; raw retrieval still works


# ─────────────────────────────────────────────
# Config  (must match index.py)
# ─────────────────────────────────────────────

TEXT_MODEL_NAME  = "intfloat/multilingual-e5-large"
CLIP_MODEL_NAME  = "openai/clip-vit-base-patch32"
COLL_TEXT        = "text_chunks"
COLL_VISUAL      = "visual_frames"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_TEXT       = 10   # text candidates per query
TOP_K_VISUAL     = 5    # visual candidates per query
TEXT_WEIGHT      = 0.6  # weight for text score in fusion
VISUAL_WEIGHT    = 0.4  # weight for visual score in fusion
DEDUP_WINDOW_SEC = 3.0  # suppress results within N sec of a higher-scored hit

ANCHOR_MODEL          = "claude-haiku-4-5"
ANCHOR_CANDIDATE_K    = 15   # how many results to give the LLM to choose from
ANCHOR_MAX_TOKENS     = 400
DENSITY_DECAY_SEC     = 300.0  # exp decay scale for the "neighbors nearby" boost

ANSWER_MAX_TOKENS     = 600
ANSWER_SYSTEM_PROMPT = """אתה עוזר ללומדים בקורס AI Engineering. תקבל שאילתת חיפוש של סטודנט ורשימה ממוספרת של קטעים מתומללים מהרצאות וידאו (כולל זמן וזיהוי הרצאה).

תפקידך: לכתוב תשובה קצרה בעברית (2-5 משפטים) שמסכמת מה המרצה הסביר על הנושא, **רק** על סמך הקטעים שניתנו. אל תמציא מידע שאינו בקטעים.

חוקים:
1. כתוב בעברית בלבד, גם אם השאילתה באנגלית.
2. בכל פעם שאתה מסתמך על קטע, הוסף ציטוט בפורמט [HH:MM:SS] מיד אחרי המשפט. אם רוצה לציין מספר זמנים, חזור על הפורמט: [02:13:15] [02:15:00].
3. אם הקטעים אינם מספקים תשובה ברורה, כתוב "המרצה מזכיר את הנושא אך לא נותן הסבר מפורט בקטעים שנמצאו" וצטט את הזמן הרלוונטי ביותר.
4. אל תפנה לקטעים לפי מספר ("קטע 3"), אלא לפי הזמן בלבד.
5. הקפד על קוהרנטיות: אם שני קטעים סותרים, ציין את שניהם בקצרה.

החזר טקסט בלבד, ללא JSON, ללא code fences."""

QUERY_EXPANSION_MAX_VARIANTS = 4
QUERY_EXPANSION_TOOL = {
    "name": "expand_query",
    "description": "Expand a user query into Hebrew + English variants for bilingual lecture search.",
    "input_schema": {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Up to 4 search variants. Include the original query, "
                    "an English-equivalent if the input is Hebrew, a Hebrew-equivalent "
                    "(or Hebrew transliteration) if the input is English, and one "
                    "broader paraphrase if relevant. No duplicates."
                ),
                "maxItems": QUERY_EXPANSION_MAX_VARIANTS,
            },
        },
        "required": ["variants"],
    },
}

QUERY_EXPANSION_PROMPT = """אתה מקבל מונח טכני שסטודנט הקליד בחיפוש על הרצאות AI Engineering.
ההרצאות מועברות בעברית עם מונחים טכניים באנגלית. הסליידים באנגלית.

החזר רשימה של עד 4 וריאנטים לחיפוש שמכסות:
1. המילה המקורית כפי שהוקלדה.
2. המקבילה בשפה השנייה: אם הקלט בעברית — האנגלית הסטנדרטית; אם הקלט באנגלית — תעתיק עברי שסביר שמרצה ישראלי ישתמש בו.
3. תעתיק או מילה חלופית נפוצה (למשל "אנקודר" → "encoder" וגם "bi-encoder", "agent" → "סוכן" וגם "איג'נט").
4. הרחבה רחבה יותר אם רלוונטי (למשל "RAG" → "retrieval augmented generation").

עקרונות:
- אל תוסיף וריאנטים שאינם רלוונטיים. עדיף 2 איכותיים מ-4 חלשים.
- שמור על קצרות: כל וריאנט מילה אחת או שתיים, לא משפט.
- ללא כפילויות.

קרא לכלי expand_query עם רשימת הוריאנטים."""

# Tool schema forces the LLM to emit valid JSON with the right fields.
ANCHOR_TOOL = {
    "name": "pick_anchor",
    "description": "Pick the single best teaching-moment anchor from the candidate list.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chosen": {
                "type": "integer",
                "description": "The 1-based index of the chosen candidate from the list.",
            },
            "reason": {
                "type": "string",
                "description": "Short Hebrew explanation of why this candidate was chosen.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": ["chosen", "reason", "confidence"],
    },
}


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


@dataclass
class AnchorPick:
    """LLM's choice for the 'teaching moment' anchor."""
    result:     SearchResult     # the chosen result
    reason:     str              # short Hebrew reason from the LLM
    confidence: str              # "high" | "medium" | "low"
    others:     list[SearchResult]  # the remaining candidates the LLM saw
    answer:     str = ""         # generated Hebrew RAG answer with [HH:MM:SS] citations


ANCHOR_SYSTEM_PROMPT = """אתה עוזר לזהות את הרגע המדויק בהרצאה שבו המרצה **מתחיל להסביר** נושא מסוים.

תקבל שאילתת חיפוש של סטודנט, ורשימה ממוספרת של קטעים. כל קטע מסומן בזמן (שעה:דקה:שניה), מקור (asr=דיבור של המרצה, ocr=טקסט מהסליידים), וטקסט.

**עיקרון מרכזי**: סליידים (ocr) הם **עדות שהנושא נלמד באזור הזמן הזה**, אבל הרגע הנכון לקפוץ אליו הוא **תחילת הדיבור של המרצה על הנושא** (asr), לא הסליד עצמו. סליד עולה אחרי שהמרצה כבר אומר "עכשיו נדבר על X" — אז קפיצה לזמן הסליד מאחרת לסטודנט את ההתחלה של ההסבר.

תהליך בחירה:

1. **זהה אשכול של רלוונטיות**: רצף של קטעים סמוכים בזמן (פערים עד 5-10 דקות) שכולם נוגעים לנושא. אזכור בודד ומבודד הוא אות חלש; אשכול הוא אות חזק.
2. **בחר את הקטע ה-asr המוקדם ביותר באשכול הרלוונטי ביותר.** זה המקום שבו המרצה אומר את הנושא לראשונה בהקשר של ההסבר. עדף קטעי asr שמזכירים מילות מפתח מהשאילתה, גם אם ציון retrieval שלהם נמוך יותר מסליד.
3. **בחר ocr רק כברירת מחדל**: אם אין באשכול שום קטע asr שמדבר על הנושא בצורה ברורה, וקטעי ה-asr באשכול הם רעש ("כן בסדר", "אז תראו"), אז בחר את הסליד שמציג את הנושא. במקרה כזה ציין בנימוק שלא נמצאה התחלה ברורה בדיבור.
4. **התעלם מאזכורים בודדים מחוץ לאשכול** — אם יש 5 אזכורים בשעה 2:00-3:00 ואזכור אחד מבודד בשעה 0:25, האזכור המבודד הוא כנראה הערה אגב ולא תחילת הסבר.

קרא לכלי pick_anchor עם:
- chosen: מספר הקטע שבחרת
- reason: משפט קצר בעברית מדוע בחרת בו, כולל הזכרה אם זה רגע דיבור או סליד
- confidence:
  • high — יש אשכול ברור + נמצאה תחילת דיבור ברורה (asr) על הנושא
  • medium — יש אשכול אבל ה-asr באזור רועש או חלקי, או שבחרת בסליד כברירת מחדל
  • low — אין אשכול ממשי, רק אזכורים מפוזרים. ייתכן שהנושא לא באמת נלמד בהרצאה."""


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

        # Anthropic client is lazy — only created if find_anchor is called
        self._anthropic: Anthropic | None = None
        self._expansion_cache: dict[str, list[str]] = {}

        print("[search] Ready.")

    def _get_anthropic(self) -> Anthropic | None:
        if self._anthropic is not None:
            return self._anthropic
        if Anthropic is None:
            print("[search] anthropic SDK not installed — anchor picker disabled.")
            return None
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[search] ANTHROPIC_API_KEY not set — anchor picker disabled.")
            return None
        self._anthropic = Anthropic()
        return self._anthropic

    def _expand_query(self, query: str) -> list[str]:
        """Expand a query into bilingual variants via Claude Haiku. Falls back
        to the raw query if the LLM is unavailable or the call fails. Cached
        in-process per query string."""
        q = query.strip()
        if not q:
            return [q]
        if q in self._expansion_cache:
            return self._expansion_cache[q]

        client = self._get_anthropic()
        if client is None:
            self._expansion_cache[q] = [q]
            return [q]

        try:
            resp = client.messages.create(
                model=ANCHOR_MODEL,
                max_tokens=200,
                system=[
                    {
                        "type": "text",
                        "text": QUERY_EXPANSION_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[QUERY_EXPANSION_TOOL],
                tool_choice={"type": "tool", "name": "expand_query"},
                messages=[{"role": "user", "content": q}],
            )
            tool_use = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
            if tool_use is None:
                raise ValueError("LLM did not call expand_query")
            variants = list(tool_use.input.get("variants") or [])
            # Always include the original; dedupe; cap.
            seen: set[str] = set()
            ordered: list[str] = []
            for v in [q] + variants:
                vv = v.strip()
                key = vv.lower()
                if vv and key not in seen:
                    seen.add(key)
                    ordered.append(vv)
                if len(ordered) >= QUERY_EXPANSION_MAX_VARIANTS:
                    break
            print(f"[search] expanded {q!r} → {ordered}")
            self._expansion_cache[q] = ordered
            return ordered
        except Exception as e:
            print(f"[search] expand_query failed: {e}")
            self._expansion_cache[q] = [q]
            return [q]

    # ── Query encoders ────────────────────────────────────────────

    def encode_text(self, query: str) -> list[float]:
        # e5-large requires "query: " prefix on user queries (paired with "passage: " at index time).
        return self.text_model.encode(
            [f"query: {query}"], normalize_embeddings=True
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

    def _search_text(
        self,
        query_vec: list[float],
        video_filter: str | None,
        n: int,
        source_filter: str | None = None,
    ) -> list[dict]:
        clauses = []
        if video_filter:
            clauses.append({"video_id": video_filter})
        if source_filter:
            clauses.append({"source": source_filter})
        if len(clauses) == 0:
            where = None
        elif len(clauses) == 1:
            where = clauses[0]
        else:
            where = {"$and": clauses}
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

    # ── Anchor picker (LLM as re-ranker) ──────────────────────────

    def _anchor_candidates(
        self,
        query: str,
        video_filter: str | None = None,
        n_asr: int = 10,
        n_ocr: int = 8,
    ) -> list[SearchResult]:
        """Retrieve a balanced mix of ASR and OCR candidates, querying each
        bilingual variant of the input separately and deduping the union.

        The keys make sense: Hebrew speech matches Hebrew variants well; English
        slide text matches English variants well; we let both have a fair shot
        rather than relying on cross-lingual cosine alone.
        """
        variants = self._expand_query(query)

        # Per-variant retrieval, then union-dedupe by (video_id, start, source).
        # Keep the highest score across variants for each (this is the chunk's
        # best signal that *any* of our variants found it useful).
        merged: dict[tuple[str, float, str], dict] = {}
        for v in variants:
            text_vec = self.encode_text(v)
            for h in (self._search_text(text_vec, video_filter, n_asr, source_filter="asr")
                      + self._search_text(text_vec, video_filter, n_ocr, source_filter="ocr")):
                key = (h["video_id"], round(h["start"], 2), h["source"])
                prev = merged.get(key)
                if prev is None or h["score"] > prev["score"]:
                    merged[key] = h

        results: list[SearchResult] = []
        for h in merged.values():
            results.append(SearchResult(
                score    = h["score"],
                video_id = h["video_id"],
                start    = h["start"],
                end      = h["end"],
                text     = h["text"],
                source   = h["source"],
            ))
        results.sort(key=lambda r: -r.score)
        return results

    @staticmethod
    def _density_boosted(candidates: list[SearchResult]) -> list[SearchResult]:
        """Return a new list where each candidate's score is multiplied by
        (1 + sum of exp(-|Δt| / DENSITY_DECAY_SEC) over the other candidates
        in the same video). Rewards results that cluster in time."""
        import math
        boosted: list[SearchResult] = []
        for c in candidates:
            bonus = 0.0
            for o in candidates:
                if o is c or o.video_id != c.video_id:
                    continue
                dt = abs(o.start - c.start)
                bonus += math.exp(-dt / DENSITY_DECAY_SEC)
            boosted.append(SearchResult(
                score    = c.score * (1.0 + bonus),
                video_id = c.video_id,
                start    = c.start,
                end      = c.end,
                text     = c.text,
                source   = c.source,
            ))
        boosted.sort(key=lambda r: -r.score)
        return boosted

    def _generate_answer(
        self,
        query: str,
        chunks: list[SearchResult],
        client: Anthropic,
    ) -> str:
        """Generate a short Hebrew answer grounded in the retrieved chunks.
        Inline citations use [HH:MM:SS] format so the UI can hyperlink them
        to video seeks."""
        if not chunks or client is None:
            return ""

        lines = []
        for c in chunks:
            ts = _hms(c.start)
            snippet = c.text.replace("\n", " ").strip()[:400]
            lines.append(f"[{ts}] ({c.video_id}, {c.source}) {snippet}")
        evidence = "\n".join(lines)
        user_msg = f"שאילתת חיפוש: {query}\n\nקטעים זמינים:\n{evidence}"

        try:
            resp = client.messages.create(
                model=ANCHOR_MODEL,
                max_tokens=ANSWER_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": ANSWER_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            text_block = next((b for b in resp.content if getattr(b, "type", None) == "text"), None)
            return text_block.text.strip() if text_block else ""
        except Exception as e:
            print(f"[search] _generate_answer failed: {e}")
            return ""

    def find_anchor(
        self,
        query: str,
        top_k_display: int = 5,
        video_filter: str | None = None,
    ) -> AnchorPick | None:
        """
        Retrieve more candidates than usual, then ask an LLM to pick the single
        'teaching moment' anchor. Falls back to None (the caller should display
        plain retrieval results) if the LLM is unavailable or the call fails.
        """
        client = self._get_anthropic()
        candidates = self._anchor_candidates(query, video_filter=video_filter)
        if not candidates:
            return None
        if client is None:
            # No LLM — return the top retrieval result as a "low confidence" pick
            remaining = candidates[1:]
            others = self._density_boosted(remaining)[:top_k_display]
            return AnchorPick(
                result=candidates[0],
                reason="(LLM disabled; showing top retrieval result.)",
                confidence="low",
                others=others,
            )

        # Format candidates for the LLM
        lines = []
        for i, c in enumerate(candidates, start=1):
            ts = _hms(c.start)
            snippet = c.text.replace("\n", " ").strip()[:300]
            lines.append(f"[{i}] {ts} ({c.source}) {snippet}")
        candidates_block = "\n".join(lines)
        user_msg = f"שאילתת חיפוש: {query}\n\nקטעים:\n{candidates_block}"

        try:
            resp = client.messages.create(
                model=ANCHOR_MODEL,
                max_tokens=ANCHOR_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": ANCHOR_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[ANCHOR_TOOL],
                tool_choice={"type": "tool", "name": "pick_anchor"},
                messages=[{"role": "user", "content": user_msg}],
            )
            tool_use = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
            if tool_use is None:
                raise ValueError("LLM did not call the pick_anchor tool")
            pick = tool_use.input
            idx = int(pick["chosen"]) - 1
            if not (0 <= idx < len(candidates)):
                raise ValueError(f"LLM returned out-of-range index {idx+1}")
            chosen = candidates[idx]
            # Re-rank the remaining candidates by density-boosted score so
            # clusters of nearby moments float up.
            remaining = [c for i, c in enumerate(candidates) if i != idx]
            others = self._density_boosted(remaining)[:top_k_display]
            # Generate the Hebrew RAG answer from the chosen + top alternatives.
            answer = self._generate_answer(query, [chosen] + others, client)
            return AnchorPick(
                result=chosen,
                reason=(pick.get("reason") or "").strip(),
                confidence=pick.get("confidence", "medium"),
                others=others,
                answer=answer,
            )
        except Exception as e:
            print(f"[search] find_anchor failed: {e}")
            remaining = candidates[1:]
            others = self._density_boosted(remaining)[:top_k_display]
            return AnchorPick(
                result=candidates[0],
                reason=f"(LLM error: {e}; showing top retrieval result.)",
                confidence="low",
                others=others,
            )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _parse_json_lenient(text: str) -> dict:
    """Parse a JSON object that may be wrapped in markdown fences or prose."""
    text = text.strip()
    if text.startswith("```"):
        # strip ``` or ```json … ```
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Find the first '{' and last '}' to extract the JSON body
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first:last+1]
    return json.loads(text)


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
    parser.add_argument("--anchor", action="store_true",
                        help="Use the LLM anchor picker (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    engine = SearchEngine(args.db)

    if args.anchor:
        pick = engine.find_anchor(args.query, top_k_display=args.top_k, video_filter=args.video)
        if pick is None:
            print("No results.")
            return
        r = pick.result
        ts = _hms(r.start)
        print(f"\nAnchor for: \"{args.query}\"")
        print(f"  Confidence: {pick.confidence}")
        print(f"  Time:       {ts}  ({r.video_id}, {r.source})")
        print(f"  Reason:     {pick.reason}")
        print(f"  Snippet:    {r.text[:200].replace(chr(10), ' ')}")
        if pick.others:
            print(f"\nOther moments:")
            for i, o in enumerate(pick.others, 1):
                ts = _hms(o.start)
                snip = o.text[:55].replace("\n", " ")
                print(f"  {i}. {ts}  ({o.source})  {snip}")
        return

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
