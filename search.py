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
import re
import urllib.error
import urllib.request
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

LLM_PROVIDER          = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
ANCHOR_MODEL          = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
OLLAMA_MODEL          = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL            = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
ANCHOR_CANDIDATE_K    = 15   # how many results to give the LLM to choose from
ANCHOR_MAX_TOKENS     = 400
DENSITY_DECAY_SEC     = 300.0  # exp decay scale for the "neighbors nearby" boost
DENSITY_WINDOW_SEC    = 600.0  # only nearby candidates can support one another
DENSITY_RELATIVE_FLOOR = 0.88  # ignore weak hits below 88% of the best candidate
DENSITY_NEIGHBOR_WEIGHT = 0.35 # keep density as a relevance tie-breaker
DENSITY_MAX_BONUS     = 0.65   # cap total score multiplier at 1.65x

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
    vector_score:  float = field(default=0.0, compare=False)
    density_bonus: float = field(default=0.0, compare=False)
    match_variant: str   = field(default="", compare=False)
    source_rank:   int   = field(default=0, compare=False)
    density_support: str = field(default="", compare=False)


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

        provider = LLM_PROVIDER if LLM_PROVIDER in {"anthropic", "ollama", "none"} else "none"
        if provider != LLM_PROVIDER:
            print(f"[search] Unknown LLM_PROVIDER={LLM_PROVIDER!r}; using 'none'.")
        self.llm_provider = provider
        self.ollama_model = OLLAMA_MODEL
        print(f"[search] LLM provider: {self.llm_provider}")
        print(f"[search] Ollama model: {self.ollama_model}")

        # Anthropic client is lazy — only created if the provider needs it.
        self._anthropic: Anthropic | None = None
        self._expansion_cache: dict[str, list[str]] = {}

        print("[search] Ready.")

    def set_llm_provider(self, provider: str) -> None:
        provider = (provider or "none").strip().lower()
        if provider not in {"anthropic", "ollama", "none"}:
            provider = "none"
        if provider != self.llm_provider:
            print(f"[search] switching LLM provider: {self.llm_provider} → {provider}")
            self.llm_provider = provider
            self._expansion_cache.clear()

    def set_ollama_model(self, model: str) -> None:
        model = (model or OLLAMA_MODEL).strip()
        if model and model != self.ollama_model:
            print(f"[search] switching Ollama model: {self.ollama_model} → {model}")
            self.ollama_model = model
            self._expansion_cache.clear()

    def _get_anthropic(self) -> Anthropic | None:
        if self.llm_provider != "anthropic":
            return None
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

    def _ollama_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 400,
        json_mode: bool = False,
    ) -> str:
        """Call a local Ollama chat model. Raises on connection/model errors."""
        payload: dict = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Ollama HTTP {e.code} at {OLLAMA_URL} for model "
                f"{self.ollama_model!r}: {detail}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {OLLAMA_URL}. Is `ollama serve` running? ({e.reason})"
            ) from e
        content = (body.get("message") or {}).get("content", "").strip()
        if not content:
            done_reason = body.get("done_reason") or body.get("error") or "unknown"
            raise RuntimeError(
                f"Ollama returned an empty response for model {self.ollama_model!r} "
                f"(done_reason={done_reason})"
            )
        return content

    def _ollama_json(self, messages: list[dict[str, str]], max_tokens: int = 400) -> dict:
        """Ask Ollama for JSON, retrying once without JSON mode for models that stumble."""
        first_error: Exception | None = None
        try:
            return _parse_json_lenient(
                self._ollama_chat(messages, max_tokens=max_tokens, json_mode=True)
            )
        except Exception as e:
            first_error = e
            print(f"[search] ollama JSON-mode response failed: {e}; retrying without JSON mode.")

        try:
            return _parse_json_lenient(
                self._ollama_chat(messages, max_tokens=max_tokens, json_mode=False)
            )
        except Exception as e:
            raise RuntimeError(f"could not parse Ollama JSON response ({first_error}; retry: {e})") from e

    @staticmethod
    def _ordered_variants(query: str, variants: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for v in [query] + variants:
            vv = str(v).strip()
            key = vv.lower()
            if vv and key not in seen:
                seen.add(key)
                ordered.append(vv)
            if len(ordered) >= QUERY_EXPANSION_MAX_VARIANTS:
                break
        return ordered

    def _expand_query(self, query: str) -> list[str]:
        """Expand a query into bilingual variants via the configured LLM. Falls back
        to the raw query if the LLM is unavailable or the call fails. Cached
        in-process per query string."""
        q = query.strip()
        if not q:
            return [q]
        if q in self._expansion_cache:
            return self._expansion_cache[q]

        if self.llm_provider == "none":
            self._expansion_cache[q] = [q]
            return [q]

        if self.llm_provider == "ollama":
            try:
                data = self._ollama_json(
                    [
                        {
                            "role": "system",
                            "content": (
                                QUERY_EXPANSION_PROMPT
                                + '\n\nReturn only JSON in this shape: {"variants": ["..."]}.'
                            ),
                        },
                        {"role": "user", "content": q},
                    ],
                    max_tokens=200,
                )
                ordered = self._ordered_variants(q, list(data.get("variants") or []))
                print(f"[search] expanded {q!r} via ollama → {ordered}")
                self._expansion_cache[q] = ordered
                return ordered
            except Exception as e:
                print(f"[search] ollama expand_query failed: {e}")
                self._expansion_cache[q] = [q]
                return [q]

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
            ordered = self._ordered_variants(q, variants)
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
        for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), start=1):
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
                "rank":     rank,
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
        for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), start=1):
            score = 1.0 - dist / 2.0
            ts = meta["timestamp"]
            hits.append({
                "score":    score,
                "video_id": meta["video_id"],
                "start":    ts,
                "end":      ts + 1.0,
                "text":     f"[visual match @ {ts:.1f}s]",
                "source":   "visual",
                "rank":     rank,
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
                vector_score = h["score"],
            ))

        for h in visual_hits:
            combined.append(SearchResult(
                score    = h["score"] * VISUAL_WEIGHT,
                video_id = h["video_id"],
                start    = h["start"],
                end      = h["end"],
                text     = h["text"],
                source   = h["source"],
                vector_score = h["score"],
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
                    h["match_variant"] = v
                    h["source_rank"] = h.get("rank", 0)
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
                vector_score = h["score"],
                match_variant = h.get("match_variant", ""),
                source_rank = h.get("source_rank", 0),
            ))
        results.sort(key=lambda r: -r.score)
        return results

    @staticmethod
    def _density_boosted(candidates: list[SearchResult]) -> list[SearchResult]:
        """Boost relevant candidates that are supported by nearby relevant hits.

        Density is intentionally conservative: weak candidates cannot become
        strong only because many other weak candidates are nearby. This keeps
        clustering as a tie-breaker among plausible matches rather than the main
        relevance signal.
        """
        import math
        if not candidates:
            return []

        max_score = max(c.score for c in candidates)
        relevance_floor = max_score * DENSITY_RELATIVE_FLOOR
        denom = max(max_score - relevance_floor, 1e-9)

        boosted: list[SearchResult] = []
        for c in candidates:
            bonus = 0.0
            support: list[str] = []
            if c.score >= relevance_floor:
                for o in candidates:
                    if (
                        o is c
                        or o.video_id != c.video_id
                        or o.score < relevance_floor
                    ):
                        continue
                    dt = abs(o.start - c.start)
                    if dt > DENSITY_WINDOW_SEC:
                        continue
                    neighbor_strength = (o.score - relevance_floor) / denom
                    source_diversity = 1.15 if o.source != c.source else 1.0
                    bonus += (
                        neighbor_strength
                        * source_diversity
                        * math.exp(-dt / DENSITY_DECAY_SEC)
                    )
                    exact_terms = _variant_terms_in_text(o.match_variant, o.text)
                    exact = ",".join(exact_terms) if exact_terms else "semantic-only"
                    support.append(
                        f"{_hms(o.start)} {o.source} "
                        f"variant={o.match_variant or '-'} exact={exact} "
                        f"vec={(o.vector_score or o.score):.3f}"
                    )
            bonus = min(DENSITY_MAX_BONUS, bonus * DENSITY_NEIGHBOR_WEIGHT)
            boosted.append(SearchResult(
                score    = c.score * (1.0 + bonus),
                video_id = c.video_id,
                start    = c.start,
                end      = c.end,
                text     = c.text,
                source   = c.source,
                vector_score = c.vector_score or c.score,
                density_bonus = bonus,
                match_variant = c.match_variant,
                source_rank = c.source_rank,
                density_support = "; ".join(support) if support else "-",
            ))
        boosted.sort(key=lambda r: -r.score)
        return boosted

    def _generate_answer(
        self,
        query: str,
        chunks: list[SearchResult],
        client: Anthropic | None = None,
    ) -> str:
        """Generate a short Hebrew answer grounded in the retrieved chunks.
        Inline citations use [HH:MM:SS] format so the UI can hyperlink them
        to video seeks."""
        if not chunks or self.llm_provider == "none":
            return ""

        lines = []
        for c in chunks:
            ts = _hms(c.start)
            snippet = c.text.replace("\n", " ").strip()[:400]
            lines.append(f"[{ts}] ({c.video_id}, {c.source}) {snippet}")
        evidence = "\n".join(lines)
        user_msg = f"שאילתת חיפוש: {query}\n\nקטעים זמינים:\n{evidence}"

        if self.llm_provider == "ollama":
            try:
                return self._ollama_chat(
                    [
                        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=ANSWER_MAX_TOKENS,
                    json_mode=False,
                )
            except Exception as e:
                print(f"[search] ollama _generate_answer failed: {e}")
                return ""

        if client is None:
            client = self._get_anthropic()
        if client is None:
            return ""

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

    def _deterministic_anchor(
        self,
        candidates: list[SearchResult],
        top_k_display: int,
        reason: str,
        confidence: str = "medium",
    ) -> AnchorPick:
        return AnchorPick(
            result=candidates[0],
            reason=reason,
            confidence=confidence,
            others=candidates[1:top_k_display + 1],
        )

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
        raw_candidates = self._anchor_candidates(query, video_filter=video_filter)
        if not raw_candidates:
            return None
        candidates = self._density_boosted(raw_candidates)[:ANCHOR_CANDIDATE_K]

        if self.llm_provider == "none":
            return self._deterministic_anchor(
                candidates,
                top_k_display,
                reason="(LLM_PROVIDER=none; showing top density-ranked retrieval result.)",
            )

        # Format candidates for the LLM
        lines = []
        for i, c in enumerate(candidates, start=1):
            ts = _hms(c.start)
            snippet = c.text.replace("\n", " ").strip()[:300]
            lines.append(f"[{i}] {c.video_id} {ts} ({c.source}, score={c.score:.3f}) {snippet}")
        candidates_block = "\n".join(lines)
        user_msg = f"שאילתת חיפוש: {query}\n\nקטעים:\n{candidates_block}"

        if self.llm_provider == "ollama":
            try:
                pick = self._ollama_json(
                    [
                        {
                            "role": "system",
                            "content": (
                                ANCHOR_SYSTEM_PROMPT
                                + '\n\nReturn only JSON in this shape: '
                                '{"chosen": 1, "reason": "...", "confidence": "high|medium|low"}.'
                            ),
                        },
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=ANCHOR_MAX_TOKENS,
                )
                idx = int(pick["chosen"]) - 1
                if not (0 <= idx < len(candidates)):
                    raise ValueError(f"Ollama returned out-of-range index {idx+1}")
                confidence = str(pick.get("confidence", "medium")).strip().lower()
                if confidence not in {"high", "medium", "low"}:
                    confidence = "medium"
                chosen = candidates[idx]
                remaining = [c for i, c in enumerate(candidates) if i != idx]
                others = remaining[:top_k_display]
                answer = self._generate_answer(query, [chosen] + others)
                return AnchorPick(
                    result=chosen,
                    reason=(pick.get("reason") or "").strip(),
                    confidence=confidence,
                    others=others,
                    answer=answer,
                )
            except Exception as e:
                print(f"[search] ollama find_anchor failed: {e}")
                return self._deterministic_anchor(
                    candidates,
                    top_k_display,
                    reason=(
                        "(Ollama could not return a valid JSON anchor; "
                        "showing top density-ranked retrieval result.)"
                    ),
                    confidence="medium",
                )

        client = self._get_anthropic()
        if client is None:
            return self._deterministic_anchor(
                candidates,
                top_k_display,
                reason="(Anthropic unavailable; showing top density-ranked retrieval result.)",
            )

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
            remaining = [c for i, c in enumerate(candidates) if i != idx]
            others = remaining[:top_k_display]
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
            return self._deterministic_anchor(
                candidates,
                top_k_display,
                reason=f"(Anthropic error: {e}; showing top density-ranked retrieval result.)",
                confidence="medium",
            )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _hms(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _variant_terms_in_text(variant: str, text: str) -> list[str]:
    """Return literal variant terms found in a retrieved chunk."""
    found: list[str] = []
    seen: set[str] = set()
    for term in re.findall(r"[\w'-]+", variant, flags=re.UNICODE):
        if len(term) < 3:
            continue
        flags = re.IGNORECASE if term.isascii() else 0
        if re.search(re.escape(term), text, flags=flags):
            key = term.lower() if term.isascii() else term
            if key not in seen:
                seen.add(key)
                found.append(term)
    whole = variant.strip()
    if len(whole) >= 3:
        flags = re.IGNORECASE if whole.isascii() else 0
        key = whole.lower() if whole.isascii() else whole
        if key not in seen and re.search(re.escape(whole), text, flags=flags):
            found.insert(0, whole)
    return found


def _parse_json_lenient(text: str) -> dict:
    """Parse a JSON object that may be wrapped in markdown fences or prose."""
    text = text.strip()
    if not text:
        raise ValueError("empty JSON response")
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
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        preview = text.replace("\n", " ")[:160]
        raise ValueError(f"non-JSON response: {preview!r}") from e


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
