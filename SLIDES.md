---
marp: true
theme: default
paginate: true
size: 16:9
header: 'ShowMe — Multimodal Semantic Search over Lecture Videos'
footer: 'AI Engineering Course Project · 2026'
style: |
  section { font-size: 26px; }
  h1 { color: #1a73e8; }
  h2 { color: #202124; border-bottom: 2px solid #1a73e8; padding-bottom: 4px; }
  code { background: #f1f3f4; padding: 2px 6px; border-radius: 4px; }
  pre { background: #f8f9fa; border-left: 3px solid #1a73e8; }
  table { font-size: 22px; }
  blockquote { border-left: 4px solid #fbbc04; background: #fffbe6; padding: 8px 16px; }
---

<!-- _class: lead -->

# ShowMe

## Multimodal Semantic Search over Lecture Videos

**Find the exact moment in a 4-hour lecture where a topic is taught,**
**in Hebrew, English, or transliteration.**

*AI Engineering Course Project — Module 4 (RAG and Autonomous Agents)*

---

## The Problem

You're a student in a 210-hour course. The lecturer demonstrated `agents`, `RAG`, `embeddings` across many 4-hour lectures.

You remember the topic. You don't remember **when** or **which lecture**.

**Existing options:**
- Re-watch the whole lecture (4 hours)
- Scrub through hoping to recognize a slide (slow, frustrating)
- Search the Hebrew transcript (it's auto-generated and noisy)

**Goal:** Type a query, jump to the lecturer **starting to teach** that topic.

---

## Why "starting to teach" is the key insight

A topic can appear in three different ways in a lecture:

| Type | Example | Useful? |
|---|---|---|
| Stray mention | *"like agents we'll cover later"* | ❌ |
| Q&A reference | *"good question, that's related to RAG"* | ❌ |
| **Teaching block** | *"now let's talk about RAG. The idea is..."* | ✅ |

Standard semantic search ranks **stray mentions equally** with teaching blocks because the vocabulary matches. We need a system that recognizes *teaching density*, not just keyword presence.

---

## System Overview

```
┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────┐
│  ingest.py   │ → │   index.py    │ → │  search.py   │ → │   app.py     │
│ video → JSON │   │ JSON → vector │   │ query → top  │   │ Gradio UI    │
│ (multimodal) │   │ DB (Chroma)   │   │ K + anchor   │   │ + video seek │
└──────────────┘   └───────────────┘   └──────────────┘   └──────────────┘
       │                  │                   │                  │
       ▼                  ▼                   ▼                  ▼
  data/index/raw/   data/index/chroma/   ChromaDB query     localhost:7860
  per-video JSON    text + visual         + Claude Haiku
                    collections           re-ranker
```

**Course mapping:** Module 4.1 advanced RAG architecture, Module 4.2 vector DB selection (Chroma), Module 3.3 LLM via API.

---

## Stage 1 — Ingest: Three modalities per lecture

For each ~4-hour MP4 we extract three parallel streams:

| Modality | Tool | What it captures | Output |
|---|---|---|---|
| **Speech** | Zoom VTT (fallback: Whisper) | Hebrew narration with English code terms | `asr.json` — `[{start, end, text}]` |
| **Slides** | EasyOCR | English slide text every 5 min | `ocr.json` — `[{timestamp, text}]` |
| **Visual** | CLIP ViT-B/32 | 512-dim frame embedding | `clip.json` — `[{timestamp, embedding}]` |

**Key engineering choices:**
- **Zoom VTT over Whisper**: 10× faster, equivalent Hebrew quality. Falls back to Whisper if VTT missing.
- **Seek-based frame sampling** (`cap.set(POS_MSEC, ...)`): one frame every 5 min, ~0.5s per frame instead of ~20s.
- **Slide-change detection** to skip OCR on duplicate frames (saves ~70% of OCR cost).

---

## Stage 2 — Index: Multilingual chunking + embedding

**Chunking strategy (`chunk_asr`):**
- Merge consecutive VTT cues into chunks bounded by `CHUNK_WINDOW_SEC = 60s` and `CHUNK_MAX_CHARS = 500`.
- A chunk covers a *thought*, not a *sentence fragment*.

**Embedding model: `intfloat/multilingual-e5-large`** (1024-dim)
- Hebrew speech + English slides forced us off `all-MiniLM-L6-v2` (English-only).
- e5 requires `"passage: "` prefix on indexed docs, `"query: "` on queries. Omitting these halves retrieval quality silently.

**Vector store: ChromaDB**
- Two collections: `text_chunks` (ASR+OCR mixed, source as metadata) + `visual_frames` (CLIP).
- Metadata filtering via `where` clauses (course Section 4.2).

> *Course concept: "Choosing and optimizing document chunking strategy" + "Multilingual embedding model selection" — both Module 4.1.*

---

## Stage 3 — Retrieval: Five steps, each principled

The interesting part. A query goes through five distinct stages before the user sees a result.

```
        Query (any language)
                ↓
   ① Bilingual expansion (Claude Haiku)
                ↓
   ② Source-balanced retrieval (Chroma)
                ↓
   ③ Density-boosted scoring
                ↓
   ④ LLM anchor picker (Claude Haiku, tool-use)
                ↓
       Anchor + 5 alternatives
```

Next slides cover each step.

---

## ① Bilingual Query Expansion

**Problem:** Hebrew speech says `"אנקודר"`. English slide says `"encoder"`. Cross-lingual cosine similarity is weaker than same-language. The user types one, we miss the other.

**Solution:** Before retrieval, expand the user query into ≤4 variants in both languages.

```
"אנקודר"  → ["אנקודר", "encoder", "Bi-Encoder", "אנקודרים"]
"agent"   → ["agent", "סוכן", "איג'נט", "AI agent"]
"RAG"     → ["RAG", "retrieval augmented generation", "ראג"]
```

**Implementation:** Claude Haiku 4.5 with structured tool-use (`expand_query` tool with `variants` schema) — guarantees valid JSON output, can't fail to parse. Expansions cached per-process.

**Cost:** ~$0.0005/query, ~200ms.

> *Course concept: "Query optimization for RAG: multi-query retrieval, query rewriting" — Module 4.1.*

---

## ② Source-Balanced Retrieval

**Problem:** English slides have clean OCR text. Hebrew VTT speech is noisy. Without forcing balance, **slides outrank speech for nearly every query**, so the user never sees the actual teaching moments.

**Solution:** For each expanded variant, query `text_chunks` **twice with metadata filters**:
- Top-10 chunks where `source = "asr"` (lecturer's words)
- Top-8 chunks where `source = "ocr"` (slide text)

Union across all variants, dedupe by `(video_id, start, source)`, keep highest score.
**Result:** ~30–60 candidate chunks balanced across modalities.

```python
hits_asr = collection.query(query_embeddings=[vec], where={"source": "asr"}, n_results=10)
hits_ocr = collection.query(query_embeddings=[vec], where={"source": "ocr"}, n_results=8)
```

> *Course concept: "Metadata filtering, hybrid search" — Module 4.2.*

---

## ③ Density-Boosted Scoring

**Problem the student articulated:** *"If there's one mention now but 20 mentions an hour later, the later one is the relevant one."*

A topic isn't taught at the moment of *one* hit. It's taught where hits **cluster** in time.

**Solution:** After retrieval, each candidate's similarity is multiplied by:

$$\text{boost}(c) = 1 + \sum_{o \in \text{others, same video}} \exp\left(-\frac{|t_o - t_c|}{\tau}\right) \quad \text{where } \tau = 300s$$

- A chunk with 5 neighbors within ±5 min gets a 2–4× boost.
- A lonely chunk gets no boost.
- Neighbors 30+ min away contribute nothing.

A **soft, principled re-ranker** with zero LLM cost.

> *Course concept: "Selecting and fine-tuning re-ranking algorithms" — Module 4.1.*

---

## ④ LLM Anchor Picker — *Not* Answer Generator

This is the novel architectural choice.

**Standard RAG:** Retrieved chunks → LLM → synthesized answer.
**ShowMe:** Retrieved chunks → LLM → **a single timestamp** to jump to.

| | Answer generator (standard RAG) | Anchor picker (ours) |
|---|---|---|
| Hallucination risk | High | **None** — only picks real timestamps |
| Truth source | The LLM's summary | **The lecturer's actual words** |
| Educational fit | Tells the student | **Shows the student** |
| Output tokens | ~200 (full answer) | ~30 (JSON: chosen, reason, confidence) |

The LLM gets the top-15 density-boosted candidates + a Hebrew system prompt encoding the heuristic.

---

## ④ Anchor Picker — Prompt Design

The system prompt (in Hebrew) tells Claude Haiku:

> "Find a **cluster** of nearby relevant candidates. Prefer the **earliest ASR chunk in the cluster** as the anchor (= the lecturer announcing the topic). Treat OCR slides as **evidence** that teaching happens nearby, but not as the seek target — slides appear *after* the speech that introduces them. Ignore isolated mentions far from any cluster."

**Output via tool-use:**
```json
{
  "chosen": 3,
  "reason": "המרצה מתחיל להסביר על אנקודר בקטע 3...",
  "confidence": "high" | "medium" | "low"
}
```

If `confidence = "low"`, the UI hides the anchor and falls back to plain top-K. **Honest about uncertainty.**

> *Course concept: "Re-ranking algorithms" + Module 3.3 "Function calling via APIs".*

---

## Stage 4 — UI: Jump-to-Moment

Gradio web interface:
- **Search box** → calls `engine.find_anchor()`
- **AI-picked panel** above results: confidence badge + Hebrew reason
- **Results table** shows the anchor (🌟 in source col) + 5 density-boosted alternatives
- **Video player** with custom JS: programmatically sets `currentTime = anchor.start` on click

**Tricky bit:** browsers block programmatic `video.play()` with sound (autoplay policy). After the seek, the user presses play themselves — sound works.

---

## End-to-End Example: query = `"אנקודר"`

```
"אנקודר" → Haiku expansion → ["אנקודר", "encoder", "Bi-Encoder", "אנקודרים"]

For each variant: top-10 ASR + top-8 OCR via Chroma  →  ~45 candidates

Density boost: cluster around 3:17-3:45 in zvi_lecture_2026-04-26 gets a 3× boost

Top-15 → Claude Haiku anchor picker

Result:
  Anchor: 3:17:32 ASR — "אני רוצה לדבר איתכם על משהו שנקרא בא עם גודר"
                       ("I want to tell you about something called bi-encoder")
  Confidence: high
  Reason: "המרצה מתחיל להסביר על בי-אנקודר באשכול ברור"
```

Total query latency: **~2 seconds**. Cost: **~$0.0015**.

---

## Engineering Choices & Trade-offs

**What we chose, and why** (defensible against questioning):

| Decision | Choice | Reason |
|---|---|---|
| Transcript source | Zoom VTT > Whisper | 10× faster, equivalent on Hebrew |
| Embedding model | e5-large multilingual | Hebrew + English corpus; mpnet-base too weak |
| Vector DB | Chroma (direct SDK, not LangChain) | Smaller dependency surface; course syllabus lists Chroma first |
| LLM | Anthropic Claude Haiku 4.5 | Cheap (~$0.001/call), fast, tool-use support |
| Re-ranker (heuristic) | Density boost (exp decay) | No LLM cost, encodes student's clustering intuition |
| Re-ranker (LLM) | Anchor picker (Haiku + tool-use) | Eliminates hallucination by design |
| Generation step | **Skipped** — anchor picker instead | Educational use case wants the lecture, not a paraphrase |

---

## Corpus Stats (this submission)

| | |
|---|---|
| Lectures ingested | **12** |
| Instructors | 7 Zvi, 5 Lev |
| Span | 2026-02-25 → 2026-05-10 |
| Total video duration | ~50 hours |
| Total MP4 size | ~11 GB (not in repo — `huji-executives.org` owns) |
| Text chunks indexed | **3,228** (~269 per lecture) |
| Visual frames | **608** |
| ChromaDB on disk | 70 MB |
| Cold start (first launch) | ~6 min (slow transformers import) |
| Warm query | ~2 sec |
| Per-query LLM cost | ~$0.0015 |

---

## Course Concept Coverage

Mapping every implementation decision to the syllabus, for the rubric:

| Course concept | Where in the code |
|---|---|
| Module 3.3 — LLM via API | `search.py:_expand_query`, `search.py:find_anchor` |
| Module 3.3 — Tool-use / structured output | `ANCHOR_TOOL`, `QUERY_EXPANSION_TOOL` |
| Module 3.3 — Image/video understanding | `ingest.py:embed_frame` (CLIP), `process_frames` (OCR) |
| Module 4.1 — Chunking strategy | `index.py:chunk_asr`, `CHUNK_WINDOW_SEC`, `CHUNK_MAX_CHARS` |
| Module 4.1 — Multilingual embedding | `index.py:TEXT_MODEL_NAME` (e5-large) |
| Module 4.1 — Multi-query / query rewriting | `_expand_query` |
| Module 4.1 — Re-ranking (heuristic) | `_density_boosted` |
| Module 4.1 — Re-ranking (LLM) | `find_anchor` |
| Module 4.2 — Chroma vector database | `index.py`, `search.py` |
| Module 4.2 — Metadata filtering | `_search_text` with `where={"source": ...}` |

---

## Honest Limitations

Things I'd want to address with more time / a v2:

1. **Anchor temporal resolution: ±5 min.** Frame sampling every 5 min misses brief slides. Mitigation: finer scan + change-triggered sampling (cost: ~2× ingest time per lecture).

2. **No formal evaluation harness.** Quality is judged qualitatively. Next step: label 20 queries with ground-truth timestamps, compute Hit@1 / MRR / time-delta.

3. **No hybrid lexical search.** Section 4.1 mentions BM-25 alongside semantic. We're dense-only. Easy addition with `BM25Retriever` + reciprocal rank fusion.

4. **No multi-stage retrieval.** At 30+ lectures, the picker's candidate pool dilutes across lectures. Future: pick lecture first, then moment within lecture.

5. **OCR has no Hebrew model.** EasyOCR doesn't support Hebrew. Hebrew slide text is dropped (most slides are English in this course). Switching to PaddleOCR closes this.

6. **No LangChain wrapper.** We use raw SDKs. Cleaner for debugging, less aligned with the syllabus's Chain/Retriever pattern.

---

## What I Got Wrong During the Build

Lessons that cost real time:

1. **e5-large needs `"passage: " / "query: "` prefixes.** Quietly halved retrieval quality before I noticed. Documented in code now.

2. **`chunk_asr` bug let chunks grow unbounded.** The window check compared `seg.start - chunk.start` (always increasing) instead of `seg.end - chunk.start` (true span). Found by inspecting chunk sizes after odd search results.

3. **torch 2.11 has an `inspect.getsource()` bug** that breaks transformers integrations. Pinned `torch<2.11` in `requirements.txt` with a comment.

4. **macOS swap pressure** caused 60-second file opens after long sessions of starting/killing Python processes. Symptom: silent hangs in `import`. Fix: reboot. Worth knowing for a graded demo day.

5. **Slide pickers overconfident before prompt tweak.** Initial prompt favored slides since slides have clean text. Rewrote to "slides as cluster evidence, ASR as anchor target." Major quality jump.

---

## Future Work

Ordered by grade impact / educational value, not engineering difficulty:

1. **Evaluation harness** with labeled ground truth — quantify the qualitative claims.
2. **Hybrid retrieval** (BM-25 + semantic) via `EnsembleRetriever`.
3. **Cross-encoder re-ranker** (`cross-encoder/mmarco-mMiniLMv2`) between balanced retrieval and the LLM picker.
4. **Hebrew OCR** via PaddleOCR or Tesseract.
5. **Two-stage retrieval** for the 30+ lecture corpus: lecture filter → moment within.
6. **Deployment** to Hugging Face Spaces (public URL, ChromaDB shipped with repo, videos via signed Drive URLs).
7. **LangChain wrapper** for the retrieval chain (syllabus alignment).
8. **Agent layer** (LangGraph) — agent decides between speech-only, slides-only, or hybrid based on query type.

---

## Demo Time

`python app.py --db data/index/chroma/ --videos data/videos/`
→ `http://127.0.0.1:7860`

**Suggested queries to demo:**

| Query | Expected anchor |
|---|---|
| `RAG` | Zvi explaining RAG architecture (cross-lecture cluster) |
| `סוכן` (Hebrew) | Zvi introducing agents |
| `agent` (English, same topic) | Same moment via bilingual expansion |
| `אנקודר` (transliteration) | Zvi's "let me tell you about bi-encoder" moment |
| `chunks` | Zvi explaining document chunking |

---

## Questions to Expect

**"Why not LangChain?"**
Smaller dependency surface, easier to debug, the abstraction was overhead for this size of project. The code IS the RAG pipeline. A v2 port to `LangChain Retriever` is on the future-work list.

**"Why Claude over OpenAI/Gemini?"**
Free initial credits, strong tool-use API, good Hebrew. Provider is one constant — `ANTHROPIC_API_KEY` swap to `OPENAI_API_KEY` is a 10-line change in `search.py`.

**"How do you know the anchor is right?"**
Qualitative validation across ~10 test queries. No formal eval yet — that's #1 on future work.

**"What's novel here?"**
The **LLM as anchor picker** pattern. Standard RAG uses the LLM to generate answers; we use it to pick a timestamp. Avoids hallucination, fits education better.

---

<!-- _class: lead -->

# Thanks

Repository: github.com/havia/ShowMe (PR #1 to smadarblumenfed/ShowMe)

Built on Module 4 concepts.
Hebrew + English bilingual.
12 lectures, 50 hours of video, ~2-second queries.

**Questions?**
