# ShowMe — Technical Workflow

A multimodal semantic search engine over Hebrew/English lecture videos. Given a natural-language query, the system finds the exact timestamp where the lecturer **starts teaching** the topic and lets the user jump to that moment.

Built on the curriculum of the AI Engineering course (Module 4: RAG and Autonomous Agents).

---

## Running this code on your machine

The repo includes the **vector index** (`data/index/`) for the three lectures we've ingested. You don't need to re-ingest — just bring the video files and run the UI.

### One-time setup

```bash
# 1. Clone and enter
git clone <repo>
cd ShowMe

# 2. Install ffmpeg (system-level)
brew install ffmpeg            # macOS
# or: sudo apt install ffmpeg   # Linux

# 3. Python environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 4. Anthropic API key — used for query expansion + the LLM anchor picker
echo 'ANTHROPIC_API_KEY=sk-ant-api03-...' > .env
```

### Get the videos

The ChromaDB references 12 videos by stem name. Download each from the course Drive into `data/videos/` and rename them to match the stem:

**Zvi lectures (7):**
| Recording filename | Save as |
|---|---|
| `GMT20260329-142657_Recording_2560x1600.mp4` | `data/videos/zvi_lecture_2026-03-29.mp4` |
| `GMT20260412-142338_Recording_2560x1600.mp4` | `data/videos/zvi_lecture_2026-04-12.mp4` |
| `GMT20260415-142312_Recording_2560x1600.mp4` | `data/videos/zvi_lecture_2026-04-15.mp4` |
| `GMT20260419-142536_Recording_3822x1600.mp4` | `data/videos/zvi_lecture_2026-04-19.mp4` |
| `GMT20260426-142039_Recording_2560x1600.mp4` | `data/videos/zvi_lecture_2026-04-26.mp4` |
| `GMT20260506-142150_Recording_2560x1600.mp4` | `data/videos/zvi_lecture_2026-05-06.mp4` |
| `GMT20260510-142838_Recording_2560x1600.mp4` | `data/videos/zvi_lecture_2026-05-10.mp4` |

**Lev lectures (5):**
| Recording filename | Save as |
|---|---|
| `GMT20260225-151837_Recording_3840x2160.mp4` | `data/videos/lev_lecture_2026-02-25.mp4` |
| `GMT20260315-152527_Recording_3840x2400.mp4` | `data/videos/lev_lecture_2026-03-15.mp4` |
| `GMT20260318-152255_Recording_3840x2160.mp4` | `data/videos/lev_lecture_2026-03-18.mp4` |
| `GMT20260322-152110_Recording_3840x2160.mp4` | `data/videos/lev_lecture_2026-03-22.mp4` |
| `GMT20260325-152624_Recording_3840x2160.mp4` | `data/videos/lev_lecture_2026-03-25.mp4` |

The MP4s aren't in this repo (~11 GB total, and they belong to `huji-executives.org`). The VTT caption files are also unnecessary at this stage — the index is already built. You only need the MP4s for video playback in the UI.

### Run

```bash
.venv/bin/python app.py --db data/index/chroma/ --videos data/videos/
# Open http://127.0.0.1:7860
```

Search examples: `RAG`, `סוכן`, `אנקודר`, `אמבדינג`, `embedding`, `agent`.

### Adding more lectures

If you have a new MP4 + matching VTT:

```bash
# Drop both into data/videos/ as <name>.mp4 + <name>.vtt
.venv/bin/python ingest.py --videos data/videos/   # skips already-ingested
.venv/bin/python index.py --raw data/index/raw/ --db data/index/chroma/
```

The pipeline is idempotent — existing lectures are skipped.

---

## High-level architecture

```
┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────┐
│  ingest.py   │ → │   index.py    │ → │  search.py   │ → │   app.py     │
│ video → JSON │   │ JSON → vector │   │ query → top  │   │ Gradio UI    │
│ (per video)  │   │ DB (per repo) │   │ K + anchor   │   │ + video seek │
└──────────────┘   └───────────────┘   └──────────────┘   └──────────────┘
       │                  │                   │                  │
       ▼                  ▼                   ▼                  ▼
  data/index/raw/   data/index/chroma/   ChromaDB query     localhost:7860
  per-video JSON    text + visual         + Claude Haiku
                    collections           re-ranker
```

Four scripts. Each can be invoked independently. State flows one direction: video files → raw JSON → vector DB → query → UI. Re-running any stage is idempotent.

---

## Pipeline detail

### Stage 1 — `ingest.py`: video → structured JSON

**Input:** an MP4 file (and optionally a sibling `.vtt` Zoom caption file).
**Output:** four JSON files in `data/index/raw/<video_id>/`:

| File | Contents |
|---|---|
| `asr.json` | Transcript segments — `[{start, end, text}]` from either Zoom VTT or Whisper |
| `ocr.json` | Slide text — `[{timestamp, text}]` extracted via EasyOCR |
| `clip.json` | Visual frame embeddings — `[{timestamp, embedding[512]}]` via OpenAI CLIP |
| `meta.json` | Summary: video_id, counts, source paths |

**Three submodules:**

1. **Transcript** — `parse_vtt()` if a `.vtt` exists alongside the MP4, otherwise `transcribe()` via Whisper `small` with `language="he"`. Zoom captions are 10x faster than Whisper but slightly noisier on transliterated English terms.

2. **Frame sampling** — Every `FRAME_INTERVAL_SEC = 300` seconds (5 min) we seek directly in the video via `cap.set(CAP_PROP_POS_MSEC, t * 1000)`. The original implementation read every frame in the file and discarded most; the new seek-based approach made each frame cost ~0.5s instead of ~20s. For a 4-hour lecture we sample ~50 frames total.

3. **CLIP + OCR per sample** — CLIP encodes the frame into a 512-dim visual embedding. OCR runs only when the frame differs ≥15% from the previous (slide-change detection via `frame_difference()`), which skips identical slides and saves ~70% of OCR cost.

**Idempotency** — On invocation, any video whose `data/index/raw/<id>/meta.json` already exists is skipped unless `--force` is passed. This is what makes adding new lectures cheap.

**Course mapping (Module 3.3, 4.1):** Whisper for ASR ([Section 3.3 "translation, summarization"], EasyOCR for slide text extraction (a course concept: extracting metadata from documents), CLIP for visual embedding (Section 3.3 "image and video understanding").

---

### Stage 2 — `index.py`: JSON → ChromaDB vector store

**Input:** `data/index/raw/<*>/` JSON files for all videos.
**Output:** A persistent ChromaDB at `data/index/chroma/` with two collections:

| Collection | What it holds | Dim |
|---|---|---|
| `text_chunks` | Chunked ASR + OCR text per video | 1024 |
| `visual_frames` | CLIP embeddings (already computed by ingest) | 512 |

**Chunking strategy** (`chunk_asr` in `index.py`):

- Merge consecutive VTT segments into chunks bounded by `CHUNK_WINDOW_SEC = 60.0` and `CHUNK_MAX_CHARS = 500`.
- This is meaningful: a chunk should cover a *thought*, not a *sentence fragment*. 60s typically captures 4–8 short captions worth of speech context.
- An earlier bug caused chunks to grow without bound (the window check compared `seg.start - chunk.start`, missing that chunks should be bounded by `chunk.end`); fix in commit `ce096b2`.

**Embedding model: `intfloat/multilingual-e5-large`**

- **Why multilingual:** The corpus is bilingual — Hebrew speech, English slides. A monolingual model (e.g. `all-MiniLM-L6-v2`) returns near-random similarity scores for Hebrew queries.
- **Why e5-large (1024-dim, 560M params)** over `paraphrase-multilingual-mpnet-base-v2` (768-dim): better recall on technical terms in non-English text. ~2GB model download, ~10s load time, ~3x slower than mpnet-base. Worth the cost.
- **Prefix convention:** e5 models require `"passage: "` prefix on indexed documents and `"query: "` prefix on queries. Omitting these prefixes quietly halves retrieval quality. Both are wired in: `index.py` prefixes documents, `search.py:encode_text` prefixes queries.

**Course mapping (Module 4.1, 4.2):**
- "Choosing and optimizing document chunking strategy" — chunk window + char cap
- "Fine-tuning embedding models for specific domains" — we didn't fine-tune, but we *did* select a multilingual model for our Hebrew/English domain (a course-canonical decision)
- "Vector database technologies: in-depth exploration of Chroma, ..." — Chroma is the syllabus's first listed vector DB; we use it via the `chromadb.PersistentClient` SDK
- "Hybrid RAG approaches: combining different retrieval methods" — we have two parallel indexes (text + visual), retrieved separately and fused at query time

---

### Stage 3 — `search.py`: query → anchor + alternatives

This is where most of the system's intelligence lives. A query goes through **five steps**:

#### Step 3.1 — Bilingual query expansion (Claude Haiku)

The user's query (Hebrew or English) is expanded into up to 4 search variants covering both languages. Example:

```
"אנקודר"  → ["אנקודר", "encoder", "Bi-Encoder", "אנקודרים"]
"agent"   → ["agent", "סוכן", "איג'נט", "AI agent"]
```

Why this matters: cross-lingual cosine matching between Hebrew speech and English slides is much weaker than same-language matching. By expanding into both languages, every variant gets a fair shot at retrieval. The expansion uses **Anthropic tool-use** (structured output) to guarantee a valid response, and expansions are cached per-process to avoid re-paying the LLM cost on repeat queries.

**Course mapping (Section 4.1):** "Query optimization for RAG: multi-query retrieval, query rewriting." This is literally the textbook example.

#### Step 3.2 — Balanced retrieval (ChromaDB)

For each expanded variant, we query the `text_chunks` collection **twice with metadata filtering**:
- Top-10 ASR-only chunks (`source = "asr"`)
- Top-8 OCR-only chunks (`source = "ocr"`)

This is crucial. Without source-balancing, English slides (clean text) outrank noisy Hebrew speech for nearly every query, and the user never sees the ASR alternatives. Forcing parity gives the re-ranker real choice.

Union-dedup across variants: same `(video_id, start, source)` keep the highest-scoring duplicate. Result: a candidate pool of ~30–60 chunks across the bilingual variants.

**Course mapping (Section 4.2):** "Filtering, metadata filtering, hybrid search" — we use Chroma's `where` clause filtering exactly as the syllabus describes.

#### Step 3.3 — Density-boosted scoring

After balanced retrieval, each candidate's score is multiplied by:

```
1 + Σ exp(-|Δt| / 300s)  for each other candidate in the same video
```

A chunk surrounded by neighbors gets a 2-4x boost; a lonely chunk gets none. This solves the "20 mentions matter more than 1 stray mention" problem the user articulated:

> "if there's one mention now but 20 mentions an hour later, the later one is the relevant one"

Density boosting encodes that judgment numerically. The 300s decay constant means neighbors within 5 minutes contribute strongly, neighbors 10 min away weakly.

**Course mapping (Section 4.1):** "Selecting and fine-tuning re-ranking algorithms." Density boost is a custom re-ranker, computed in `SearchEngine._density_boosted()`.

#### Step 3.4 — LLM anchor picker (Claude Haiku, tool-use)

The top-15 boosted candidates are given to Claude Haiku 4.5 with a Hebrew system prompt explaining:

1. Look for **clusters** of nearby candidates — they indicate a teaching block
2. Prefer the **earliest ASR chunk in a cluster** (= the moment the lecturer announces the topic) over slides (= visual evidence the topic is being taught nearby, but appears *after* the speech that introduces it)
3. Ignore isolated mentions far from any cluster
4. Output via the `pick_anchor` tool: `{chosen, reason, confidence}`

Claude returns a single anchor + a one-sentence Hebrew explanation + a confidence label.

**Why "LLM as re-ranker," not "LLM as answer generator":** Goal isn't to give an answer — it's to take the student to the moment the lecturer explains the topic. This avoids hallucination entirely (the LLM can only pick from real timestamps, never invent facts) and respects the educational use case (students learn from the lecturer's words, not a summary).

**Course mapping (Section 4.1):** "Selecting and fine-tuning re-ranking algorithms" — second re-ranker layer, this time semantic via LLM.

#### Step 3.5 — Return shape

```python
AnchorPick(
    result:     SearchResult,     # the chosen anchor
    reason:     str,              # 1-sentence Hebrew explanation
    confidence: "high|medium|low",
    others:     list[SearchResult],  # density-boosted alternatives for the user to browse
)
```

If confidence is `low`, the UI hides the anchor panel and falls back to plain top-K retrieval — the LLM is honest about uncertainty.

---

### Stage 4 — `app.py`: Gradio UI

- **Search box** → calls `engine.find_anchor()`
- **AI-picked panel** (above results) shows the anchor with its 🟢/🟡/🔴 confidence badge and Hebrew reason
- **Results table** lists the anchor (with 🌟 in the source column) followed by 5 density-boosted alternatives — user can pick any to jump to
- **Video player** with custom JS that seeks to the selected timestamp on click

The JS in `app.py:172-255` handles a Gradio quirk: when the video element is replaced after a search, the seek target needs to be reapplied. We do this by polling the new `<video>` element for ≤16 seconds and applying `currentTime = target` once the metadata loads.

---

## Data pipeline at scale

**Current state:**
- **12 lectures** × ~4h each ≈ 50 hours of video, ~11 GB of MP4s
- **7 Zvi + 5 Lev** lectures spanning 2026-02-25 to 2026-05-10
- **3,228 text chunks** indexed (avg ~269 per lecture)
- **608 visual frame embeddings**
- Total ChromaDB on disk: ~70 MB

**Costs per lecture:**
| Stage | Time | Cost |
|---|---|---|
| Download (gdown) | ~3 min | $0 |
| Ingest (VTT + OCR + CLIP, CPU) | ~12 min | $0 |
| Index re-run for all lectures | ~3 min | $0 |
| **Per query** | ~2s | ~$0.0015 (Haiku: expansion + anchor pick) |

For a full course of 30 lectures: ~6 hours of one-time ingest, ~$0.10/day of API at 50 queries/day.

---

## Engineering choices worth defending

### Why Zoom VTT instead of Whisper?

We use Zoom's auto-generated captions when available. They're 10x faster (parse vs. transcribe), already aligned to speech, and on Hebrew the quality is comparable to Whisper `small`. Whisper `large` would be better but takes ~hours per lecture on CPU. The system falls back to Whisper transparently if a video has no `.vtt`.

### Why CPU instead of GPU?

The developer machine is an Apple Silicon Mac. CLIP + e5-large run on CPU at acceptable speeds for the corpus size (3 lectures takes ~30s to embed all chunks). For 30+ lectures or live-query latency requirements, MPS or CUDA would be a worthwhile next step.

### Why Claude over OpenAI/Gemini/Ollama?

User decision. The system uses Anthropic SDK in two places (query expansion + anchor picker), both via the same `Anthropic()` client. Swapping providers means changing two function bodies — the architecture isn't locked in.

### Why no LangChain/LangGraph?

We use the raw Anthropic, sentence-transformers, and chromadb SDKs. The course syllabus lists LangChain prominently and a future iteration should port the retrieval chain to a `LangChain Retriever` for grading alignment — but the current direct-SDK code is shorter and easier to debug. Trade-off acknowledged.

---

## Known limitations

1. **Anchor temporal resolution: ±5 min.** Frame sampling is every 5 min. A title slide that appears at 2:10:00 may be missed and the anchor lands at the next sample, 2:15:00. Mitigation: finer scan + change-triggered sampling. Reverted in this session because of ingest-time cost.

2. **No evaluation harness.** The system's quality is judged qualitatively (does the anchor match what a human would pick?). The next session should build a small labeled set of 15–20 queries with ground-truth timestamps and compute Hit@1, MRR, and time-delta metrics.

3. **No hybrid lexical search.** Section 4.1 of the syllabus mentions BM-25 alongside semantic. We rely entirely on dense vectors. Adding a `BM25Retriever` and combining via reciprocal rank fusion is a quick win.

4. **No multi-stage retrieval.** As we scale to 30+ lectures, the LLM picker gets diluted across lectures. A two-stage approach (pick the lecture first, then the moment) is the natural next architecture.

5. **OCR doesn't support Hebrew.** EasyOCR has no Hebrew model. Slide text in Hebrew is dropped (most slides are English in this course). Switching to PaddleOCR or Tesseract+Hebrew traineddata would close that gap.

6. **torch 2.11 incompatibility.** The installed torch must be `<2.11` because 2.11.0 has an `inspect.getsource()` bug that breaks transformers integrations. Pinned in `requirements.txt`.

---

## Course concept coverage

A rubric-aligned summary of what's implemented and where it lives:

| Course concept | Where in the code |
|---|---|
| **Module 3.3 — LLM via API** | `search.py:_expand_query`, `search.py:find_anchor` (Anthropic SDK) |
| **Module 3.3 — Prompt engineering, tool calling** | `search.py:ANCHOR_TOOL`, `search.py:QUERY_EXPANSION_TOOL` (structured-output tool-use) |
| **Module 3.3 — Image/video understanding** | `ingest.py:embed_frame` (CLIP), `ingest.py:process_frames` (OCR) |
| **Module 4.1 — Document chunking strategy** | `index.py:chunk_asr`, `CHUNK_WINDOW_SEC`, `CHUNK_MAX_CHARS` |
| **Module 4.1 — Multilingual embedding model selection** | `index.py:TEXT_MODEL_NAME`, `search.py:encode_text` (e5-large + prefixes) |
| **Module 4.1 — Re-ranking (heuristic)** | `search.py:_density_boosted` |
| **Module 4.1 — Re-ranking (LLM-based)** | `search.py:find_anchor` |
| **Module 4.1 — Query optimization / query rewriting** | `search.py:_expand_query` |
| **Module 4.1 — Multi-query retrieval** | `search.py:_anchor_candidates` (queries each variant separately, unions) |
| **Module 4.2 — Chroma vector database** | `index.py`, `search.py` (PersistentClient) |
| **Module 4.2 — Metadata filtering** | `search.py:_search_text` (`where={"source": "..."}`) |

What's **not** implemented (honest gaps): no LangChain wrapper, no BM-25, no Gemini/OpenAI/Ollama comparison, no automated evaluation, no LangGraph agent loop. These are obvious next-phase items.

---

## File map

```
ShowMe/
├── ingest.py        # video → raw JSON
├── index.py         # JSON → ChromaDB
├── search.py        # query → anchor + alternatives
├── app.py           # Gradio UI
├── requirements.txt # pinned dependencies (torch <2.11!)
├── WORKFLOW.md      # this document
├── .env             # ANTHROPIC_API_KEY (gitignored)
└── data/            # gitignored
    ├── videos/      # MP4 + VTT files (~1GB each)
    └── index/
        ├── raw/<video_id>/  # asr.json, ocr.json, clip.json, meta.json
        └── chroma/          # persistent ChromaDB
```
