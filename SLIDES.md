---
marp: true
theme: default
paginate: true
size: 16:9
header: 'ShowMe — Multimodal Semantic Search over Lecture Videos'
footer: 'AI Engineering Course · 2026'
style: |
  section { font-size: 28px; }
  h1 { color: #1a73e8; font-size: 46px; }
  h2 { color: #202124; border-bottom: 2px solid #1a73e8; padding-bottom: 4px; font-size: 36px; }
  code { background: #f1f3f4; padding: 2px 6px; border-radius: 4px; }
  table { font-size: 24px; }
  blockquote { border-left: 4px solid #fbbc04; background: #fffbe6; padding: 8px 16px; }
---

<!-- _class: lead -->

# ShowMe

## Find the moment a topic is taught
## in a 4-hour lecture


---

## The Problem

You remember a topic.
You don't remember **when** or **which lecture**.

- 12 lectures × 4 hours = 50 hours of video
- Hebrew speech + English slides
- Standard search returns *mentions*, not the *teaching moment*

---

<!-- _class: lead -->

# 1. Three Sources of Information

---

## Three Modalities per Lecture

| Source | Tool | What it captures |
|---|---|---|
| 🎙 Speech | Zoom VTT | Hebrew narration + English code terms |
| 📄 Slides | EasyOCR | Slide text (English) |
| 🖼 Visual | CLIP ViT-B/32 | 512-dim visual frame embedding |

One MP4 → three parallel streams of timestamped data.

---

## Sampling Strategy

| Stream | Cadence | Per 4h lecture |
|---|---|---|
| Speech (VTT) | All cues | ~1,000 cues |
| Slides + visual frames | **1 frame every 5 min** | ~50 frames |
| OCR runs only on changed slides | Pixel-diff vs previous kept frame | ~20-30 OCR calls |

**Why 5 min:** slide-driven lectures hold each slide for 4-8 min.
**Cost:** ~12 min CPU per lecture.

---

<!-- _class: lead -->

# 2. Multimodality in Action

---

## Why multimodality matters

- **Speech alone:** noisy Hebrew transliterations ("בא עם גודר" = "bi-encoder")
- **Slides alone:** clean text but they appear *after* the lecturer announces the topic
- **Visual alone:** semantic but coarse

**Combined:** speech tells us *when the topic starts*, slides confirm we're in the *right region*, visual catches *what looks similar*.

---

## How the three streams feed search

```
Query → embed → match against [speech chunks + slide chunks + visual frames]
              ↘                                                            ↙
                        Balanced retrieval (forced parity per source)
```

We never let one source dominate.
The LLM picker sees evidence from all three.

---

## Chunking the Speech

Raw Zoom VTT cues are short — *"now let's"*, *"see this"* — too small to embed alone.

**Strategy:** merge consecutive cues into chunks bounded by:
- `60 seconds` of speech, or
- `500 characters` of text

A chunk = one **thought**, not one sentence fragment.

~270 chunks per lecture.

---

<!-- _class: lead -->

# 3. Retrieval: Five Steps

---

## Step 1 — Bilingual Query Expansion

User types `"agent"`. Hebrew speech says `"סוכן"` or `"איג'נט"`.

Claude Haiku expands the query into ≤4 cross-language variants
*before* embedding.

```
"agent" → ["agent", "סוכן", "איג'נט", "AI agent"]
```

Cached per query. Cost ~$0.0005.

---

## Step 2 — Source-Balanced Retrieval

For each variant, query ChromaDB twice with metadata filters:

- **Top 10** chunks where `source = "asr"`
- **Top 8** chunks where `source = "ocr"`

Union, dedup. ~30–60 candidates per query.

*Without this: English slides outrank Hebrew speech 5-to-1.*

---

## Step 3 — Density-Boosted Scoring

A topic is taught where mentions **cluster**, not where one stray hit appears.

$$\text{boost}(c) = 1 + \sum_{o \in \text{others}} \exp\left(-\frac{|t_o - t_c|}{300s}\right)$$

Cluster of 5 nearby hits → 2–4× boost.
Lone hit → no boost.
No LLM cost.

---

## Step 4 — LLM Anchor Picker

Claude Haiku 4.5 with **tool-use** (structured output).

Top-15 boosted candidates + Hebrew prompt:

> "Find the **earliest ASR chunk in the cluster** — the lecturer announcing the topic. Slides confirm but appear *after* speech. Ignore isolated mentions."

Returns: `{chosen, reason, confidence}`.

---

## Step 5 — Honest Confidence

```json
{
  "chosen": 3,
  "reason": "המרצה מתחיל להסביר אנקודר באשכול ברור",
  "confidence": "high"
}
```

| Confidence | UI behavior |
|---|---|
| high / medium | Show anchor 🌟 above results |
| low | Hide anchor, show plain top-K |

---

<!-- _class: lead -->

# 4. Architecture

---

## The Pipeline

```
ingest.py    →    index.py    →    search.py    →    app.py
video → JSON      JSON → vectors    query → anchor    Gradio UI
                  (Chroma)          (Haiku re-rank)   + video seek
```

Four small scripts. Each runs independently.
Adding a lecture is idempotent (skip if already ingested).

---

## Anchor Picker, *Not* Generator

This is the key architectural choice.

| | Standard RAG | ShowMe |
|---|---|---|
| LLM output | Synthesized answer | A single timestamp |
| Hallucination risk | High | **None** |
| Truth source | LLM's summary | The lecturer's words |

The LLM can only choose — never invent.
The student watches the actual lecture.

---

<!-- _class: lead -->

# 5. Engineering Choices

---

## Tech Stack

| Layer | Tool |
|---|---|
| Transcript | Zoom VTT (fallback: OpenAI Whisper `small`) |
| OCR | EasyOCR (English) |
| Visual embedding | `openai/clip-vit-base-patch32` (512-dim) |
| Text embedding | `intfloat/multilingual-e5-large` (1024-dim) |
| Vector DB | ChromaDB (PersistentClient, cosine) |
| LLM (re-rank + expand) | Anthropic Claude Haiku 4.5 (tool-use) |
| UI | Gradio 6 |

**All embedding runs on CPU** (Apple Silicon, no GPU/MPS).

---

## What We Chose, and Why

| Choice | Reason |
|---|---|
| Zoom VTT > Whisper | 10× faster, equivalent Hebrew quality |
| e5-large multilingual | Hebrew + English; mpnet-base too weak |
| ChromaDB direct SDK | No LangChain overhead for this scale |
| Anthropic Haiku | Cheap (~$0.001/call), strong tool-use |
| Density boost (custom) | Encodes "clusters > lone hits" heuristic |

---

## What We Skipped, and Why

| Skipped | Reason |
|---|---|
| Answer generation | Wrong tool for the job — students want the lecture |
| BM-25 hybrid | v2 — dense alone is enough at this scale |
| LangChain wrapper | Smaller surface area for debugging |
| Evaluation harness | Manual validation only; next step |

---

<!-- _class: lead -->

# Demo

---

## Try These Queries

| Query | Lands on |
|---|---|
| `RAG` | Zvi explaining RAG architecture |
| `סוכן` / `agent` | Same moment, both languages |
| `אנקודר` | "בי-אנקודר" introduction |
| `chunks` | Chunking explanation |

12 lectures · 3,228 chunks · 608 visual frames · ~2 sec / query

---

<!-- _class: lead -->

# Thank You

github.com/havia/ShowMe

Questions?
