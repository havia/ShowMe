# Multimodal Semantic Video Search Engine

Search across course lecture videos using natural language.
Finds moments where a phrase was **spoken**, **shown on a slide**, or **visually present**.

## How it works

| Signal | Model | What it finds |
|--------|-------|---------------|
| Speech | OpenAI Whisper | Timestamps where the lecturer *said* your phrase |
| Slide text | EasyOCR | Timestamps where your phrase *appeared on screen* |
| Visual | CLIP ViT-B/32 | Frames that are *semantically related* to your phrase |

Results are fused and ranked by a weighted combination of all three scores.

---

## Setup

### 1. Install ffmpeg (system-level)
```bash
# Ubuntu / WSL
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 2. Install Python packages
```bash
pip install -r requirements.txt
```

> **GPU recommended** but CPU works fine for small collections.
> Whisper `base` model runs comfortably on CPU.

---

## Usage

### Step 1 — Add your videos
```
data/
└── videos/
    ├── lecture_01.mp4
    ├── lecture_02.mp4
    └── ...
```

### Step 2 — Ingest (extract speech, OCR, CLIP embeddings)
```bash
python ingest.py --videos data/videos/ --out data/index/raw/
```
This saves JSON files in `data/index/raw/<video_name>/`:
- `asr.json`  — Whisper transcript segments with timestamps
- `ocr.json`  — OCR text with timestamps
- `clip.json` — CLIP visual embeddings per frame

### Step 3 — Index (build vector database)
```bash
python index.py --raw data/index/raw/ --db data/index/chroma/
```

### Step 4 — Search (CLI quick test)
```bash
python search.py --query "RAG retrieval augmented generation" --db data/index/chroma/
```

### Step 5 — Launch the web UI
```bash
python app.py --db data/index/chroma/ --videos data/videos/
# Open http://localhost:7860
```

---

## Project structure

```
video_search_engine/
├── ingest.py          # Stage 1: video → JSON (Whisper + OCR + CLIP)
├── index.py           # Stage 2: JSON → ChromaDB vector index
├── search.py          # Stage 3: query → ranked results
├── app.py             # Gradio web UI
├── requirements.txt
└── data/
    ├── videos/        # your .mp4 files go here
    └── index/
        ├── raw/       # intermediate JSON from ingest.py
        └── chroma/    # ChromaDB vector store
```

---

## Configuration

Edit the constants at the top of each file:

| File | Variable | Default | Description |
|------|----------|---------|-------------|
| `ingest.py` | `FRAME_INTERVAL_SEC` | `1` | How often to sample frames |
| `ingest.py` | `WHISPER_MODEL_SIZE` | `base` | `tiny/base/small/medium/large` |
| `search.py` | `TEXT_WEIGHT` | `0.6` | Weight of text score in fusion |
| `search.py` | `VISUAL_WEIGHT` | `0.4` | Weight of visual score in fusion |
| `search.py` | `DEDUP_WINDOW_SEC` | `3.0` | Suppress near-duplicate results |

---

## Possible extensions for your report

- **Re-ranking with a cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for higher precision
- **Query expansion** using an LLM to generate synonyms before searching
- **Speaker diarization** (pyannote.audio) to label which person is talking
- **Multilingual support** — Whisper supports 99 languages, swap EasyOCR language list
- **YouTube ingestion** — `yt-dlp` to download lecture videos automatically
