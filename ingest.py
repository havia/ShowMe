"""
ingest.py  —  Multimodal Video Search Engine
============================================
Processes one or more video files and extracts:
  • Speech transcript with word-level timestamps  (Whisper)
  • Slide / on-screen text with timestamps        (EasyOCR)
  • Visual frame embeddings                       (CLIP)

Run:
    python ingest.py --videos data/videos/ --out data/index/raw/
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    print("[setup] opencv-python not found. Installing into current interpreter...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2
try:
    import easyocr
except ModuleNotFoundError:
    print("[setup] easyocr not found. Installing into current interpreter...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr"])
    import easyocr
import numpy as np
import torch
try:
    import whisper
except ModuleNotFoundError:
    print("[setup] openai-whisper not found. Installing into current interpreter...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
    import whisper
from PIL import Image
try:
    from transformers import CLIPModel, CLIPProcessor
except ModuleNotFoundError:
    print("[setup] transformers not found. Installing into current interpreter...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

FRAME_INTERVAL_SEC = 1        # extract 1 frame per second
OCR_CHANGE_THRESHOLD = 0.15   # skip OCR if frame looks same as previous
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
WHISPER_MODEL_SIZE = "base"   # tiny | base | small | medium | large
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Model loading  (called once, shared across videos)
# ─────────────────────────────────────────────

def load_models():
    print(f"[models] Loading on device: {DEVICE}")

    print("[models] Loading Whisper …")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)

    print("[models] Loading EasyOCR …")
    ocr_reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))

    print("[models] Loading CLIP …")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip_model.eval()

    return whisper_model, ocr_reader, clip_processor, clip_model


# ─────────────────────────────────────────────
# Audio extraction
# ─────────────────────────────────────────────

def get_ffmpeg_executable() -> str:
    """Return an ffmpeg executable path, installing a bundled fallback if needed."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
        print("[setup] ffmpeg not found on PATH. Installing imageio-ffmpeg fallback...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
        import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()

def extract_audio(video_path: Path, tmp_dir: str) -> Path:
    """Use ffmpeg to extract audio as 16kHz mono WAV (required by Whisper)."""
    audio_path = Path(tmp_dir) / "audio.wav"
    ffmpeg_exe = get_ffmpeg_executable()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-ar", "16000",   # 16 kHz sample rate
        "-ac", "1",       # mono
        "-vn",            # no video
        str(audio_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_path


# ─────────────────────────────────────────────
# Whisper transcription
# ─────────────────────────────────────────────

def transcribe(audio_path: Path, whisper_model) -> list[dict]:
    """
    Returns a list of segments, each:
        { "start": float, "end": float, "text": str }
    Whisper's word_timestamps=True gives us fine-grained timing.
    """
    print("  [whisper] Transcribing audio …")
    result = whisper_model.transcribe(
        str(audio_path),
        word_timestamps=True,
        verbose=False,
    )

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end":   round(seg["end"],   2),
            "text":  seg["text"].strip(),
        })

    print(f"  [whisper] {len(segments)} segments found.")
    return segments


# ─────────────────────────────────────────────
# Frame extraction + OCR + CLIP
# ─────────────────────────────────────────────

def frame_difference(prev: np.ndarray, curr: np.ndarray) -> float:
    """Mean absolute pixel difference (0–1 range) to detect slide changes."""
    if prev is None:
        return 1.0
    diff = cv2.absdiff(
        cv2.resize(prev, (64, 64)),
        cv2.resize(curr, (64, 64)),
    )
    return float(diff.mean()) / 255.0


def embed_frame(frame_rgb: np.ndarray, clip_processor, clip_model) -> list[float]:
    """Return CLIP visual embedding as a plain Python list (for JSON serialisation)."""
    image = Image.fromarray(frame_rgb)
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
        if not isinstance(embedding, torch.Tensor):
            if hasattr(embedding, "image_embeds") and embedding.image_embeds is not None:
                embedding = embedding.image_embeds
            elif hasattr(embedding, "pooler_output") and embedding.pooler_output is not None:
                embedding = embedding.pooler_output
            else:
                raise TypeError(f"Unexpected CLIP output type: {type(embedding)!r}")
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalise
    return embedding.squeeze().cpu().tolist()


def process_frames(
    video_path: Path,
    ocr_reader,
    clip_processor,
    clip_model,
) -> tuple[list[dict], list[dict]]:
    """
    Iterate over video frames at FRAME_INTERVAL_SEC intervals.

    Returns:
        ocr_records   — [{ "timestamp": float, "text": str }, …]
        clip_records  — [{ "timestamp": float, "embedding": [float, …] }, …]
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    frame_step = int(fps * FRAME_INTERVAL_SEC)

    ocr_records  = []
    clip_records = []
    prev_frame   = None
    frame_idx    = 0

    pbar = tqdm(total=int(duration_sec), unit="s", desc="  [frames]")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp = round(frame_idx / fps, 2)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            diff = frame_difference(prev_frame, frame_rgb)

            # ── CLIP embedding (every sampled frame) ──────────────────
            clip_vec = embed_frame(frame_rgb, clip_processor, clip_model)
            clip_records.append({"timestamp": timestamp, "embedding": clip_vec})

            # ── OCR (only when slide has changed enough) ───────────────
            if diff > OCR_CHANGE_THRESHOLD:
                ocr_results = ocr_reader.readtext(frame_rgb, detail=0, paragraph=True)
                text = " ".join(ocr_results).strip()
                if text:
                    ocr_records.append({"timestamp": timestamp, "text": text})

            prev_frame = frame_rgb
            pbar.update(FRAME_INTERVAL_SEC)

        frame_idx += 1

    pbar.close()
    cap.release()

    print(f"  [frames] {len(clip_records)} CLIP embeddings, {len(ocr_records)} OCR records.")
    return ocr_records, clip_records


# ─────────────────────────────────────────────
# Main ingestion function
# ─────────────────────────────────────────────

def ingest_video(
    video_path: Path,
    out_dir: Path,
    whisper_model,
    ocr_reader,
    clip_processor,
    clip_model,
) -> dict:
    """
    Full pipeline for one video. Saves three JSON files:
        <out_dir>/<video_stem>/asr.json      — Whisper transcript segments
        <out_dir>/<video_stem>/ocr.json      — OCR text records
        <out_dir>/<video_stem>/clip.json     — CLIP embeddings
    Returns a summary dict.
    """
    video_id = video_path.stem
    save_dir = out_dir / video_id
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Processing: {video_path.name}")
    print(f"{'='*60}")

    # ── 1. Audio → Whisper ────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = extract_audio(video_path, tmp)
        asr_segments = transcribe(audio_path, whisper_model)

    # ── 2. Frames → OCR + CLIP ────────────────────────────────────
    ocr_records, clip_records = process_frames(
        video_path, ocr_reader, clip_processor, clip_model
    )

    # ── 3. Save ───────────────────────────────────────────────────
    def save_json(data, filename):
        with open(save_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    save_json(asr_segments, "asr.json")
    save_json(ocr_records,  "ocr.json")
    save_json(clip_records,  "clip.json")

    # Save a small metadata file too
    meta = {
        "video_id":     video_id,
        "video_path":   str(video_path),
        "asr_segments": len(asr_segments),
        "ocr_records":  len(ocr_records),
        "clip_records": len(clip_records),
    }
    save_json(meta, "meta.json")

    print(f"  ✓ Saved to {save_dir}/")
    return meta


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest videos for semantic search.")
    parser.add_argument(
        "--videos", type=Path, required=True,
        help="Path to a single .mp4 file OR a folder of .mp4 files",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/index/raw"),
        help="Output directory for JSON files (default: data/index/raw/)",
    )
    args = parser.parse_args()

    # Collect video files
    if args.videos.is_dir():
        video_files = sorted(args.videos.glob("*.mp4"))
    else:
        video_files = [args.videos]

    if not video_files:
        print("No .mp4 files found. Exiting.")
        return

    print(f"Found {len(video_files)} video(s).")

    # Load models once
    whisper_model, ocr_reader, clip_processor, clip_model = load_models()

    # Process each video
    all_meta = []
    for vf in video_files:
        meta = ingest_video(
            vf, args.out,
            whisper_model, ocr_reader, clip_processor, clip_model,
        )
        all_meta.append(meta)

    # Write a summary index
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "index.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\n✓ Ingestion complete. {len(all_meta)} video(s) processed.")
    print(f"  Index saved to: {args.out / 'index.json'}")


if __name__ == "__main__":
    main()
