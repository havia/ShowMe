"""
app.py  —  Multimodal Video Search Engine
==========================================
Gradio web interface.  Run with:
    python app.py --db data/index/chroma/ --videos data/videos/

The UI lets you:
  • Type a search phrase (e.g. "RAG", "attention mechanism")
  • See results as timestamped cards (video name, time, source, snippet)
  • Click a result → the video opens at exactly that moment
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import gradio as gr
except ModuleNotFoundError:
    print("[setup] gradio not found. Installing into current interpreter...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

from search import SearchEngine, SearchResult


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def format_timestamp(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


SOURCE_LABEL = {
    "asr":    "🎙 Speech",
    "ocr":    "📄 Slide",
    "visual": "🖼 Visual",
}


def results_to_rows(
    results: list[SearchResult],
    anchor_index: int | None = None,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for i, r in enumerate(results):
        src = SOURCE_LABEL.get(r.source, r.source)
        if anchor_index is not None and i == anchor_index:
            src = f"🌟 {src}"
        rows.append([
            str(i + 1),
            format_timestamp(r.start),
            r.video_id,
            src,
            r.text[:200],
        ])
    return rows


def results_to_choices(
    results: list[SearchResult],
    anchor_index: int | None = None,
) -> list[str]:
    choices: list[str] = []
    for i, r in enumerate(results):
        ts = format_timestamp(r.start)
        src = SOURCE_LABEL.get(r.source, r.source)
        if anchor_index is not None and i == anchor_index:
            src = f"🌟 {src}"
        choices.append(f"{i + 1}. {ts} | {r.video_id} | {src}")
    return choices


# ─────────────────────────────────────────────
# Build Gradio app
# ─────────────────────────────────────────────

def build_app(engine: SearchEngine, videos_dir: Path):
    # Map video_id -> absolute file path
    video_map: dict[str, Path] = {}

    def add_videos_from_dir(directory: Path):
        if directory.exists() and directory.is_dir():
            for vf in directory.glob("*.mp4"):
                video_map[vf.stem] = vf.resolve()

    # Primary user-provided directory
    add_videos_from_dir(videos_dir)

    # Fallbacks for common project layouts
    if not video_map:
        add_videos_from_dir(Path("videos"))
        add_videos_from_dir(Path("data/videos"))

    # Store last results in a closure list
    _last_results: list[SearchResult] = []

    CONF_BADGE = {"high": "🟢 high", "medium": "🟡 medium", "low": "🔴 low"}

    def do_search(query: str, top_k: int, video_filter: str):
        nonlocal _last_results
        if not query.strip():
            return gr.update(value="", visible=False), [], gr.update(choices=[], value=None), None

        vf = video_filter if video_filter and video_filter != "All videos" else None
        pick = engine.find_anchor(query, top_k_display=int(top_k), video_filter=vf)
        if pick is None:
            _last_results = []
            return gr.update(value="", visible=False), [], gr.update(choices=[], value=None), None

        if pick.confidence == "low":
            # No clear AI pick — fall back to raw top-K, no badge in the table
            _last_results = engine.search(query, top_k=int(top_k), video_filter=vf)
            anchor_index = None
            anchor_md = gr.update(value="", visible=False)
        else:
            # Merge anchor into a single list — anchor first, density-boosted alternates after
            _last_results = [pick.result] + pick.others
            anchor_index = 0
            badge = CONF_BADGE.get(pick.confidence, pick.confidence)
            anchor_md = gr.update(
                value=(
                    f"**🌟 AI-picked teaching moment — {badge}**  \n"
                    f"> {pick.reason}"
                ),
                visible=True,
            )

        rows = results_to_rows(_last_results, anchor_index=anchor_index)
        choices = results_to_choices(_last_results, anchor_index=anchor_index)
        default_choice = choices[0] if choices else None
        return anchor_md, rows, gr.update(choices=choices, value=default_choice), None

    def open_selected_result(choice: str | None):
        if not choice or not _last_results:
            return None

        try:
            idx = int(choice.split(".", 1)[0]) - 1
        except (ValueError, IndexError):
            return None

        if idx < 0 or idx >= len(_last_results):
            return None

        r = _last_results[idx]
        video_path = video_map.get(r.video_id)
        if video_path is None:
            gr.Warning(f"Video file not found for: {r.video_id}")
            return None

        return str(video_path.resolve())

    with gr.Blocks(title="🎬 Video Search Engine", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎬 Multimodal Video Search Engine\n"
            "Search across **speech** 🎙, **slides** 📄, and **visuals** 🖼 simultaneously."
        )

        with gr.Row():
            with gr.Column(scale=3):
                query_box = gr.Textbox(
                    placeholder='e.g. "RAG", "self-attention", "transformer architecture"',
                    label="Search phrase",
                    lines=1,
                )
            with gr.Column(scale=1):
                video_filter = gr.Dropdown(
                    choices=["All videos"] + list(video_map.keys()),
                    value="All videos",
                    label="Filter by video",
                )
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Results")
            with gr.Column(scale=1):
                search_btn = gr.Button("🔍 Search", variant="primary")

        # AI-picked anchor panel (visible only on high/medium confidence)
        anchor_panel = gr.Markdown(value="", visible=False)

        # Results panel
        results_table = gr.Dataframe(
            headers=["#", "Time", "Video", "Source", "Snippet"],
            datatype=["str", "str", "str", "str", "str"],
            value=[],
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
            col_count=(5, "fixed"),
            label="Results",
        )

        selected_result = gr.Dropdown(
            choices=[],
            value=None,
            label="Select Result",
            elem_id="selected_result",
        )
        open_btn = gr.Button("▶ Open Selected Result", variant="secondary", elem_id="open_result_btn")

        gr.Markdown("---\n### ▶ Video player")
        video_player = gr.Video(label="", interactive=False)

        gr.HTML(
            """
<script>
(function () {
    if (window.__showmeSeekInstalledV3) return;
    window.__showmeSeekInstalledV3 = true;
    window.__showmeSeekTarget = null;
    window.__showmeSeekToken = 0;

    function applySeek(video, token) {
        if (!video) return;
        if (token !== window.__showmeSeekToken) return;
        var target = Number(window.__showmeSeekTarget);
        if (!Number.isFinite(target)) return;
        try {
            video.currentTime = Math.max(0, target);
            // Don't auto-play: browsers force programmatic .play() to be muted
            // until the user interacts with the page. Let the user press play.
        } catch (e) {}
    }

    window.__showmeSeekApply = function () {
        var token = window.__showmeSeekToken;
        var tries = 0;
        var lastSrc = null;
        var timer = setInterval(function () {
            tries += 1;
            if (tries > 160 || token !== window.__showmeSeekToken) {
                clearInterval(timer);
                return;
            }

            var video = document.querySelector('video');
            if (!video) return;

            if (video.src !== lastSrc) {
                lastSrc = video.src;
                video.addEventListener('loadedmetadata', function onMeta() {
                    video.removeEventListener('loadedmetadata', onMeta);
                    applySeek(video, token);
                });
            }

            if (video.readyState >= 1) {
                applySeek(video, token);
            }
        }, 100);
    };

        function parseChoiceTime(text) {
                if (!text) return null;
                var m = text.match(/^\s*(\d+)\.\s+((?:\d{1,2}:)?\d{1,2}:\d{2})\s+\|/);
                if (!m) return null;
                var ts = m[2];
                var parts = ts.split(':').map(function (x) { return parseInt(x, 10); });
                if (parts.length === 2) return parts[0] * 60 + parts[1];
                if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
                return null;
        }

        function getSelectedChoiceText() {
                var input = document.querySelector('#selected_result input');
                return input ? input.value : null;
        }

        function installOpenHandler() {
                var btn = document.querySelector('#open_result_btn button');
                if (!btn || btn.__showmeSeekBound) return;
                btn.__showmeSeekBound = true;
                btn.addEventListener('click', function () {
                        var choice = getSelectedChoiceText();
                        var sec = parseChoiceTime(choice);
                        if (sec === null) return;
                        window.__showmeSeekTarget = sec;
                        window.__showmeSeekToken = (window.__showmeSeekToken || 0) + 1;
                        setTimeout(function () { if (window.__showmeSeekApply) window.__showmeSeekApply(); }, 120);
                        setTimeout(function () { if (window.__showmeSeekApply) window.__showmeSeekApply(); }, 800);
                });
        }

        setInterval(installOpenHandler, 300);
})();
</script>
        """
    )

        # Wire up search
        search_btn.click(
            fn=do_search,
            inputs=[query_box, top_k_slider, video_filter],
            outputs=[anchor_panel, results_table, selected_result, video_player],
        )
        query_box.submit(
            fn=do_search,
            inputs=[query_box, top_k_slider, video_filter],
            outputs=[anchor_panel, results_table, selected_result, video_player],
        )
        open_btn.click(
            fn=open_selected_result,
            inputs=[selected_result],
            outputs=[video_player],
        )
        selected_result.change(
            fn=open_selected_result,
            inputs=[selected_result],
            outputs=[video_player],
        )

        gr.Markdown(
            "<small>Powered by Whisper · EasyOCR · CLIP · sentence-transformers · ChromaDB</small>"
        )

    return demo


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",     type=Path, default=Path("data/index/chroma"))
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--port",   type=int,  default=7860)
    parser.add_argument("--share",  action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    engine = SearchEngine(args.db)
    app    = build_app(engine, args.videos)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
