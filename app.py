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

    def linkify_timestamps(text: str) -> str:
        """Turn [HH:MM:SS] markers in the answer into clickable seek links.
        Clicking dispatches a custom event that the JS handler picks up."""
        import re
        def _sub(m: re.Match) -> str:
            ts_text = m.group(1)
            parts = [int(x) for x in ts_text.split(":")]
            if len(parts) == 3:
                sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                sec = parts[0] * 60 + parts[1]
            else:
                return m.group(0)
            return (
                f'<a href="javascript:void(0)" '
                f'onclick="window.__showmeSeekTo({sec})" '
                f'style="color:#1a73e8; text-decoration:underline; cursor:pointer; '
                f'font-family:monospace; background:#eef3fc; padding:1px 4px; border-radius:3px;">'
                f'[{ts_text}]</a>'
            )
        return re.sub(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]", _sub, text)

    def do_search(query: str, top_k: int, video_filter: str, progress=gr.Progress()):
        nonlocal _last_results
        if not query.strip():
            return (
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                [], gr.update(choices=[], value=None), None,
            )

        progress(0.1, desc="מרחיב את השאילתה (Hebrew + English)…")
        vf = video_filter if video_filter and video_filter != "All videos" else None
        progress(0.3, desc="מאחזר קטעים רלוונטיים…")
        pick = engine.find_anchor(query, top_k_display=int(top_k), video_filter=vf)
        progress(0.9, desc="מארגן תוצאות…")
        if pick is None:
            _last_results = []
            return (
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                [], gr.update(choices=[], value=None), None,
            )

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

        # RAG answer panel — Hebrew summary with clickable [HH:MM:SS] citations.
        # Inherits color from the surrounding theme (light or dark); only the
        # accent stripe is colored.
        if pick.answer:
            answer_html = linkify_timestamps(pick.answer)
            answer_md = gr.update(
                value=(
                    f'<div dir="rtl" style="border-left:4px solid var(--color-accent, #1a73e8); '
                    f'padding:12px 16px; border-radius:4px; line-height:1.6; '
                    f'color:inherit;">'
                    f'<div style="font-weight:bold; color:var(--color-accent, #1a73e8); '
                    f'margin-bottom:6px;">'
                    f'🤖 תשובת RAG מבוססת על קטעי ההרצאה</div>'
                    f'<div style="color:inherit;">{answer_html}</div></div>'
                ),
                visible=True,
            )
        else:
            answer_md = gr.update(value="", visible=False)

        rows = results_to_rows(_last_results, anchor_index=anchor_index)
        choices = results_to_choices(_last_results, anchor_index=anchor_index)
        default_choice = choices[0] if choices else None
        return answer_md, anchor_md, rows, gr.update(choices=choices, value=default_choice), None

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
            with gr.Column(scale=5):
                query_box = gr.Textbox(
                    placeholder='e.g. "RAG", "self-attention", "transformer architecture"',
                    label="Search phrase",
                    lines=1,
                )
            with gr.Column(scale=1, min_width=120):
                search_btn = gr.Button("🔍 Search", variant="primary")

        with gr.Accordion("⚙ Settings", open=False):
            with gr.Row():
                video_filter = gr.Dropdown(
                    choices=["All videos"] + list(video_map.keys()),
                    value="All videos",
                    label="Filter by video",
                )
                top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Number of results")

        # RAG answer panel (visible whenever we have an LLM-generated answer)
        answer_panel = gr.HTML(value="", visible=False)

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
    if (window.__showmeSeekInstalledV4) return;
    window.__showmeSeekInstalledV4 = true;

    // Target timestamp we want the next-loaded video to seek to.
    window.__showmeSeekTarget = null;
    window.__showmeSeekTargetSrc = null;  // which video URL this target applies to (or null = any)

    function log() { try { console.log.apply(console, ['[showme]'].concat([].slice.call(arguments))); } catch (e) {} }

    function doSeekAndPlay(video) {
        var target = Number(window.__showmeSeekTarget);
        if (!Number.isFinite(target)) return false;
        try {
            log('seeking to', target, 'on video src=', video.currentSrc || video.src);
            // Set currentTime first; then unmute; then play.
            video.currentTime = Math.max(0, target);
            video.muted = false;
            var p = video.play();
            if (p && typeof p.catch === 'function') {
                p.catch(function (err) {
                    log('play-with-sound blocked, retrying muted:', err && err.name);
                    video.muted = true;
                    video.play().catch(function () {});
                });
            }
            return true;
        } catch (e) {
            log('seek failed:', e);
            return false;
        }
    }

    // Attempt the seek on a video. If readyState < 1 (no metadata), wait.
    // If readyState >= 1, seek now AND on the next 'canplay' (in case the
    // first currentTime= got rejected because src hadn't fully loaded).
    function applyToVideo(video) {
        if (!video) return;
        if (video.__showmeSeekDone === window.__showmeSeekTarget) return;  // idempotent
        video.__showmeSeekDone = window.__showmeSeekTarget;

        var onReady = function () {
            video.removeEventListener('loadedmetadata', onReady);
            video.removeEventListener('canplay', onReady);
            // Try seek now. Then again after a tick (some browsers need it).
            doSeekAndPlay(video);
            setTimeout(function () { doSeekAndPlay(video); }, 50);
        };

        if (video.readyState >= 1) {
            // Metadata already there
            doSeekAndPlay(video);
        } else {
            video.addEventListener('loadedmetadata', onReady, { once: true });
            video.addEventListener('canplay', onReady, { once: true });
        }
    }

    // Watch the DOM for video elements appearing/changing.
    function attachToAllVideos() {
        var videos = document.querySelectorAll('video');
        for (var i = 0; i < videos.length; i++) {
            var v = videos[i];
            if (v.__showmeSeekObserved) continue;
            v.__showmeSeekObserved = true;
            log('attached to video', v);
            // When src changes (Gradio loads a new file into the same element)
            // reset the seek-done flag so we'll re-seek for the new src.
            var resetSeek = function () { this.__showmeSeekDone = null; applyToVideo(this); };
            v.addEventListener('loadstart', resetSeek);
            v.addEventListener('emptied', resetSeek);
            v.addEventListener('loadedmetadata', function () { applyToVideo(this); });
            applyToVideo(v);
        }
    }

    // Watch for new video elements being added to the DOM.
    var mo = new MutationObserver(function () { attachToAllVideos(); });
    mo.observe(document.body, { childList: true, subtree: true });
    attachToAllVideos();
    // Also poll periodically in case the MutationObserver misses a swap.
    setInterval(attachToAllVideos, 500);

    // ── Choice parsing (anchor row in the dropdown) ──
    function parseChoiceTime(text) {
        if (!text) return null;
        var m = text.match(/(\d{1,2}):(\d{2}):(\d{2})/) || text.match(/(\d{1,2}):(\d{2})/);
        if (!m) return null;
        if (m.length === 4) return (+m[1]) * 3600 + (+m[2]) * 60 + (+m[3]);
        return (+m[1]) * 60 + (+m[2]);
    }

    function getSelectedChoiceText() {
        var input = document.querySelector('#selected_result input');
        return input ? input.value : null;
    }

    function triggerSeekFromChoice() {
        var choice = getSelectedChoiceText();
        var sec = parseChoiceTime(choice);
        if (sec === null) {
            log('no parsable time in choice:', choice);
            return;
        }
        log('user requested seek to', sec, 'from choice:', choice);
        seekTo(sec);
    }

    // Programmatic seek-to-time, callable from inline onclick handlers in the
    // RAG answer's [HH:MM:SS] citation links.
    function seekTo(sec) {
        if (!Number.isFinite(sec)) return;
        window.__showmeSeekTarget = sec;
        var vids = document.querySelectorAll('video');
        for (var i = 0; i < vids.length; i++) {
            vids[i].__showmeSeekDone = null;
            applyToVideo(vids[i]);
        }
    }
    window.__showmeSeekTo = seekTo;

    function installOpenHandler() {
        var btn = document.querySelector('#open_result_btn button');
        if (btn && !btn.__showmeSeekBound) {
            btn.__showmeSeekBound = true;
            btn.addEventListener('click', triggerSeekFromChoice);
            log('open-handler installed');
        }
        // Also react to the dropdown itself changing (selecting a row).
        var dropdown = document.querySelector('#selected_result input');
        if (dropdown && !dropdown.__showmeSeekBound) {
            dropdown.__showmeSeekBound = true;
            dropdown.addEventListener('change', function () { setTimeout(triggerSeekFromChoice, 100); });
            log('dropdown-handler installed');
        }
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
            outputs=[answer_panel, anchor_panel, results_table, selected_result, video_player],
            show_progress="minimal",
            show_progress_on=[answer_panel],
        )
        query_box.submit(
            fn=do_search,
            inputs=[query_box, top_k_slider, video_filter],
            outputs=[answer_panel, anchor_panel, results_table, selected_result, video_player],
            show_progress="minimal",
            show_progress_on=[answer_panel],
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
