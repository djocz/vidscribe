"""
================================================================================
  Subtitle & Transcript Generator  (YouTube + Local Video)
================================================================================

DESCRIPTION
-----------
Accepts either a YouTube URL or a local video file, extracts its audio,
transcribes it using OpenAI Whisper (runs fully locally, no API cost), and
produces:

  - SRT subtitle file  (.srt)
  - WebVTT subtitle file (.vtt)   <- ready for manual upload to YouTube
  - Plain text transcript (.txt)
  - chapters.txt                  <- YouTube-ready chapter timestamps (with --chapters)
  - manifest.json                 <- metadata for the run
  - audio.mp3                     <- extracted audio, kept in the output folder

All outputs land in a single folder named after the video's unique ID:

  outputs/
    {video_id}/                   <- YouTube video ID  (e.g. dQw4w9WgXcQ)
        audio.mp3
        {title}.srt
        {title}.vtt
        {title}_transcript.txt
        chapters.txt
        manifest.json

  For local files the folder is named after the sanitised filename stem
  (e.g. outputs/my_lecture/).  Re-running the same input overwrites the
  existing folder so everything for one video always stays in one place.

REQUIREMENTS
------------
Python 3.8+

Install dependencies:
    pip install yt-dlp openai-whisper anthropic python-dotenv

FFmpeg is required for audio extraction:
  - macOS:   brew install ffmpeg
  - Ubuntu:  sudo apt install ffmpeg
  - Windows: https://ffmpeg.org/download.html

ANTHROPIC API KEY  (only needed for --chapters)
-----------------------------------------------
Chapter generation uses the Claude API.  Store your key in a .env file in the
same directory as this script:

    echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env

The .env file is loaded automatically at startup.  As a fallback the script
also checks the shell environment variable ANTHROPIC_API_KEY, so both work:

    # Option A — .env file  (recommended, never committed to git)
    ANTHROPIC_API_KEY=sk-ant-...

    # Option B — shell environment variable
    export ANTHROPIC_API_KEY="sk-ant-..."

Get a key at: https://console.anthropic.com

IMPORTANT: Add .env to your .gitignore so the key is never committed:
    echo '.env' >> .gitignore

CLI USAGE
---------
  # YouTube URL:
      python vidscribe.py run --input "https://youtube.com/watch?v=ID"

  # Local video file:
      python vidscribe.py run --input /path/to/my_lecture.mp4

  # Generate AI chapters alongside subtitles:
      python vidscribe.py run --input "..." --chapters

  # Choose Whisper model size (default: small):
      python vidscribe.py run --input "..." --model medium

  # Full help:
      python vidscribe.py --help
      python vidscribe.py run --help

WHISPER MODEL SIZES
-------------------
  tiny    -- Fastest,  lowest accuracy  (~1 GB VRAM)
  base    -- Fast,     moderate accuracy (~1 GB VRAM)
  small   -- Medium,   good accuracy     (~2 GB VRAM)  <- Default
  medium  -- Slow,     very good         (~5 GB VRAM)
  large   -- Slowest,  best accuracy     (~10 GB VRAM)

UPLOADING TO YOUTUBE
--------------------
The .vtt file generated in the output folder can be uploaded manually:
  1. Go to YouTube Studio -> your video -> Subtitles
  2. Click "Add" -> "Upload file" -> select the .vtt file
  3. Set the language and save

================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from datetime import datetime, timezone


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Supported local video and audio extensions
LOCAL_VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".webm", ".m4v", ".mpeg", ".mpg", ".3gp",
}
LOCAL_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}


# ── Load .env file  ────────────────────────────────────────────────────────────

def _load_env() -> None:
    """
    Load variables from a .env file in the project root into os.environ.

    Uses python-dotenv when available; falls back to a minimal built-in parser
    so the script works even if python-dotenv is not installed.

    Priority (highest to lowest):
      1. Variables already present in the shell environment  (never overwritten)
      2. Variables defined in .env
    """
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.isfile(env_path):
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)   # override=False preserves shell env
    except ImportError:
        # Minimal fallback parser — handles KEY=value and KEY="value"
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:   # never overwrite shell env
                    os.environ[key] = val


# Load .env once at import time so all functions see the variables
_load_env()


# ── Lazy import helper ─────────────────────────────────────────────────────────

def _require(module: str, pip_name: str | None = None):
    """Import a module or exit with a clear install message."""
    try:
        return importlib.import_module(module)
    except ImportError:
        pip = pip_name or module
        print(f"\n[ERROR] '{module}' is not installed.  Run:\n    pip install {pip}\n")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Input detection helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_youtube_url(value: str) -> bool:
    """Return True if the input looks like a YouTube URL."""
    return bool(re.search(r"(youtube\.com|youtu\.be)", value, re.IGNORECASE))


def _is_local_file(value: str) -> bool:
    """Return True if the input is a path to an existing file."""
    return os.path.isfile(value)


def _local_video_id(path: str) -> str:
    """
    Derive a stable folder-safe identifier from a local file path.
    Uses the filename stem, sanitised to alphanumerics, hyphens, and underscores.
    Falls back to an 8-char SHA-256 hash prefix if the stem is blank.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", stem).strip("_")
    if not safe:
        with open(path, "rb") as f:
            safe = hashlib.sha256(f.read(65536)).hexdigest()[:8]
    return safe


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1a — Download audio from a YouTube URL
# ══════════════════════════════════════════════════════════════════════════════

def download_youtube(url: str, video_folder: str) -> tuple[str, str, str, str]:
    """
    Download audio from a YouTube URL using yt-dlp.
    Saved as {video_folder}/audio.mp3.

    Returns:
        (audio_filepath, video_id, video_title, video_description)
    """
    yt_dlp     = _require("yt_dlp", "yt-dlp")
    audio_base = os.path.join(video_folder, "audio")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{audio_base}.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info        = ydl.extract_info(url, download=True)
        vid_id      = info.get("id", "unknown")
        title       = info.get("title", "Unknown Title")
        description = info.get("description", "").strip()

    audio_file = f"{audio_base}.mp3"
    print(f"  Audio saved      -> {audio_file}")
    return audio_file, vid_id, title, description


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1b — Prepare audio from a local video or audio file
# ══════════════════════════════════════════════════════════════════════════════

def prepare_local(path: str, video_folder: str) -> tuple[str, str, str]:
    """
    Extract audio from a local video file using FFmpeg, or copy a local audio
    file directly.  Result is saved as {video_folder}/audio.mp3.

    Returns:
        (audio_filepath, video_id, title)
        Both video_id and title are derived from the filename stem.
    """
    import shutil as _shutil
    import subprocess

    vid_id     = _local_video_id(path)
    title      = os.path.splitext(os.path.basename(path))[0]
    ext        = os.path.splitext(path)[1].lower()
    audio_file = os.path.join(video_folder, "audio.mp3")

    if ext in LOCAL_AUDIO_EXTS:
        # Source is already audio — copy it in directly
        _shutil.copy2(path, audio_file)
        print(f"  Audio copied     -> {audio_file}")
    else:
        # Extract audio stream from video using FFmpeg
        print("  Extracting audio via FFmpeg ...")
        cmd = [
            "ffmpeg", "-y",
            "-i", path,
            "-vn",          # discard video stream
            "-ar", "16000", # 16 kHz sample rate (optimal for Whisper)
            "-ac", "1",     # mono
            "-q:a", "0",    # highest quality
            audio_file,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n[ERROR] FFmpeg failed:\n{result.stderr}")
            sys.exit(1)
        print(f"  Audio saved      -> {audio_file}")

    return audio_file, vid_id, title


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Transcribe with OpenAI Whisper  (fully local, no API cost)
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_path: str, model_size: str = "small") -> dict:
    """
    Transcribe an audio file using a locally-run OpenAI Whisper model.

    Args:
        audio_path:  Path to the audio file (.mp3, .wav, etc.)
        model_size:  tiny | base | small | medium | large

    Returns:
        Whisper result dict  { text, segments, language }
    """
    whisper = _require("whisper", "openai-whisper")

    print(f"  Loading Whisper '{model_size}' model ...")
    model = whisper.load_model(model_size)

    print("  Transcribing ... (may take a while for long videos)")
    result = model.transcribe(
        audio_path,
        verbose=False,
        word_timestamps=True,
        task="transcribe",   # change to "translate" to force English output
    )

    lang = result.get("language", "unknown")
    print(f"  Transcription done  -- detected language: {lang}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Generate output files
# ══════════════════════════════════════════════════════════════════════════════

def _to_srt_time(s: float) -> str:
    h, r   = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    ms     = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _to_vtt_time(s: float) -> str:
    h, r   = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    ms     = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


def generate_srt(result: dict, output_file: str) -> str:
    """Write a standard SRT subtitle file."""
    blocks = []
    for i, seg in enumerate(result["segments"], 1):
        blocks.append(
            f"{i}\n"
            f"{_to_srt_time(seg['start'])} --> {_to_srt_time(seg['end'])}\n"
            f"{seg['text'].strip()}\n"
        )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    print(f"  SRT saved        -> {output_file}")
    return output_file


def generate_vtt(result: dict, output_file: str) -> str:
    """Write a WebVTT subtitle file — ready for manual upload to YouTube."""
    lines = ["WEBVTT\n"]
    for i, seg in enumerate(result["segments"], 1):
        lines.append(
            f"{i}\n"
            f"{_to_vtt_time(seg['start'])} --> {_to_vtt_time(seg['end'])}\n"
            f"{seg['text'].strip()}\n"
        )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  VTT saved        -> {output_file}")
    return output_file


def generate_transcript(result: dict, output_file: str) -> str:
    """Write a human-readable transcript with [MM:SS] timestamps."""
    lines = ["=" * 60, "  TRANSCRIPT", "=" * 60, ""]
    for seg in result["segments"]:
        mm, ss = divmod(int(seg["start"]), 60)
        lines.append(f"[{mm:02d}:{ss:02d}]  {seg['text'].strip()}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Transcript saved -> {output_file}")
    return output_file


# ══════════════════════════════════════════════════════════════════════════════
#  Manifest helper
# ══════════════════════════════════════════════════════════════════════════════

def _write_manifest(folder: str, meta: dict) -> str:
    """Write (or overwrite) manifest.json inside the output folder."""
    path = os.path.join(folder, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Manifest saved   -> {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — AI-based chapter generation  (Claude API)
# ══════════════════════════════════════════════════════════════════════════════

def generate_chapters(result: dict, title: str, output_file: str) -> str | None:
    """
    Use the Claude API to detect natural topic boundaries in the transcript
    and produce YouTube-ready chapter timestamps.

    The transcript (with timestamps) is sent to Claude, which returns chapters
    in the exact format YouTube expects:
        00:00 Introduction
        03:45 Setting up the project
        ...

    Rules Claude is instructed to follow:
      - First chapter must be 00:00
      - At least 3 chapters, no more than 1 per 2 minutes of content
      - Timestamps in MM:SS  (or HH:MM:SS for videos over 1 hour)
      - Chapter titles are concise (3-6 words)

    Output is written to {output_file} and also printed to the terminal.

    Args:
        result:       Whisper transcription result dict
        title:        Video title (used as context for Claude)
        output_file:  Destination path for chapters.txt

    Returns:
        Path to the written chapters file, or None if the API call failed.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print(
            "\n  [SKIP] ANTHROPIC_API_KEY is not set — cannot generate chapters.\n"
            "  Add it to a .env file in the project root:\n"
            "      echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env\n"
            "  Or export it in your shell:\n"
            "      export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "  Get a key at: https://console.anthropic.com\n"
        )
        return None

    # ── Build a compact timestamped transcript for the prompt ─────────────────
    segments = result.get("segments", [])
    if not segments:
        print("  [SKIP] No transcript segments found — cannot generate chapters.")
        return None

    # Determine total duration to inform Claude about video length
    total_secs = int(segments[-1]["end"])
    total_mm, total_ss = divmod(total_secs, 60)
    total_hh, total_mm = divmod(total_mm, 60)
    duration_str = (
        f"{total_hh}h {total_mm}m {total_ss}s" if total_hh
        else f"{total_mm}m {total_ss}s"
    )

    # Build compact transcript: [MM:SS] text
    transcript_lines = []
    for seg in segments:
        mm, ss = divmod(int(seg["start"]), 60)
        hh, mm = divmod(mm, 60)
        ts = f"{hh}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"
        transcript_lines.append(f"[{ts}] {seg['text'].strip()}")
    transcript_text = "\n".join(transcript_lines)

    prompt = f"""You are creating YouTube chapters for a video.

Video title: {title}
Video duration: {duration_str}

Below is the full timestamped transcript:

{transcript_text}

---

Generate YouTube chapter markers following these rules exactly:
1. The FIRST chapter must always start at 00:00
2. Identify natural topic changes — do not split arbitrarily
3. Minimum 3 chapters, maximum 1 chapter per 2 minutes of content
4. Each timestamp must match the exact moment the topic begins in the transcript
5. Use MM:SS format (use HH:MM:SS only if the video is over 1 hour)
6. Chapter titles must be concise: 3 to 6 words, title case
7. Output ONLY the chapter list, no preamble, no explanation, no markdown

Example format:
00:00 Introduction
04:22 Installing Dependencies
09:15 Writing the Core Logic
18:40 Testing and Debugging
24:00 Wrap Up"""

    print("  Generating chapters via Claude API ...")

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        message  = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1024,
            messages   = [{"role": "user", "content": prompt}],
        )
        chapters_text = message.content[0].text.strip()
    except Exception as e:
        print(f"\n  [ERROR] Claude API call failed: {e}\n")
        return None

    # ── Validate the response looks like chapter timestamps ───────────────────
    valid_lines = [
        ln for ln in chapters_text.splitlines()
        if re.match(r"^\d{1,2}:\d{2}", ln.strip())
    ]
    if not valid_lines:
        print("  [WARN] Claude returned an unexpected format — chapters not saved.")
        print(f"  Raw response:\n{chapters_text}")
        return None

    chapters_text = "\n".join(valid_lines)

    # ── Write to file ─────────────────────────────────────────────────────────
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(chapters_text + "\n")

    # ── Print to terminal ─────────────────────────────────────────────────────
    print(f"  Chapters saved   -> {output_file}")
    print("\n" + "─" * 60)
    print("  CHAPTERS  (paste into YouTube video description)")
    print("─" * 60)
    for line in valid_lines:
        print(f"  {line}")
    print("─" * 60 + "\n")

    return output_file


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Podcast publishing  (optional, --publish)
# ══════════════════════════════════════════════════════════════════════════════

def _xml_escape(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _format_duration(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _rfc822_now() -> str:
    from email.utils import formatdate
    return formatdate(localtime=False)


def _generate_rss(title: str, audio_path: str, result: dict, output_file: str, audio_url: str = "", description: str = "") -> str:
    """
    Write a podcast RSS 2.0 feed for the episode.

    If audio_url is empty a placeholder is written — the user must replace it
    with the public URL of the hosted audio before submitting to platforms.
    description — if empty, falls back to the first 300 chars of the transcript.
    """
    segments   = result.get("segments", [])
    duration   = _format_duration(segments[-1]["end"]) if segments else "00:00:00"
    audio_size = os.path.getsize(audio_path) if os.path.isfile(audio_path) else 0
    url        = audio_url or "REPLACE_WITH_PUBLIC_AUDIO_URL"
    if not description:
        description = result.get("text", "").strip()[:300]
        if len(result.get("text", "")) > 300:
            description += "..."

    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0"\n'
        '     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">\n'
        '  <channel>\n'
        f'    <title>{_xml_escape(title)}</title>\n'
        f'    <description>{_xml_escape(description)}</description>\n'
        '    <itunes:author>vidscribe</itunes:author>\n'
        '    <itunes:explicit>false</itunes:explicit>\n'
        '    <item>\n'
        f'      <title>{_xml_escape(title)}</title>\n'
        f'      <description>{_xml_escape(description)}</description>\n'
        f'      <enclosure url="{url}" length="{audio_size}" type="audio/mpeg"/>\n'
        f'      <guid isPermaLink="false">{url}</guid>\n'
        f'      <pubDate>{_rfc822_now()}</pubDate>\n'
        f'      <itunes:duration>{duration}</itunes:duration>\n'
        '    </item>\n'
        '  </channel>\n'
        '</rss>\n'
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(feed)

    print(f"  RSS feed saved   -> {output_file}")
    if not audio_url:
        print("  [NOTE] Edit podcast_feed.xml and replace REPLACE_WITH_PUBLIC_AUDIO_URL")
        print("         with the public URL of your hosted audio before submitting to platforms.")
    return output_file


def publish_podcast(
    title: str,
    audio_path: str,
    result: dict,
    video_folder: str,
    audio_url: str = "",
    description: str = "",
) -> dict:
    """
    Write a podcast_feed.xml RSS 2.0 feed for the episode.

    audio_url   — full public URL of the hosted audio file. If empty, a
                  placeholder is written that the user must replace manually.
    description — episode description. Falls back to transcript excerpt.

    Returns {"rss_file": str}.
    """
    rss_file = os.path.join(video_folder, "podcast_feed.xml")
    _generate_rss(title, audio_path, result, rss_file, audio_url=audio_url, description=description)
    return {"rss_file": rss_file}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Per-show RSS feed  (accumulates all episodes for a show)
# ══════════════════════════════════════════════════════════════════════════════

def _load_show_index(show_dir: str) -> dict:
    """Load show_index.json or return a fresh empty index."""
    path = os.path.join(show_dir, "show_index.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"episodes": []}


def _save_show_index(show_dir: str, index: dict) -> None:
    path = os.path.join(show_dir, "show_index.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def _build_show_feed(show: str, index: dict) -> str:
    """Build a complete podcast RSS 2.0 feed from the show index."""
    show_title = show.replace("-", " ").replace("_", " ").title() if show else "vidscribe Podcast"

    items = []
    for ep in index["episodes"]:
        items.append(
            "    <item>\n"
            f"      <title>{_xml_escape(ep['title'])}</title>\n"
            f"      <description>{_xml_escape(ep['description'])}</description>\n"
            f"      <enclosure url=\"{ep['audio_url']}\" length=\"{ep['audio_size']}\" type=\"audio/mpeg\"/>\n"
            f"      <guid isPermaLink=\"false\">{ep['guid']}</guid>\n"
            f"      <pubDate>{ep['pub_date']}</pubDate>\n"
            f"      <itunes:duration>{ep['duration']}</itunes:duration>\n"
            "    </item>"
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0"\n'
        '     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">\n'
        '  <channel>\n'
        f'    <title>{_xml_escape(show_title)}</title>\n'
        f'    <description>{_xml_escape(show_title)}</description>\n'
        '    <itunes:author>vidscribe</itunes:author>\n'
        '    <itunes:explicit>false</itunes:explicit>\n'
        + "\n".join(items) + "\n"
        '  </channel>\n'
        '</rss>\n'
    )


def update_show_feed(
    show: str,
    vid_id: str,
    title: str,
    audio_path: str,
    audio_url: str,
    result: dict,
    description: str = "",
) -> str:
    """
    Add or update an episode in the show-level RSS feed.

    The feed lives at outputs/{show}/feed.xml and accumulates every episode
    processed under that show name, newest first.  Re-running the same video
    replaces its existing entry rather than duplicating it.

    Returns the path to the written feed.xml.
    """
    show_dir = os.path.join(OUTPUTS_DIR, show)
    os.makedirs(show_dir, exist_ok=True)

    segments = result.get("segments", [])
    if not description:
        description = result.get("text", "").strip()[:300]
        if len(result.get("text", "")) > 300:
            description += "..."

    episode = {
        "guid":        vid_id,
        "title":       title,
        "description": description,
        "audio_url":   audio_url or f"REPLACE_WITH_PUBLIC_AUDIO_URL/{vid_id}/audio.mp3",
        "audio_size":  os.path.getsize(audio_path) if os.path.isfile(audio_path) else 0,
        "duration":    _format_duration(segments[-1]["end"]) if segments else "00:00:00",
        "pub_date":    _rfc822_now(),
    }

    index = _load_show_index(show_dir)

    # Replace existing entry for this vid_id, or prepend as new episode
    existing = [ep for ep in index["episodes"] if ep["guid"] != vid_id]
    index["episodes"] = [episode] + existing

    _save_show_index(show_dir, index)

    feed_path = os.path.join(show_dir, "feed.xml")
    with open(feed_path, "w", encoding="utf-8") as f:
        f.write(_build_show_feed(show, index))

    ep_count = len(index["episodes"])
    print(f"  Show feed saved  -> {feed_path}  ({ep_count} episode{'s' if ep_count != 1 else ''})")
    return feed_path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Upload audio to Cloudflare R2  (optional, --upload r2)
# ══════════════════════════════════════════════════════════════════════════════

def _upload_to_r2(audio_path: str, vid_id: str, show: str) -> str | None:
    """
    Upload audio.mp3 to Cloudflare R2 using the S3-compatible API.

    Requires in .env:
        R2_ACCOUNT_ID        — found on the R2 overview page
        R2_ACCESS_KEY_ID     — R2 API token (Access Key ID)
        R2_SECRET_ACCESS_KEY — R2 API token (Secret Access Key)
        R2_BUCKET            — bucket name  (e.g. vidscribe-podcast)
        R2_PUBLIC_URL        — public bucket URL  (e.g. https://pub-abc.r2.dev)

    File is stored at:
        {show}/{vid_id}/audio.mp3   (when --show is provided)
        {vid_id}/audio.mp3          (when --show is omitted)

    Returns the public URL of the uploaded file, or None on failure.
    """
    boto3 = _require("boto3")

    account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    bucket     = os.environ.get("R2_BUCKET", "").strip()
    public_url = os.environ.get("R2_PUBLIC_URL", "").strip()

    missing = [k for k, v in {
        "R2_ACCOUNT_ID": account_id, "R2_ACCESS_KEY_ID": access_key,
        "R2_SECRET_ACCESS_KEY": secret_key, "R2_BUCKET": bucket,
        "R2_PUBLIC_URL": public_url,
    }.items() if not v]

    if missing:
        print(
            f"\n  [SKIP] R2 upload: missing .env vars: {', '.join(missing)}\n"
            "  Add to .env:\n"
            "      R2_ACCOUNT_ID=...\n"
            "      R2_ACCESS_KEY_ID=...\n"
            "      R2_SECRET_ACCESS_KEY=...\n"
            "      R2_BUCKET=vidscribe-podcast\n"
            "      R2_PUBLIC_URL=https://pub-xxx.r2.dev\n"
        )
        return None

    key = f"{show}/{vid_id}/audio.mp3" if show else f"{vid_id}/audio.mp3"

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url         = f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id    = access_key,
            aws_secret_access_key= secret_key,
            region_name          = "auto",
        )
        print(f"  Uploading to R2  -> {bucket}/{key} ...")
        s3.upload_file(
            audio_path, bucket, key,
            ExtraArgs={"ContentType": "audio/mpeg"},
        )
        url = public_url.rstrip("/") + "/" + key
        print(f"  R2 audio URL     -> {url}")
        return url
    except Exception as e:
        print(f"\n  [ERROR] R2 upload failed: {e}\n")
        return None


def _upload_file_to_r2(local_path: str, key: str, content_type: str = "application/xml") -> None:
    """Upload any local file to R2 at the given key. Used for show feed.xml."""
    boto3 = _require("boto3")

    account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    bucket     = os.environ.get("R2_BUCKET", "").strip()
    public_url = os.environ.get("R2_PUBLIC_URL", "").strip()

    if not all([account_id, access_key, secret_key, bucket]):
        return

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url          = f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id     = access_key,
            aws_secret_access_key = secret_key,
            region_name           = "auto",
        )
        s3.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": content_type})
        url = public_url.rstrip("/") + "/" + key
        print(f"  Show feed on R2  -> {url}")
    except Exception as e:
        print(f"\n  [ERROR] R2 feed upload failed: {e}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline  (called by the `run` sub-command)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    input_value: str,
    whisper_model: str = "small",
    chapters: bool = False,
    publish: bool = False,
    base_url: str = "",
    upload: str = "",
    show: str = "",
) -> dict:
    print("\n" + "=" * 60)
    print("  Subtitle & Transcript Generator")
    print("=" * 60 + "\n")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── Sanitise show name to snake_case ─────────────────────────────────────
    if show:
        show = re.sub(r"[^\w\s\-]", "", show, flags=re.UNICODE)
        show = re.sub(r"\s+", "_", show.strip()).lower() or "show"

    # ── Detect input type and resolve video ID ───────────────────────────────
    if _is_youtube_url(input_value):
        source_type = "youtube"
        print("  Source           -> YouTube URL")

        # Fetch metadata first to get the video ID before creating the folder
        yt_dlp = _require("yt_dlp", "yt-dlp")
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info   = ydl.extract_info(input_value, download=False)
            vid_id = info.get("id", "unknown")

        video_folder = os.path.join(OUTPUTS_DIR, vid_id)
        os.makedirs(video_folder, exist_ok=True)
        print(f"  Output folder    -> outputs/{vid_id}/\n")

        audio_file, _, title, yt_description = download_youtube(input_value, video_folder)

    elif _is_local_file(input_value):
        ext = os.path.splitext(input_value)[1].lower()
        if ext not in LOCAL_VIDEO_EXTS | LOCAL_AUDIO_EXTS:
            print(f"\n[ERROR] Unsupported file type: '{ext}'")
            print(f"  Supported video: {', '.join(sorted(LOCAL_VIDEO_EXTS))}")
            print(f"  Supported audio: {', '.join(sorted(LOCAL_AUDIO_EXTS))}\n")
            sys.exit(1)

        source_type  = "local"
        vid_id       = _local_video_id(input_value)
        video_folder = os.path.join(OUTPUTS_DIR, vid_id)
        os.makedirs(video_folder, exist_ok=True)
        print(f"  Source           -> Local file  ({input_value})")
        print(f"  Output folder    -> outputs/{vid_id}/\n")

        audio_file, _, title = prepare_local(input_value, video_folder)
        yt_description = ""

    else:
        print(
            f"\n[ERROR] Input not recognised: '{input_value}'\n"
            f"  Provide a YouTube URL or the path to an existing local file.\n"
        )
        sys.exit(1)

    # ── Transcribe ───────────────────────────────────────────────────────────
    result        = transcribe_audio(audio_file, model_size=whisper_model)
    detected_lang = result.get("language", "en")

    # ── Build a safe filename from the video title ───────────────────────────
    safe = re.sub(r"[^\w\s\-]", "", title, flags=re.UNICODE)
    safe = re.sub(r"\s+", "_", safe.strip()).lower()[:60] or vid_id

    # ── Write subtitle and transcript files ──────────────────────────────────
    print()
    srt_file = generate_srt(result,        os.path.join(video_folder, f"{safe}.srt"))
    vtt_file = generate_vtt(result,        os.path.join(video_folder, f"{safe}.vtt"))
    txt_file = generate_transcript(result, os.path.join(video_folder, f"{safe}_transcript.txt"))

    # ── Generate AI chapters (optional) ──────────────────────────────────────
    chapters_file = None
    if chapters:
        chapters_file = generate_chapters(
            result,
            title       = title,
            output_file = os.path.join(video_folder, "chapters.txt"),
        )

    # ── Upload audio to R2 (optional) ────────────────────────────────────────
    r2_audio_url = ""
    if upload == "r2":
        r2_audio_url = _upload_to_r2(audio_file, vid_id, show) or ""

    # ── Resolve final audio URL for RSS ──────────────────────────────────────
    audio_url = (
        r2_audio_url
        or (base_url.rstrip("/") + "/" + os.path.basename(audio_file) if base_url else "")
    )

    # ── Publish as podcast (optional) ────────────────────────────────────────
    podcast_result = None
    show_feed_file = None
    if publish or upload:
        podcast_result = publish_podcast(
            title        = title,
            audio_path   = audio_file,
            result       = result,
            video_folder = video_folder,
            audio_url    = audio_url,
            description  = yt_description,
        )

    # ── Update per-show RSS feed (when --show is given) ───────────────────────
    if show and (publish or upload):
        show_feed_file = update_show_feed(
            show        = show,
            vid_id      = vid_id,
            title       = title,
            audio_path  = audio_file,
            audio_url   = audio_url,
            result      = result,
            description = yt_description,
        )
        # Upload show feed to R2 so the URL is always current
        if upload == "r2":
            _upload_file_to_r2(show_feed_file, f"{show}/feed.xml")

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest_data = {
        "video_id":      vid_id,
        "title":         title,
        "source_type":   source_type,
        "input":         input_value,
        "language":      detected_lang,
        "whisper_model": whisper_model,
        "processed_at":  datetime.now(timezone.utc).isoformat(),
        "files": {
            "audio":      os.path.basename(audio_file),
            "srt":        os.path.basename(srt_file),
            "vtt":        os.path.basename(vtt_file),
            "transcript": os.path.basename(txt_file),
            "chapters":   os.path.basename(chapters_file) if chapters_file else None,
            "podcast_rss":  os.path.basename(podcast_result["rss_file"]) if podcast_result else None,
            "show_feed":    show_feed_file,
            "r2_audio_url": r2_audio_url or None,
        },
    }
    manifest_file = _write_manifest(video_folder, manifest_data)

    print(f"\n  Done!  outputs/{vid_id}/\n")
    return {
        "video_id":      vid_id,
        "folder":        video_folder,
        "audio":         audio_file,
        "srt":           srt_file,
        "vtt":           vtt_file,
        "transcript":    txt_file,
        "chapters":      chapters_file,
        "manifest":      manifest_file,
        "podcast_rss":   podcast_result["rss_file"] if podcast_result else None,
        "show_feed":     show_feed_file,
    }


def cmd_run(args: argparse.Namespace) -> None:
    run_pipeline(
        input_value   = args.input,
        whisper_model = args.model,
        chapters      = args.chapters,
        publish       = args.publish,
        base_url      = args.base_url,
        upload        = args.upload,
        show          = args.show,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — argument parser
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidscribe",
        description=(
            "Generate subtitles, transcripts, and AI-based chapter markers "
            "from a YouTube URL or local video file using OpenAI Whisper."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  YouTube video:
    python vidscribe.py run --input "https://youtube.com/watch?v=abc123"

  Local video file:
    python vidscribe.py run --input /path/to/lecture.mp4

  Local audio file:
    python vidscribe.py run --input /path/to/recording.mp3

  Generate AI chapters (requires ANTHROPIC_API_KEY):
    python vidscribe.py run --input "..." --chapters

  Use a more accurate Whisper model:
    python vidscribe.py run --input /path/to/lecture.mp4 --model large

  Transcribe + chapters in one go:
    python vidscribe.py run --input "https://youtube.com/watch?v=abc123" --model medium --chapters

  Publish as a podcast (generates local RSS feed):
    python vidscribe.py run --input "..." --publish

  Publish with audio host URL pre-filled in the RSS feed:
    python vidscribe.py run --input "..." --publish --base-url https://pub-abc.r2.dev/podcast/my_video

  Upload audio to Cloudflare R2 + generate RSS (requires R2_* vars in .env):
    python vidscribe.py run --input "..." --upload r2 --show my-tech-podcast

uploading subtitles to youtube:
  YouTube Studio -> your video -> Subtitles -> Add -> Upload file -> select .vtt

adding chapters to youtube:
  Paste the contents of chapters.txt into the video description on YouTube Studio.
        """,
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── run ───────────────────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help="Transcribe a YouTube URL or local video/audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Transcribe a YouTube URL or local video/audio file with Whisper and\n"
            "save SRT, VTT, transcript, and audio into outputs/{video_id}/."
        ),
    )
    run_p.add_argument(
        "--input", "-i",
        required=True,
        metavar="URL_OR_PATH",
        help=(
            "YouTube URL  (https://youtube.com/watch?v=...)  "
            "or path to a local video/audio file"
        ),
    )
    run_p.add_argument(
        "--model", "-m",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        metavar="MODEL",
        help="Whisper model size: tiny | base | small (default) | medium | large",
    )
    run_p.add_argument(
        "--chapters", "-c",
        action="store_true",
        default=False,
        help=(
            "Generate AI-based chapter markers using the Claude API. "
            "Requires ANTHROPIC_API_KEY environment variable to be set. "
            "Output is saved to chapters.txt and printed to the terminal."
        ),
    )
    run_p.add_argument(
        "--publish", "-p",
        action="store_true",
        default=False,
        help=(
            "Publish the extracted audio as a podcast episode. "
            "Generates podcast_feed.xml in the output folder. "
            "Use --host to upload directly to a hosting service."
        ),
    )
    run_p.add_argument(
        "--base-url",
        default="",
        metavar="URL",
        dest="base_url",
        help=(
            "Base URL where you will host the audio file (used when not uploading). "
            "The audio filename is appended automatically. "
            "Example: --base-url https://pub-abc123.r2.dev/podcast/my_video"
        ),
    )
    run_p.add_argument(
        "--upload",
        default="",
        choices=["r2"],
        metavar="SERVICE",
        help=(
            "Upload audio to a cloud storage service and generate RSS automatically. "
            "Supported: r2 (Cloudflare R2). "
            "Requires R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, "
            "R2_BUCKET, R2_PUBLIC_URL in .env."
        ),
    )
    run_p.add_argument(
        "--show",
        default="",
        metavar="SHOW",
        help=(
            "Show name used as a folder prefix in R2 storage. "
            "Organises multiple shows under one bucket. "
            "Example: --show my-tech-podcast  ->  my-tech-podcast/{video_id}/audio.mp3"
        ),
    )
    run_p.set_defaults(func=cmd_run)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)