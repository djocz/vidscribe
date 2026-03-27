# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Tool

```bash
# YouTube video
python vidscribe.py run --input "https://youtube.com/watch?v=VIDEO_ID"

# Local video or audio file
python vidscribe.py run --input /path/to/file.mp4

# With AI chapter generation (requires ANTHROPIC_API_KEY)
python vidscribe.py run --input "..." --chapters

# With a specific Whisper model (default: small)
python vidscribe.py run --input "..." --model medium

# Help
python vidscribe.py run --help
```

## Dependencies

```bash
pip install yt-dlp openai-whisper anthropic python-dotenv
brew install ffmpeg  # macOS; see README for other platforms
```

## Architecture

Single-file script (`youtube_subtitle_generator.py`) with a linear pipeline:

1. **Input detection** — determines if input is a YouTube URL or local file path
2. **Audio extraction** — `download_youtube()` via yt-dlp, or `prepare_local()` via FFmpeg
3. **Transcription** — `transcribe_audio()` runs OpenAI Whisper locally (no API cost)
4. **Output generation** — writes `.srt`, `.vtt`, `_transcript.txt` from Whisper segments
5. **Chapter generation** (optional, `--chapters`) — `generate_chapters()` sends the timestamped transcript to the Claude API and returns YouTube-ready chapter markers
6. **Manifest** — writes `manifest.json` with metadata for each run

All outputs land in `outputs/{video_id}/` (YouTube ID) or `outputs/{filename_stem}/` (local files). Re-running overwrites the folder.

## Environment

`ANTHROPIC_API_KEY` is required only for `--chapters`. Store it in a `.env` file in the project root (loaded automatically at startup). Shell environment variables also work.

## Note on Script Name

The script file is `vidscribe.py`.
