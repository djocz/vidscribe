# Subtitle & Transcript Generator

> Generate subtitles and transcripts from YouTube videos or local video files using OpenAI Whisper — fully local, no API cost.

---

## What It Does

Accepts either a **YouTube URL** or a **local video file**, extracts its audio, transcribes it, and produces:

| File | Description |
|------|-------------|
| `.srt` | Standard subtitle format (VLC, most media players) |
| `.vtt` | WebVTT format — ready for manual upload to YouTube |
| `_transcript.txt` | Human-readable transcript with `[MM:SS]` timestamps |
| `audio.mp3` | Extracted audio, kept for reference |
| `manifest.json` | Metadata: title, language, model used, timestamps |

All outputs are written into a **single folder per video** named after its unique ID:

```
outputs/
  dQw4w9WgXcQ/                     ← YouTube video ID
      audio.mp3
      Never Gonna Give You Up.srt
      Never Gonna Give You Up.vtt
      Never Gonna Give You Up_transcript.txt
      manifest.json

  my_lecture/                       ← local file stem
      audio.mp3
      my_lecture.srt
      ...
```

Re-running the same input overwrites the existing folder, keeping everything for one video in one place.

---

## Requirements

**Python 3.8+**

Install Python dependencies:

```bash
pip install yt-dlp openai-whisper
```

Install FFmpeg (required for audio extraction):

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

---

## Usage

### YouTube video

```bash
python subtitle_generator.py run --input "https://youtube.com/watch?v=abc123"
```

### Local video file

```bash
python subtitle_generator.py run --input /path/to/lecture.mp4
```

### Local audio file

```bash
python subtitle_generator.py run --input /path/to/recording.mp3
```

### Choose Whisper model size

```bash
python subtitle_generator.py run --input "..." --model large
```

### Get help

```bash
python subtitle_generator.py --help
python subtitle_generator.py run --help
```

---

## Whisper Model Sizes

| Model | Speed | Accuracy | VRAM | Recommended For |
|-------|-------|----------|------|-----------------|
| `tiny` | Fastest | Low | ~1 GB | Quick drafts |
| `base` | Fast | Moderate | ~1 GB | General testing |
| `small` | Medium | Good | ~2 GB | **Default — best balance** |
| `medium` | Slow | Very good | ~5 GB | High accuracy needs |
| `large` | Slowest | Best | ~10 GB | Professional / final output |

---

## Supported File Types

**Video:** `.mp4` `.mkv` `.avi` `.mov` `.wmv` `.flv` `.webm` `.m4v` `.mpeg` `.mpg` `.3gp`

**Audio:** `.mp3` `.wav` `.flac` `.aac` `.ogg` `.m4a`

---

## Uploading Subtitles to YouTube

The `.vtt` file in the output folder can be uploaded manually via YouTube Studio:

1. Go to [YouTube Studio](https://studio.youtube.com) → select your video
2. Click **Subtitles** in the left sidebar
3. Click **Add language** → select the language
4. Under *Subtitles*, click **Add** → **Upload file**
5. Select the `.vtt` file from `outputs/{video_id}/`
6. Click **Save**

---

## Output: manifest.json

Every run writes a `manifest.json` alongside the subtitle files:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Never Gonna Give You Up",
  "source_type": "youtube",
  "input": "https://youtube.com/watch?v=dQw4w9WgXcQ",
  "language": "en",
  "whisper_model": "small",
  "processed_at": "2024-03-15T14:30:22+00:00",
  "files": {
    "audio": "audio.mp3",
    "srt": "Never Gonna Give You Up.srt",
    "vtt": "Never Gonna Give You Up.vtt",
    "transcript": "Never Gonna Give You Up_transcript.txt"
  }
}
```

---

## Project Structure

```
project/
├── subtitle_generator.py   ← main script
├── README.md
└── outputs/
    └── {video_id}/         ← one folder per video
        ├── audio.mp3
        ├── {title}.srt
        ├── {title}.vtt
        ├── {title}_transcript.txt
        └── manifest.json
```