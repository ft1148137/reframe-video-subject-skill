# Reframe Video Subject Skill

Codex skill for reframing videos around a selected person. It supports YOLO-based person tracking, fixed subject-centered crops, and light clarity repair for tight crops that look soft after scaling.

## Layout

```text
.
├── README.md
└── reframe-video-subject/
    ├── SKILL.md
    ├── agents/
    └── scripts/
```

Install or copy the `reframe-video-subject/` folder into your Codex skills directory.

## What It Does

- Tracks a chosen person and keeps them centered in a requested aspect ratio.
- Falls back to fixed crops when similar subjects make detection jump.
- Preserves audio while rendering H.264 MP4 outputs.
- Adds optional light denoise/sharpen export settings for tight crops.

## Requirements

- `ffmpeg` and `ffprobe`
- Python packages: `opencv-python`, `numpy`, `ultralytics`

Install Python dependencies when needed:

```bash
python3 -m pip install opencv-python numpy ultralytics
```

## Usage

Use the skill from Codex, or run the helper script directly:

```bash
python3 reframe-video-subject/scripts/reframe_subject.py input.mp4 \
  --output output.mp4 \
  --target-description "center dancer in black jacket" \
  --aspect 4:3
```

See `reframe-video-subject/SKILL.md` for the full workflow and ffmpeg fallback commands.
