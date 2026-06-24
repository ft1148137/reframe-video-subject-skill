---
name: reframe-video-subject
description: Reframe a video into a subject-centered edit using ffmpeg plus person detection or fixed crop framing, with light clarity repair when a tight crop looks soft. Use when the user asks to keep a selected person centered, crop tightly around a subject while removing nearby people, make a vertical/TikTok/Reels/YouTube/1:1/4:3 crop, auto-reframe a dancer or speaker, track a person by clothing color or position, sharpen or lightly denoise a cropped subject video, generate smooth camera keyframes, or produce a resized video from an input .mp4/.mov/.mkv file. Before editing, ask the user which subject to track and what output ratio to use if either is missing.
---

# Reframe Video Subject

## Workflow

Before running the script, identify two inputs:

1. The target subject to track, such as "rightmost man in yellow shirt", "speaker in white hoodie", or "center dancer".
2. The output ratio, such as `9:16`, `1:1`, `4:5`, or `16:9`.

If the user did not specify either one, ask a short question before editing.

Use `scripts/reframe_subject.py` for moving-camera subject tracking. It does:

1. Probe the video with `ffprobe`.
2. Extract sampled frames, default `2 fps` for one keyframe every `0.5s`.
3. Run YOLO person detection on each sampled frame.
4. Select the target person by clothing color plus continuity.
5. Center the crop on the selected person's bbox center.
6. Smooth the camera path with Hermite interpolation so motion is continuous between keyframes.
7. Render the vertical output with `ffmpeg`, preserving audio.

Default output is 9:16, 1080x1920, H.264, CRF 18, unless the user requests another ratio.

For tight crops where the subject should stay centered and nearby people should be removed, first try a fixed crop if the subject stays within a small region. Fixed crop is simpler and avoids YOLO jumping between similar people.

## Commands

Basic:

```bash
python3 scripts/reframe_subject.py \
  input.mp4 \
  --output input_vertical.mp4
```

For a yellow/brown/orange shirt subject:

```bash
python3 scripts/reframe_subject.py \
  input.mp4 \
  --output input_vertical.mp4 \
  --hsv-low 5,35,35 \
  --hsv-high 35,255,235 \
  --prefer right \
  --sample-fps 2 \
  --target-description "rightmost man in yellow/brown shirt" \
  --aspect 9:16
```

For square output:

```bash
python3 scripts/reframe_subject.py \
  input.mp4 \
  --output input_square.mp4 \
  --target-description "main speaker" \
  --aspect 1:1
```

Write visual QA sheets:

```bash
python3 scripts/reframe_subject.py \
  input.mp4 \
  --output input_vertical.mp4 \
  --debug-dir /tmp/reframe-debug
```

Fixed centered crop fallback:

```bash
mkdir -p /tmp/reframe-qa
ffmpeg -hide_banner -loglevel error -y -ss 12 -i input.mp4 \
  -vf "crop=480:360:440:230,scale=1440:1080:flags=lanczos,setsar=1" \
  -frames:v 1 /tmp/reframe-qa/check.jpg

ffmpeg -hide_banner -y -i input.mp4 \
  -vf "crop=480:360:440:230:exact=1,scale=1440:1080:flags=lanczos,setsar=1" \
  -map 0:v:0 -map '0:a?' \
  -c:v libx264 -crf 18 -preset medium -c:a copy -movflags +faststart \
  output_4x3_centered.mp4
```

For fixed crops, choose `crop_w:crop_h` to match the requested ratio, then choose `x` and `y` from sampled frames so the target is centered and never clipped. For `4:3`, examples include `480:360`, `520:390`, `560:420`, and `960:720`.

Same crop, clearer export:

```bash
ffmpeg -hide_banner -y -i input.mp4 \
  -vf "crop=480:360:440:230:exact=1,hqdn3d=1.0:1.0:3.0:3.0,scale=1440:1080:flags=lanczos,cas=0.55,unsharp=3:3:0.25:3:3:0.0,setsar=1" \
  -map 0:v:0 -map '0:a?' \
  -c:v libx264 -crf 14 -preset slow -c:a copy -movflags +faststart \
  output_4x3_centered_sharp.mp4
```

Use this only from the original input, not from an already cropped export. Keep the same `crop=` values when the user says the framing is right but soft.

## Rules

- Do not use 60fps sampling by default. If the crop jumps, improve detection/trajectory smoothing first.
- Prefer person bbox center over shirt-color center. Use color only to identify the target among multiple people.
- Use Hermite smoothing for camera paths unless the user explicitly wants hard keyframe jumps.
- Ask for the target subject and output ratio before rendering if they are not already clear.
- If the subject still feels off-center, inspect the debug sheets before re-rendering.
- If YOLO picks the wrong person during overlaps, increase continuity weight or manually patch the JSON plan instead of increasing sample FPS.
- If automatic tracking jumps between two similar subjects, stop tracking and use a fixed crop. Sample start/middle/end frames, pick the tightest ratio-correct crop that keeps the target fully in frame, then adjust `x`/`y` until the target is centered.
- Prioritize keeping the target complete and centered over removing every nearby person. A slightly wider crop is better than clipping hands, feet, or head during movement.
- When a tight crop looks blurry, first re-export from the original with lower compression (`-crf 14` to `16`, `-preset slow`) and light `hqdn3d`, `cas`, or `unsharp`; do not re-encode the previous cropped file.
- Be honest about the limit: a small source crop such as `480x360` scaled to `1440x1080` can look cleaner and sharper, but not truly high-resolution. Avoid aggressive sharpening that creates halos or amplified noise.

## Dependencies

The script expects:

- `ffmpeg` and `ffprobe` on `PATH`, or installed at `/opt/homebrew/bin/ffmpeg` and `/opt/homebrew/bin/ffprobe`.
- Python packages: `opencv-python`, `numpy`, `ultralytics`.

Install missing packages only when needed:

```bash
python3 -m pip install ultralytics opencv-python numpy
```
