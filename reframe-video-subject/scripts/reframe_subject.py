#!/usr/bin/env python3
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def bin_path(name):
    found = shutil.which(name)
    if found:
        return found
    homebrew = Path("/opt/homebrew/bin") / name
    if homebrew.exists():
        return str(homebrew)
    raise SystemExit(f"missing {name}; install ffmpeg first")


FFMPEG = bin_path("ffmpeg")
FFPROBE = bin_path("ffprobe")


def run(cmd):
    subprocess.run(cmd, check=True)


def parse_triplet(text):
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected H,S,V")
    return np.array(parts, dtype=np.uint8)


def parse_aspect(text):
    try:
        w, h = [int(p) for p in text.split(":")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected W:H, for example 9:16") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("aspect values must be positive")
    return w, h


def default_output_size(aspect):
    w, h = aspect
    if w == h:
        return 1080, 1080
    if w < h:
        return 1080, int(round(1080 * h / w))
    return 1920, int(round(1920 * h / w))


def probe_video(path):
    out = subprocess.check_output(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=index,codec_type,width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    data = json.loads(out)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    return int(video["width"]), int(video["height"]), float(data["format"]["duration"])


def extract_frames(src, frames_dir, sample_fps):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.jpg"):
        old.unlink()
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"fps={sample_fps}",
            str(frames_dir / "frame_%04d.jpg"),
        ]
    )
    return sorted(frames_dir.glob("frame_*.jpg"))


def load_yolo(model_name):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("missing ultralytics; run: python3 -m pip install ultralytics") from exc
    return YOLO(model_name)


def torso_color_ratio(hsv, box, hsv_low, hsv_high):
    h, w = hsv.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1 = max(0, int(x1 + bw * 0.20))
    tx2 = min(w, int(x2 - bw * 0.20))
    ty1 = max(0, int(y1 + bh * 0.18))
    ty2 = min(h, int(y1 + bh * 0.62))
    if tx2 <= tx1 or ty2 <= ty1:
        return 0.0
    roi = hsv[ty1:ty2, tx1:tx2]
    mask = cv2.inRange(roi, hsv_low, hsv_high)
    return float(np.count_nonzero(mask)) / max(1, mask.size)


def choose_target(frames, model, args):
    results = model.predict(
        [str(p) for p in frames],
        imgsz=args.imgsz,
        conf=args.conf,
        classes=[0],
        verbose=False,
    )
    last_cx = None
    last_box = None
    selected = []
    centers = []

    for idx, (frame, result) in enumerate(zip(frames, results)):
        img = cv2.imread(str(frame))
        height, width = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        candidates = []
        if result.boxes is not None:
            for b in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf[0])
                bw, bh = x2 - x1, y2 - y1
                if bw < args.min_box_width or bh < args.min_box_height:
                    continue
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                color = torso_color_ratio(hsv, (x1, y1, x2, y2), args.hsv_low, args.hsv_high)
                side_bonus = cx if args.prefer == "right" else (width - cx if args.prefer == "left" else 0)
                if last_cx is None:
                    score = color * args.color_weight + side_bonus * args.side_weight + conf * args.conf_weight
                else:
                    score = (
                        color * args.color_weight
                        - abs(cx - last_cx) * args.continuity_weight
                        + conf * args.conf_weight
                    )
                candidates.append((score, x1, y1, x2, y2, cx, cy, conf, color))
        if candidates:
            score, x1, y1, x2, y2, cx, cy, conf, color = max(candidates, key=lambda v: v[0])
        elif last_box is not None:
            x1, y1, x2, y2, cx, cy, conf, color = last_box
            score = -9999.0
        else:
            cx = width * (0.72 if args.prefer == "right" else 0.5)
            cy = height / 2
            x1, y1, x2, y2, conf, color, score = cx - 80, 120, cx + 80, height - 40, 0.0, 0.0, -9999.0
        last_cx = cx
        last_box = (x1, y1, x2, y2, cx, cy, conf, color)
        centers.append(float(cx))
        selected.append(
            {
                "t": idx / args.sample_fps,
                "cx": float(cx),
                "cy": float(cy),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf),
                "color_ratio": float(color),
                "score": float(score),
                "candidate_count": len(candidates),
            }
        )
    return selected, centers


def median_smooth(values, radius):
    if radius <= 0:
        return [float(v) for v in values]
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        smoothed.append(float(np.median(values[lo:hi])))
    return smoothed


def crop_positions(centers, width, height, aspect_w, aspect_h, smooth_radius):
    crop_w = int(round(height * aspect_w / aspect_h))
    crop_w = min(width, crop_w)
    max_x = width - crop_w
    smoothed = median_smooth(centers, smooth_radius)
    xs = [float(max(0, min(max_x, c - crop_w / 2))) for c in smoothed]
    return crop_w, smoothed, xs


def hermite_expr(xs, dt, max_velocity):
    velocities = []
    for i, x in enumerate(xs):
        if i == 0:
            v = (xs[1] - xs[0]) / dt if len(xs) > 1 else 0.0
        elif i == len(xs) - 1:
            v = (xs[-1] - xs[-2]) / dt
        else:
            v = (xs[i + 1] - xs[i - 1]) / (2 * dt)
        velocities.append(float(max(-max_velocity, min(max_velocity, v))))

    def segment(i):
        x0, x1 = xs[i], xs[i + 1]
        m0, m1 = velocities[i], velocities[i + 1]
        t0 = i * dt
        u = f"((t-{t0:.3f})/{dt:.3f})"
        u2 = f"({u}*{u})"
        u3 = f"({u2}*{u})"
        h00 = f"(2*{u3}-3*{u2}+1)"
        h10 = f"({u3}-2*{u2}+{u})"
        h01 = f"(-2*{u3}+3*{u2})"
        h11 = f"({u3}-{u2})"
        return f"({h00}*{x0:.2f}+{h10}*{dt*m0:.2f}+{h01}*{x1:.2f}+{h11}*{dt*m1:.2f})"

    expr = f"{xs[-1]:.2f}"
    for i in range(len(xs) - 2, -1, -1):
        expr = f"if(lt(t,{(i + 1) * dt:.3f}),{segment(i)},{expr})"
    return expr, velocities


def linear_expr(xs, dt):
    expr = f"{xs[-1]:.2f}"
    for i in range(len(xs) - 2, -1, -1):
        t0 = i * dt
        t1 = (i + 1) * dt
        speed = (xs[i + 1] - xs[i]) / dt
        expr = f"if(lt(t,{t1:.3f}),{xs[i]:.2f}+({speed:.2f})*(t-{t0:.3f}),{expr})"
    return expr, []


def write_debug_sheets(frames, selected, centers, crop_x, crop_w, out_dir):
    thumbs = []
    for idx, frame in enumerate(frames):
        img = cv2.imread(str(frame))
        height = img.shape[0]
        row = selected[idx]
        x = int(round(crop_x[idx]))
        x1, y1, x2, y2 = [int(round(v)) for v in row["box"]]
        cv2.rectangle(img, (x, 0), (x + crop_w, height), (0, 255, 0), 4)
        cv2.line(img, (x + crop_w // 2, 0), (x + crop_w // 2, height), (0, 255, 255), 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (int(centers[idx]), int(row["cy"])), 10, (0, 0, 255), -1)
        cv2.putText(
            img,
            f'{row["t"]:04.1f}s c={centers[idx]:.0f} r={row["color_ratio"]:.2f}',
            (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (255, 255, 255),
            3,
        )
        thumbs.append(cv2.resize(img, (320, 180)))

    for sheet, start in enumerate(range(0, len(thumbs), 21), 1):
        canvas = np.zeros((180 * 3, 320 * 7, 3), np.uint8)
        for i, tile in enumerate(thumbs[start : start + 21]):
            r, c = divmod(i, 7)
            canvas[r * 180 : (r + 1) * 180, c * 320 : (c + 1) * 320] = tile
        cv2.imwrite(str(out_dir / f"plan_sheet_{sheet}.jpg"), canvas)


def render(src, dst, width, height, crop_w, expr, args):
    vf = (
        f"crop=w={crop_w}:h={height}:x='{expr}':y=0:exact=1,"
        f"scale={args.output_width}:{args.output_height}:flags=lanczos,setsar=1"
    )
    run(
        [
            FFMPEG,
            "-hide_banner",
            "-y",
            "-i",
            str(src),
            "-vf",
            vf,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-crf",
            str(args.crf),
            "-preset",
            args.preset,
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Reframe a video around a selected person.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--target-description", default="")
    parser.add_argument("--aspect", type=parse_aspect, default=parse_aspect("9:16"))
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--hsv-low", type=parse_triplet, default=parse_triplet("5,35,35"))
    parser.add_argument("--hsv-high", type=parse_triplet, default=parse_triplet("35,255,235"))
    parser.add_argument("--prefer", choices=["right", "left", "none"], default="right")
    parser.add_argument("--color-weight", type=float, default=5000.0)
    parser.add_argument("--continuity-weight", type=float, default=9.0)
    parser.add_argument("--side-weight", type=float, default=1.2)
    parser.add_argument("--conf-weight", type=float, default=120.0)
    parser.add_argument("--min-box-width", type=float, default=25.0)
    parser.add_argument("--min-box-height", type=float, default=90.0)
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument("--camera", choices=["hermite", "linear"], default="hermite")
    parser.add_argument("--max-velocity", type=float, default=260.0)
    parser.add_argument("--output-width", type=int)
    parser.add_argument("--output-height", type=int)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args()
    if args.output_width is None or args.output_height is None:
        args.output_width, args.output_height = default_output_size(args.aspect)

    src = args.input.expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    dst = args.output.expanduser().resolve() if args.output else src.with_name(src.stem + "_vertical.mp4")
    debug_dir = args.debug_dir.expanduser().resolve() if args.debug_dir else dst.with_suffix("").with_name(dst.stem + "_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    width, height, duration = probe_video(src)
    frames = extract_frames(src, debug_dir / "frames", args.sample_fps)
    if len(frames) < 2:
        raise SystemExit("not enough sampled frames")

    model = load_yolo(args.model)
    selected, raw_centers = choose_target(frames, model, args)
    aspect_w, aspect_h = args.aspect
    crop_w, centers, xs = crop_positions(raw_centers, width, height, aspect_w, aspect_h, args.smooth_radius)

    dt = 1.0 / args.sample_fps
    if args.camera == "hermite":
        expr, velocities = hermite_expr(xs, dt, args.max_velocity)
    else:
        expr, velocities = linear_expr(xs, dt)

    plan = {
        "input": str(src),
        "output": str(dst),
        "target_description": args.target_description,
        "aspect": f"{aspect_w}:{aspect_h}",
        "duration": duration,
        "width": width,
        "height": height,
        "sample_fps": args.sample_fps,
        "camera": args.camera,
        "crop_width": crop_w,
        "selected": selected,
        "centers": centers,
        "crop_x": xs,
        "velocity_px_per_s": velocities,
    }
    (debug_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    (debug_dir / "crop_expr.txt").write_text(expr)
    write_debug_sheets(frames, selected, centers, xs, crop_w, debug_dir)
    render(src, dst, width, height, crop_w, expr, args)
    print(dst)
    print(debug_dir)


if __name__ == "__main__":
    main()
