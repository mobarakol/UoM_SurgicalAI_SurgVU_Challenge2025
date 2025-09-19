#!/usr/bin/env python3
"""
Batch-blur rectangular regions in many videos.

Directory layout example:
root/
  case011/
    case011_1.mp4
    case011_2.mp4
  case122/
    case122.mp4
  ...

What this script does:
- Walks the immediate subfolders of ROOT_DIR.
- In each subfolder, finds ALL input .mp4 files (case-insensitive).
- Applies Gaussian blur to the configured regions.
- Writes "<stem>_final.mp4" beside each input.

Requires:
- ffmpeg installed and on PATH.

Blur logic:
- If Region.w is None -> use full input width (iw).
- If Region.y is None -> align to bottom (y = ih - h).
"""

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# =============== USER SETTINGS ===============================================

# Set this to your "root" directory that contains the case folders.
#ROOT_DIR = Path("/SAN/medic/CARES/mobarak/SurgVU/SURGVU25_cat_2_sample_set_public")
ROOT_DIR = Path("Path to your target/root_directory")

# If True, skip work when the output file already exists.
SKIP_IF_OUTPUT_EXISTS = True

# Output filename suffix (placed before .mp4)
OUTPUT_SUFFIX = "_final"

# Video encoding settings
VIDEO_PRESET = "medium"
VIDEO_CRF = "18"

@dataclass
class Region:
    # Rectangle to blur. If w is None -> full width (iw).
    # If y is None -> align to bottom (ih - h).
    x: Optional[int]
    y: Optional[int]
    w: Optional[int]
    h: int
    # Blur strength (Gaussian sigma)
    sigma: float = 5.0
    # Optional time window (in seconds) during which the blur is active
    start: Optional[float] = None
    end: Optional[float] = None

# Example: bottom 70 px strip across full width, any resolution
REGIONS: List[Region] = [
    #            x    y      w     h   sigma  start end
    Region(0,   None,  None,  70,  20.0, None, None),
]

# ============================================================================

def build_filter(regions: List[Region]) -> str:
    if not regions:
        # no-op (still pass through to keep code simple)
        return "[0:v]format=yuv420p[vout]"

    n = len(regions)
    # Split source into (n+1) copies: base + one branch per region
    labels_split = "[0:v]split=" + str(n + 1) + "[base]" + "".join(f"[s{i}]" for i in range(n))
    parts = [labels_split]

    # For each region: blur -> crop to ROI (using expressions to stay in-bounds)
    for i, r in enumerate(regions):
        w_expr = "iw" if r.w is None else str(r.w)
        h_expr = str(r.h)
        x_expr = "0" if r.x is None else str(r.x)
        # If y is None, anchor to bottom; else use provided y
        y_crop_expr = f"ih-{h_expr}" if r.y is None else str(r.y)

        parts.append(f"[s{i}]gblur=sigma={r.sigma}[b{i}]")
        parts.append(f"[b{i}]crop=w={w_expr}:h={h_expr}:x={x_expr}:y={y_crop_expr}[c{i}]")

    # Sequentially overlay each blurred crop onto the base
    prev = "base"
    for i, r in enumerate(regions):
        enable = ""
        if r.start is not None and r.end is not None:
            enable = f":enable='between(t,{r.start},{r.end})'"

        # Match the crop placement: bottom-align if y is None, else use y
        y_overlay_expr = "main_h-overlay_h" if r.y is None else str(r.y)
        x_overlay_expr = "0" if r.x is None else str(r.x)

        parts.append(f"[{prev}][c{i}]overlay=x={x_overlay_expr}:y={y_overlay_expr}{enable}[o{i}]")
        prev = f"o{i}"

    # Ensure standard pixel format for broad compatibility
    parts.append(f"[{prev}]format=yuv420p[vout]")

    return ";".join(parts)

def iter_input_mp4s(folder: Path) -> List[Path]:
    """
    Return ALL candidate .mp4 inputs in this folder (case-insensitive),
    excluding files that already look like outputs (ending with OUTPUT_SUFFIX).
    Prefer "<foldername>.mp4" first if present, but process all.
    """
    # All .mp4 files (case-insensitive)
    mp4s = sorted([p for p in folder.glob("*.mp4")] + [p for p in folder.glob("*.MP4")])
    # Exclude outputs (avoid reprocessing)
    mp4s = [p for p in mp4s if not p.name.endswith(f"{OUTPUT_SUFFIX}.mp4")]

    # Prefer exact foldername.mp4 first if present
    preferred = folder / f"{folder.name}.mp4"
    if preferred in mp4s:
        # Put preferred at the front, keep others afterwards
        others = [p for p in mp4s if p != preferred]
        return [preferred] + others

    return mp4s

def output_path_for_input(input_mp4: Path) -> Path:
    return input_mp4.with_name(f"{input_mp4.stem}{OUTPUT_SUFFIX}.mp4")

def run_ffmpeg(input_mp4: Path, output_mp4: Path, filter_complex: str) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-stats",
        "-y",                  # overwrite (we'll guard at caller if skipping)
        "-i", str(input_mp4),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",        # keep audio if present
        "-c:v", "libx264",
        "-preset", VIDEO_PRESET,
        "-crf", VIDEO_CRF,
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_mp4),
    ]
    print("Running:\n ", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)

def process_root(root_dir: Path, regions: List[Region]) -> Tuple[int, int, int]:
    """
    Returns (processed_count, skipped_count, missing_count)
    """
    processed = skipped = missing = 0
    filter_complex = build_filter(regions)

    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue

        inputs = iter_input_mp4s(entry)
        if not inputs:
            print(f"[MISS] No input .mp4 found in: {entry}")
            missing += 1
            continue

        for input_mp4 in inputs:
            output_mp4 = output_path_for_input(input_mp4)

            if SKIP_IF_OUTPUT_EXISTS and output_mp4.exists():
                print(f"[SKIP] Output already exists: {output_mp4}")
                skipped += 1
                continue

            output_mp4.parent.mkdir(parents=True, exist_ok=True)
            try:
                run_ffmpeg(input_mp4, output_mp4, filter_complex)
                print(f"[OK] Wrote: {output_mp4}")
                processed += 1
            except subprocess.CalledProcessError as e:
                print(f"[ERR] ffmpeg failed for {input_mp4} -> {output_mp4}: {e}")

    return processed, skipped, missing

def main():
    if not ROOT_DIR.exists() or not ROOT_DIR.is_dir():
        raise SystemExit(f"ROOT_DIR does not exist or is not a directory: {ROOT_DIR}")

    print(f"Scanning cases in: {ROOT_DIR}")
    processed, skipped, missing = process_root(ROOT_DIR, REGIONS)
    print(f"\nSummary: processed={processed}, skipped={skipped}, missing_input={missing}")

if __name__ == "__main__":
    main()
