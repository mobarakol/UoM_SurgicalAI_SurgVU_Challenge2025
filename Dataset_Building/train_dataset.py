# #!/usr/bin/env python3
"""
Batch random 30-second crops from case videos using per-case tasks.csv annotations
and tool presence from tools.csv. Also writes a per-clip labels CSV with the format:

    Tools | Yes_No | Task | Yes_No.1

What's new in this version
--------------------------
- **Removes "other" task clips** entirely: rows whose task normalizes to "other" are skipped.
- Per-clip label CSV is limited to the **7 tasks** and **12 instruments** in the presets.
- Prints the **output clip paths** for any clips whose labels are **not** in the presets
  (unknown task or unknown tools). Also prints unique unknown task/tool sets seen.
- Still ignores "suction irrigator" and "nan(camera in)" (and bare "nan").
- Canonicalizes "large needle driver" → "needle driver" everywhere.

Paths are hard-coded to your cluster directories. Uses FFmpeg (no MoviePy).
"""

import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# --------------------------- Config (edit these) --------------------------- #
CASES_ROOT = Path("Path to your target/cases_root")
VIDEOS_ROOT = Path("Path to your target/videos_root")
OUT_ROOT = Path("Path to your target/out_root")
CLIPS_PER_CSV = 2
RANDOM_SEED = 42

# --------------------------- Helpers -------------------------------------- #

def _to_seconds(ts) -> Optional[float]:
    if pd.isna(ts):
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2}(?:\.\d{1,3})?)", s)
    if m:
        h, m_, s_ = m.group(1), m.group(2), m.group(3)
        return int(h) * 3600 + int(m_) * 60 + float(s_)
    m = re.fullmatch(r"(\d{1,2}):(\d{2}(?:\.\d{1,3})?)", s)
    if m:
        m_, s_ = m.group(1), m.group(2)
        return int(m_) * 60 + float(s_)
    return None

def _norm_colname(name: str) -> str:
    return name.replace("\ufeff", "").strip().lower().replace(" ", "_")

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None

def _pad3(x: int) -> str:
    return f"{int(x):03d}"

def _normalize_label(s: Optional[str]) -> Optional[str]:
    """Lower-case + collapse spaces; return None if empty."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    t = " ".join(str(s).strip().split()).lower()
    return t if t else None

# Regex to ignore nan(camera in) regardless of case/spacing/parentheses
_IGNORE_NAN_CAMERA_RE = re.compile(r"^\s*nan\s*\(?\s*camera\s*in\s*\)?\s*$", re.IGNORECASE)

def _clean_tool_name(val) -> Optional[str]:
    """
    Return a normalized tool name (lowercase, trimmed) or None to ignore.
    Ignores: 'nan', 'nan(camera in)', and 'suction irrigator'.
    Canonicalizes synonyms: 'large needle driver' -> 'needle driver'.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = " ".join(str(val).strip().split())
    if not s:
        return None
    # ignore placeholders
    if s.lower() == "nan" or _IGNORE_NAN_CAMERA_RE.fullmatch(s):
        return None
    s_norm = s.lower()
    if s_norm == "suction irrigator":
        return None
    # synonym unification: any variant that contains "needle driver" becomes canonical
    if "needle driver" in s_norm:
        s_norm = "needle driver"
    return s_norm

# --------------------------- Preset vocab (strict) ------------------------- #
# Instruments (12)
TOOLS_PRESET = [
    "needle driver", "monopolar curved scissors", "force bipolar", "clip applier",
    "cadiere forceps", "bipolar forceps", "vessel sealer",
    "permanent cautery hook/spatula", "prograsp forceps", "stapler",
    "grasping retractor", "tip-up fenestrated grasper",
]
ALLOWED_TOOLS = set(TOOLS_PRESET)

# Tasks (7) – "other" is intentionally excluded
TASKS_PRESET = [
    "suturing", "uterine horn", "suspensory ligaments", "rectal artery/vein",
    "skills application", "range of motion", "retraction and collision avoidance",
]
ALLOWED_TASKS = set(TASKS_PRESET)

@dataclass
class ClipPlan:
    case_id: str
    csv_path: Path
    video_path: Path
    start_part: Optional[int]
    row_index: int
    clip_start: float
    clip_end: float
    task: str               # original as in CSV (not normalized)
    match_desc: str
    out_path: Path

# --------------------------- Video discovery ------------------------------ #
ALLOWED_SUFFIXES = {".mp4", ".MP4", ""}

def _case_num_strings(case_id: str) -> Tuple[str, str]:
    m = re.search(r"(\d+)$", case_id)
    if not m:
        return case_id, case_id
    num = m.group(1)
    return num.zfill(3), str(int(num))

def find_video_file(videos_root: Path, case_id: str, part: Optional[int]) -> Optional[Path]:
    """Find the appropriate video file for a case and part."""
    n3, n_plain = _case_num_strings(case_id)
    subdir_candidates = [
        case_id,
        f"case_{n3}",
        f"case_{n_plain}",
        f"case{n3}",
        f"case{n_plain}",
    ]
    existing_dirs = [videos_root / s for s in subdir_candidates if (videos_root / s).is_dir()]
    if not existing_dirs:
        print(f"[WARN] No video directory found for {case_id} under {videos_root}. Tried: {subdir_candidates}")
        return None
    vd = existing_dirs[0]

    def _glob_one(pattern: str) -> Optional[Path]:
        for p in vd.glob(pattern):
            if p.is_file() and p.suffix in ALLOWED_SUFFIXES:
                return p
        return None

    if part is not None:
        p3 = _pad3(part)
        patterns = [
            f"case_{n3}_video_part_{p3}*",
            f"case_{n_plain}_video_part_{p3}*",
            f"case{n3}_video_part_{p3}*",
            f"case{n_plain}_video_part_{p3}*",
            f"*part*{p3}*",
        ]
        for pat in patterns:
            found = _glob_one(pat)
            if found:
                return found
        print(f"[WARN] Could not find part {part} video in {vd} for {case_id}")
        return None

    default = _glob_one(f"case_{n3}_video_part_001*") or _glob_one(f"*part*001*")
    if default:
        return default

    vids = [p for p in vd.iterdir() if p.is_file() and p.suffix in ALLOWED_SUFFIXES]
    if len(vids) == 1:
        return vids[0]

    fallback = _glob_one(f"case_{n3}_video_part_*") or _glob_one(f"*.mp4")
    if fallback:
        return fallback

    print(f"[WARN] No video files found in {vd} for {case_id}")
    return None

# --------------------------- FFmpeg clipping ------------------------------- #

def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ERROR: 'ffmpeg' not found on PATH. Please install FFmpeg.")

def ffmpeg_extract_clip(src: Path, start_s: float, end_s: float, dst: Path) -> None:
    duration = max(0.0, end_s - start_s)
    if duration < 29.95:
        raise RuntimeError("Clip duration < 30s after bounds check")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src} -> {dst}:\n{proc.stderr.decode(errors='ignore')}")

# --------------------------- Tools handling -------------------------------- #

def load_tools_df(case_dir: Path) -> Optional[pd.DataFrame]:
    tools_path = case_dir / "tools.csv"
    if not tools_path.exists():
        print(f"[INFO] {case_dir.name} has no tools.csv; skipping tool detection.")
        return None
    df = pd.read_csv(tools_path)
    df.columns = [_norm_colname(c) for c in df.columns]

    tool_col = _find_col(df, [
        "groundtruth_toolname", "ground_truth_toolname", "toolname", "tool_name", "instrument", "name"
    ])
    install_col = _find_col(df, [
        "install_case_time", "install_time", "start_time", "begin", "onset"
    ])
    uninstall_col = _find_col(df, [
        "uninstall_case_time", "uninstall_time", "stop_time", "end_time", "finish", "finish_time"
    ])
    arm_col = _find_col(df, ["arm", "robot_arm", "arm_id", "arm_name"])  # optional

    if not tool_col or not install_col or not uninstall_col:
        print(f"[WARN] {tools_path} missing required tool/time columns. Available: {list(df.columns)}")
        return None

    # Normalize and parse times
    keep_cols = [c for c in [tool_col, install_col, uninstall_col, arm_col] if c is not None]
    df = df[keep_cols].copy()
    df.rename(columns={tool_col: "tool", install_col: "install", uninstall_col: "uninstall", (arm_col or "arm"): "arm"}, inplace=True)
    df["install_s"] = df["install"].apply(_to_seconds)
    df["uninstall_s"] = df["uninstall"].apply(_to_seconds)

    # Handle open-ended uninstall times (NaN -> large number)
    df["uninstall_s"].fillna(10**12, inplace=True)

    # Clean tool names (drop ignored items early; normalize + canonicalize)
    df["tool"] = df["tool"].apply(_clean_tool_name)
    df = df[df["tool"].notna()].copy()

    return df

def tools_in_window(tools_df: pd.DataFrame, start_s: float, end_s: float) -> Tuple[List[str], Dict[str, List[str]]]:
    """Return (unique_tool_names, tools_by_arm) overlapping [start_s, end_s]."""
    if tools_df is None or tools_df.empty:
        return [], {}
    mask = (tools_df["install_s"] < end_s) & (tools_df["uninstall_s"] > start_s)
    if not mask.any():
        return [], {}
    present = tools_df.loc[mask]

    # Names (unique, cleaned)
    names = sorted({n for n in present["tool"].apply(_clean_tool_name) if n})

    # By arm (dedup per arm)
    by_arm: Dict[str, List[str]] = defaultdict(list)
    for _, row in present.iterrows():
        arm = str(row.get("arm")) if not pd.isna(row.get("arm")) else "unknown"
        nm = _clean_tool_name(row.get("tool"))
        if nm:
            by_arm[arm].append(nm)
    by_arm = {arm: sorted(set(vals)) for arm, vals in by_arm.items()}

    return names, by_arm

# --------------------------- Planning & rendering -------------------------- #

@dataclass
class ClipPlan:
    case_id: str
    csv_path: Path
    video_path: Path
    start_part: Optional[int]
    row_index: int
    clip_start: float
    clip_end: float
    task: str
    match_desc: str
    out_path: Path

def plan_random_clips_for_csv(
    csv_path: Path,
    case_id: str,
    videos_root: Path,
    clips_per_csv: int,
) -> List[ClipPlan]:
    df = pd.read_csv(csv_path)
    df.columns = [_norm_colname(c) for c in df.columns]

    start_col = _find_col(df, ["start_time"])  # strict per your dataset
    end_col = _find_col(df, ["stop_time"])    # strict per your dataset

    task_col = _find_col(df, [
        "groundtruth_taskname", "ground_truth_taskname", "taskname", "task_name", "task", "label", "task_label"
    ]) or ""
    match_col = _find_col(df, [
        "matched_description", "match_description", "matched_desc", "match_desc", "match", "description"
    ]) or ""
    part_col = _find_col(df, ["start_part", "part", "video_part", "segment_part"])  

    if not start_col or not end_col:
        print(f"[WARN] {csv_path} missing time columns. Available columns: {list(df.columns)}")
        return []

    rows = []
    for idx, r in df.iterrows():
        t0 = _to_seconds(r[start_col])
        t1 = _to_seconds(r[end_col])
        if t0 is None or t1 is None:
            continue

        # Task value (may be missing)
        task_val = ""
        if task_col and not pd.isna(r.get(task_col)):
            task_val = str(r[task_col])
        tnorm = _normalize_label(task_val)

        # ----- skip "other" tasks entirely -----
        if tnorm == "other":
            continue

        if t1 - t0 >= 30.0:
            match_val = ""
            if match_col and not pd.isna(r.get(match_col)):
                match_val = str(r[match_col])
            part_val = None
            if part_col and not pd.isna(r.get(part_col)):
                try:
                    part_val = int(r[part_col])
                except Exception:
                    part_val = None
            rows.append((idx, t0, t1, task_val, match_val, part_val))

    if not rows:
        print(f"[INFO] {csv_path} has no eligible >=30s segments after filtering. Skipping.")
        return []

    chosen = [random.choice(rows) for _ in range(clips_per_csv)]

    plans: List[ClipPlan] = []
    for (row_index, t0, t1, task_val, match_val, part_val) in chosen:
        max_offset = max(0.0, (t1 - t0) - 30.0)
        rnd_offset = random.random() * max_offset
        clip_start = t0 + rnd_offset
        clip_end = clip_start + 30.0

        video_path = find_video_file(videos_root, case_id, part_val)
        if video_path is None:
            print(f"[WARN] No video found for {case_id} (CSV: {csv_path.name}, row {row_index}). Skipping clip.")
            continue

        plans.append(
            ClipPlan(
                case_id=case_id,
                csv_path=csv_path,
                video_path=video_path,
                start_part=part_val,
                row_index=row_index,
                clip_start=clip_start,
                clip_end=clip_end,
                task=task_val,          # keep original in manifest
                match_desc=match_val,
                out_path=Path(),
            )
        )

    return plans

# --------------------------- Per-clip label CSV ---------------------------- #

def build_clip_label_df(tools_present: List[str], task_name: str) -> pd.DataFrame:
    """
    Build a DataFrame like your example:
        Tools | Yes_No | Task | Yes_No.1
    using ONLY the preset vocabularies (strict).
    """
    ordered_tools = list(TOOLS_PRESET)
    ordered_tasks = list(TASKS_PRESET)

    n = max(len(ordered_tools), len(ordered_tasks))
    tools_present_set = {(_clean_tool_name(t) or "") for t in tools_present}
    tools_present_set = {t for t in tools_present_set if t}  # drop empties

    task_norm = _normalize_label(task_name) or ""

    rows = []
    for i in range(n):
        tool_name = ordered_tools[i] if i < len(ordered_tools) else None
        tool_yes = 1 if (tool_name and tool_name in tools_present_set) else (0 if tool_name else None)
        task = ordered_tasks[i] if i < len(ordered_tasks) else None
        task_yes = 1 if (task and task == task_norm) else (0 if task else None)
        rows.append({"Tools": tool_name, "Yes_No": tool_yes, "Task": task, "Yes_No.1": task_yes})

    return pd.DataFrame(rows)

# --------------------------- Rendering ------------------------------------ #

def render_clip(plan: ClipPlan, out_dir: Path) -> Optional[ClipPlan]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_stem = plan.csv_path.stem
    part_suffix = f"_part{_pad3(plan.start_part)}" if plan.start_part is not None else ""
    out_name = f"{plan.case_id}__{csv_stem}{part_suffix}__row{plan.row_index}__{int(plan.clip_start)}-{int(plan.clip_end)}.mp4"
    out_path = out_dir / out_name

    try:
        ffmpeg_extract_clip(plan.video_path, plan.clip_start, plan.clip_end, out_path)
    except Exception as e:
        print(f"[ERROR] Failed to export {out_path.name}: {e}")
        return None

    plan.out_path = out_path
    return plan

# --------------------------- Main ----------------------------------------- #

def discover_case_dirs(cases_root: Path) -> List[Path]:
    return sorted([p for p in cases_root.iterdir() if p.is_dir() and re.match(r"case_\d+", p.name)])

def main():
    if shutil.which("ffmpeg") is None:
        sys.exit("ERROR: 'ffmpeg' not found on PATH. Please install FFmpeg.")
    random.seed(RANDOM_SEED)

    manifest_rows = []
    out_clips_root = OUT_ROOT / "clips_v2"
    out_clips_root.mkdir(parents=True, exist_ok=True)

    # Counters (normalized keys) & unknown label tracking
    task_counter: Counter = Counter()
    tool_counter: Counter = Counter()
    # per-clip unknowns: clip_path -> dict with unknown_task and unknown_tools
    unknown_by_clip: List[Dict[str, object]] = []
    unknown_tasks_set = set()
    unknown_tools_set = set()

    case_dirs = discover_case_dirs(CASES_ROOT)
    if not case_dirs:
        print(f"[WARN] No case_* directories found in {CASES_ROOT}")

    for case_dir in case_dirs:
        case_id = case_dir.name
        csv_path = case_dir / "tasks.csv"
        if not csv_path.exists():
            print(f"[INFO] {case_id} has no tasks.csv; skipping.")
            continue

        # Load tools for this case (if present)
        tools_df = load_tools_df(case_dir)

        plans = plan_random_clips_for_csv(
            csv_path=csv_path,
            case_id=case_id,
            videos_root=VIDEOS_ROOT,
            clips_per_csv=CLIPS_PER_CSV,
        )
        if not plans:
            continue

        case_out_dir = out_clips_root / case_id
        for plan in plans:
            completed = render_clip(plan, case_out_dir)
            if not completed:
                continue

            # Tools present in this 30s window (already cleaned/lowercased)
            tool_names, tools_by_arm = tools_in_window(tools_df, completed.clip_start, completed.clip_end)

            # Normalized task
            tnorm = _normalize_label(completed.task)

            # --------- counts (only for allowed vocab) ----------
            if tnorm and tnorm in ALLOWED_TASKS:
                task_counter[tnorm] += 1
            # tools: count only allowed ones, once per clip
            for nm in set(tool_names):
                if nm in ALLOWED_TOOLS:
                    tool_counter[nm] += 1

            # --------- track unknown labels per clip for reporting ----------
            clip_unknown_task = None
            if tnorm and tnorm not in ALLOWED_TASKS:
                clip_unknown_task = tnorm
                unknown_tasks_set.add(tnorm)

            clip_unknown_tools = sorted([nm for nm in set(tool_names) if nm not in ALLOWED_TOOLS])
            for nm in clip_unknown_tools:
                unknown_tools_set.add(nm)

            if clip_unknown_task or clip_unknown_tools:
                unknown_by_clip.append({
                    "output_clip": str(completed.out_path),
                    "unknown_task": clip_unknown_task,
                    "unknown_tools": clip_unknown_tools,
                })

            # --------- per-clip labels CSV (strict to presets) ----------
            labels_df = build_clip_label_df(tool_names, completed.task)
            labels_csv_path = completed.out_path.with_suffix("").with_name(completed.out_path.stem + "_tools_tasks.csv")
            labels_df.to_csv(labels_csv_path, index=False)

            # Manifest row
            manifest_rows.append(
                {
                    "case_id": completed.case_id,
                    "source_csv": str(completed.csv_path),
                    "video_file": str(completed.video_path),
                    "start_part": completed.start_part,
                    "row_index": completed.row_index,
                    "clip_start_s": round(completed.clip_start, 3),
                    "clip_end_s": round(completed.clip_end, 3),
                    "task": completed.task,                 # original as in CSV
                    "task_normalized": tnorm,               # normalized (may be None or unknown)
                    "match_description": completed.match_desc,
                    "tools_present": "|".join(tool_names),  # includes any unknowns for transparency
                    "tools_by_arm": "; ".join(
                        f"{arm}: " + "|".join(names) for arm, names in sorted(tools_by_arm.items())
                    ),
                    "tools_tasks_csv": str(labels_csv_path),
                    "output_clip": str(completed.out_path),
                }
            )

    # Write manifest CSV
    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path = OUT_ROOT / "cropped_30s_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        print(f"[OK] Wrote manifest: {manifest_path}")

        # ----- Print the requested counts (allowed vocab only) -----
        if task_counter:
            print("\nCounts per groundtruth_taskname (normalized, allowed vocab only):")
            for name, cnt in task_counter.most_common():
                print(f"  {name}: {cnt}")
        else:
            print("\nNo allowed task names found to count.")

        if tool_counter:
            print("\nCounts per groundtruth_toolname (allowed vocab only):")
            for name, cnt in tool_counter.most_common():
                print(f"  {name}: {cnt}")
        else:
            print("\nNo allowed tools found to count.")

        # ----- Print clips containing labels outside the presets -----
        if unknown_by_clip:
            print("\nClips with labels outside preset vocab (7 tasks / 12 instruments):")
            for rec in unknown_by_clip:
                clip = rec["output_clip"]
                utask = rec["unknown_task"]
                utools = rec["unknown_tools"]
                parts = []
                if utask:
                    parts.append(f"unknown task='{utask}'")
                if utools:
                    parts.append(f"unknown tools={utools}")
                print(f"  {clip}  -->  " + "; ".join(parts))

            if unknown_tasks_set:
                print("\nUnique unknown task labels (normalized):")
                for t in sorted(unknown_tasks_set):
                    print(f"  - {t}")
            if unknown_tools_set:
                print("\nUnique unknown tool labels (normalized):")
                for t in sorted(unknown_tools_set):
                    print(f"  - {t}")
        else:
            print("\nNo unknown task/tool labels encountered.")
    else:
        print("[INFO] No clips were generated; manifest not written.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

