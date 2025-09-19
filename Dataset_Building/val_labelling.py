#!/usr/bin/env python3
"""
Build per-video Q&A text files from per-clip CSVs, enforce QA names with _1, _2, ...
and remove unrelated .txt files. Also rename final MP4s to include the index:
  caseXXX_final.mp4 or caseXXX_finial.mp4  ->  caseXXX_1_final.mp4  (for the 1st video)
  caseXXX_2_final.mp4 is left as-is (we only fix/migrate missing/typo'd index names).

Layout example:
root/
  case011/
    case011_1.mp4
    case011_2.mp4
    tools_tasks_case011_1.csv
    tools_tasks_case011_2.csv
    case011_final.mp4        -> will be renamed to case011_1_final.mp4
    case011_2_final.mp4      -> kept as-is
    -> qa_case011_1.txt
    -> qa_case011_2.txt
"""

from pathlib import Path
import re
import numbers
import pandas as pd

# ========= CONFIG =========
ROOT_DIR = Path("Path to your target/root_directory")
# exclude outputs like *_final.mp4 and also misspelled *_finial.mp4
EXCLUDE_MP4_SUFFIXES = ["_final", "_finial"]
# ==========================

# Forceps tool names to detect (case-insensitive)
FORCEPS_ALIASES = {
    "bipolar forceps",
    "cadiere forceps",
    "prograsp forceps",  # common spelling
    "prograp forceps",   # variant spelling sometimes seen
}

def normalize_string(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def coerce_numeric_01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(0).apply(lambda v: 1 if float(v) >= 0.5 else 0).astype(int)

def is_positive(v) -> bool:
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    if isinstance(v, numbers.Number):
        return float(v) >= 0.5
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    try:
        return float(s) >= 0.5
    except Exception:
        return False

def find_adjacent_value_column(df: pd.DataFrame, left_col_name: str) -> str:
    cols = [str(c) for c in df.columns]
    try:
        idx = cols.index(left_col_name)
    except ValueError:
        raise ValueError(f"Expected column '{left_col_name}' not found in CSV. Columns: {cols}")
    if idx + 1 >= len(cols):
        raise ValueError(f"No column to the right of '{left_col_name}'. Columns: {cols}")
    return cols[idx + 1]

def build_procedure_qas() -> list[str]:
    return [
        "What procedure is this summary describing? | The summary is describing endoscopic or laparoscopic surgery."
    ]

def build_tool_qas(df: pd.DataFrame) -> list[str]:
    tools_col = "Tools"
    val_col = find_adjacent_value_column(df, tools_col)
    df[val_col] = coerce_numeric_01(df[val_col])
    qas = []
    for _, row in df.iterrows():
        tool_name = normalize_string(row.get(tools_col, ""))
        if not tool_name:
            continue
        present = is_positive(row.get(val_col, 0))
        if present:
            qas.append(f"Is a {tool_name} among the listed tools? | Yes, a {tool_name} is listed.")
            qas.append(f"Was a {tool_name} used in this clip? | Yes, a {tool_name} was utilized.")
            qas.append(f"Was a {tool_name} used during the surgery? | Yes, a {tool_name} was used.")
        else:
            qas.append(f"Is a {tool_name} among the listed tools? | No, a {tool_name} is not listed.")
            qas.append(f"Was a {tool_name} used in this clip? | No, a {tool_name} was not utilized.")
            qas.append(f"Was a {tool_name} used during the surgery? | No, a {tool_name} was not used.")
    return qas

def build_forceps_qas(df: pd.DataFrame) -> list[str]:
    tools_col = "Tools"
    val_col = find_adjacent_value_column(df, tools_col)
    df[val_col] = coerce_numeric_01(df[val_col])
    present_forceps = []
    for _, row in df.iterrows():
        raw_name = normalize_string(row.get(tools_col, ""))
        tool_name = raw_name.lower()
        if tool_name in FORCEPS_ALIASES and is_positive(row.get(val_col, 0)):
            present_forceps.append(raw_name)
    qas = []
    if present_forceps:
        qas.append("Are there forceps being used here? | Yes, forceps are mentioned.")
        types_str = ", ".join(dict.fromkeys(present_forceps))
        qas.append(f"What type of forceps are used? | {types_str}")
    else:
        qas.append("Are there forceps being used here? | No, forceps are not mentioned.")
    return qas

def build_task_qas(df: pd.DataFrame) -> list[str]:
    if "Task" not in df.columns:
        return []
    task_val_col = find_adjacent_value_column(df, "Task")
    df[task_val_col] = coerce_numeric_01(df[task_val_col])
    flagged = []
    for _, row in df.iterrows():
        task_name = normalize_string(row.get("Task", ""))
        if task_name and is_positive(row.get(task_val_col, 0)):
            flagged.append(task_name)
    unique_tasks = list(dict.fromkeys(flagged))
    answer = ", ".join(unique_tasks) if unique_tasks else "None"
    return [f"What task is being performed in this clip? | {answer}"]

# ---------- Matching CSV to a video ----------
_norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())

def find_csv_for_video(case_dir: Path, video_stem: str) -> Path | None:
    """
    Pick the most relevant CSV for a video named <video_stem>.mp4 in case_dir.
    Preference order (case-insensitive):
      1) *<video_stem>*tools_tasks*.csv
      2) *<video_stem>*.csv
      3) tools_tasks*.csv
      4) first *.csv
    """
    stem_norm = _norm(video_stem)
    csvs = sorted(case_dir.glob("*.csv"))
    if not csvs:
        return None

    def contains_token(paths, token):
        return [p for p in paths if token in _norm(p.stem)]

    pri1 = [p for p in contains_token(csvs, stem_norm) if "toolstasks" in _norm(p.stem)]
    if pri1:
        return pri1[0]
    pri2 = contains_token(csvs, stem_norm)
    if pri2:
        return pri2[0]
    pri3 = [p for p in csvs if "toolstasks" in _norm(p.stem)]
    if pri3:
        return pri3[0]
    return csvs[0]

# ---------- Video enumeration & naming ----------
def natural_key(p: Path):
    """Natural sort key: splits digits and text so case011_2 < case011_10."""
    s = p.stem.lower()
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def iter_input_videos(case_dir: Path):
    vids = []
    for ext in ("*.mp4", "*.MP4"):
        vids.extend(case_dir.glob(ext))
    videos = [v for v in vids if not any(v.stem.endswith(suf) for suf in EXCLUDE_MP4_SUFFIXES)]
    return sorted(videos, key=natural_key)

# ---------- Final MP4 renaming ----------
def rename_final_mp4s(case_dir: Path, videos_sorted: list[Path]):
    """
    For each input video (non-final) assigned index i, ensure any matching final file is named:
      <case_name>_<i>_final.mp4
    We handle both *_final.mp4 and the misspelling *_finial.mp4.
    """
    if not videos_sorted:
        return

    case_name = case_dir.name

    # Collect all existing final files (cover both correct & misspelled)
    final_candidates = set()
    for ext in ("*.mp4", "*.MP4"):
        for p in case_dir.glob(ext):
            stem_low = p.stem.lower()
            if stem_low.endswith("_final") or stem_low.endswith("_finial"):
                final_candidates.add(p)

    if not final_candidates:
        return

    # Map each non-final video to its target final name
    for idx, v in enumerate(videos_sorted, start=1):
        target = case_dir / f"{case_name}_{idx}_final.mp4"

        # Possible old names we want to migrate
        candidates = [
            case_dir / f"{v.stem}_final.mp4",
            case_dir / f"{v.stem}_finial.mp4",          # typo variant
        ]

        # If the raw stem is like 'case011' (no _1), also look for case011_final/finial
        if v.stem == case_name:
            candidates.extend([
                case_dir / f"{case_name}_final.mp4",
                case_dir / f"{case_name}_finial.mp4",
            ])

        for old in candidates:
            if old in final_candidates and old.exists():
                if target.exists():
                    # If target already exists and old == target (maybe case-only rename), skip.
                    if old.resolve() == target.resolve():
                        continue
                    print(f"[WARN] Target exists, skipping rename: {old.name} -> {target.name}")
                    continue
                try:
                    old.rename(target)
                    print(f"[REN] {old.name} -> {target.name}")
                    # update set so we don't try to rename the same file twice
                    final_candidates.discard(old)
                    final_candidates.add(target)
                except Exception as e:
                    print(f"[WARN] Failed to rename {old.name} -> {target.name}: {e}")

    # Also repair any remaining *_finial.mp4 (typo) that already have an index:
    for p in list(final_candidates):
        stem_low = p.stem.lower()
        if stem_low.endswith("_finial"):
            fixed = case_dir / (p.stem[:-7] + "_final.mp4")  # replace suffix
            if fixed.exists() and fixed.resolve() != p.resolve():
                print(f"[WARN] Not renaming {p.name} -> {fixed.name} (target exists)")
            else:
                try:
                    p.rename(fixed)
                    print(f"[REN] {p.name} -> {fixed.name}")
                except Exception as e:
                    print(f"[WARN] Failed to rename {p.name} -> {fixed.name}: {e}")

# ---------- QA writing ----------
def write_qa_for_video(case_dir: Path, video_path: Path, out_index: int) -> Path | None:
    """
    out_index is 1-based per case, used to FORCE the QA filename: qa_<caseFolder>_<out_index>.txt
    """
    csv_path = find_csv_for_video(case_dir, video_path.stem)
    if not csv_path:
        print(f"[WARN] {case_dir.name}: No CSV found for video {video_path.name}")
        return None

    # Load CSV (or Excel fallback)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        if csv_path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(csv_path)
        else:
            raise
    df.columns = [str(c).strip() for c in df.columns]

    lines: list[str] = []
    lines.extend(build_procedure_qas())

    # Tools / Forceps (tolerate missing layout)
    try:
        lines.extend(build_tool_qas(df))
        lines.extend(build_forceps_qas(df))
    except ValueError as e:
        print(f"[WARN] {case_dir.name}/{video_path.name}: {e}")

    # Tasks (tolerate missing layout)
    try:
        lines.extend(build_task_qas(df))
    except ValueError as e:
        print(f"[WARN] {case_dir.name}/{video_path.name}: {e}")

    # FORCE the requested naming scheme: qa_<caseFolderName>_<index>.txt
    forced_stem = f"qa_{case_dir.name}_{out_index}"
    out_txt = case_dir / f"{forced_stem}.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_txt.name} (from {video_path.name}) in {case_dir.name}")
    return out_txt

def clean_other_txt(case_dir: Path, keep_txt_names: set[str]) -> None:
    for txt in case_dir.glob("*.txt"):
        if txt.name not in keep_txt_names:
            try:
                txt.unlink()
                print(f"[DEL] Removed {case_dir.name}/{txt.name}")
            except Exception as e:
                print(f"[WARN] Failed to delete {case_dir.name}/{txt.name}: {e}")

def process_case_folder(case_dir: Path):
    videos = iter_input_videos(case_dir)
    if not videos:
        print(f"[SKIP] No non-final videos in {case_dir.name}")
        # You can still fix any lingering *_finial.mp4 typo files even without inputs:
        rename_final_mp4s(case_dir, [])
        clean_other_txt(case_dir, set())
        return

    # 1) Rename final videos to include index according to the sorted input list
    rename_final_mp4s(case_dir, videos)

    # 2) Write QA files with forced _1, _2,... names
    keep = set()
    for idx, v in enumerate(videos, start=1):
        out = write_qa_for_video(case_dir, v, idx)
        if out:
            keep.add(out.name)

    # 3) Delete any other txt files in this case folder
    clean_other_txt(case_dir, keep)

def main():
    if not ROOT_DIR.exists() or not ROOT_DIR.is_dir():
        raise SystemExit(f"Root directory not found or not a directory: {ROOT_DIR}")

    for case_dir in sorted(p for p in ROOT_DIR.iterdir() if p.is_dir() and p.name.lower().startswith("case")):
        process_case_folder(case_dir)

if __name__ == "__main__":
    main()
