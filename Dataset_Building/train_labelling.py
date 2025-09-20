#!/usr/bin/env python3
"""
Modified QA generator
"""

from pathlib import Path
import re
import numbers
import pandas as pd

# ========= HARD-CODE YOUR PATHS / TEST TOGGLES HERE =========
ROOT_DIR = Path("")   # <-- change this to your actual Root path
RUN_SINGLE_CASE = False
SINGLE_CASE_NAME = "case_154"
# ============================================================

FORCEPS_ALIASES = {
    "bipolar forceps",
    "bipolar forpces",
    "cadiere forceps",
    "prograsp forceps",
    "prograp forceps",
}

# ---------- Utilities ----------

def normalize_string(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def display_tool_name(raw_name: str) -> str:
    name = normalize_string(raw_name)
    if re.search(r"(?i)\bneedle\s+driver\b", name) and not re.search(r"(?i)\blarge\s+needle\s+driver\b", name):
        name = re.sub(r"(?i)\bneedle\s+driver\b", "large needle driver", name)
    return name

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

def get_column_name(df: pd.DataFrame, target: str) -> str:
    for c in df.columns:
        if str(c).strip().lower() == target.strip().lower():
            return c
    raise ValueError(f"Expected column '{target}' not found. Columns present: {list(df.columns)}")

def find_adjacent_value_column(df: pd.DataFrame, left_col_name: str) -> str:
    cols = list(df.columns)
    idx = cols.index(left_col_name)
    if idx + 1 >= len(cols):
        raise ValueError(f"No column to the right of '{left_col_name}'")
    return cols[idx + 1]

# ---------- Q/A Builders ----------

def build_tool_qas(df: pd.DataFrame) -> list[str]:
    tools_col_actual = get_column_name(df, "Tools")
    val_col = find_adjacent_value_column(df, tools_col_actual)
    df[val_col] = coerce_numeric_01(df[val_col])

    qas: list[str] = []
    for _, row in df.iterrows():
        tool_raw = normalize_string(row.get(tools_col_actual, ""))
        if not tool_raw:
            continue
        present = is_positive(row.get(val_col, 0))
        tool_disp = display_tool_name(tool_raw)

        if present:
            qas.append(f"Is a {tool_disp} among the listed tools? | Yes, a {tool_disp} is listed.")
            qas.append(f"Was a {tool_disp} used in this clip? | Yes, a {tool_disp} was utilized.")
            qas.append(f"Was a {tool_disp} used during the surgery? | Yes, a {tool_disp} was used.")
        else:
            qas.append(f"Is a {tool_disp} among the listed tools? | No, a {tool_disp} is not listed.")
            qas.append(f"Was a {tool_disp} used in this clip? | No, a {tool_disp} was not utilized.")
            qas.append(f"Was a {tool_disp} used during the surgery? | No, a {tool_disp} was not used.")

    return qas

def build_forceps_qas(df: pd.DataFrame) -> list[str]:
    tools_col_actual = get_column_name(df, "Tools")
    val_col = find_adjacent_value_column(df, tools_col_actual)
    df[val_col] = coerce_numeric_01(df[val_col])

    present_forceps = []
    for _, row in df.iterrows():
        raw_name = normalize_string(row.get(tools_col_actual, ""))
        if not raw_name:
            continue
        if raw_name.lower() in FORCEPS_ALIASES and is_positive(row.get(val_col, 0)):
            present_forceps.append(raw_name)

    qas: list[str] = []
    if present_forceps:
        types_str = ", ".join(dict.fromkeys(present_forceps))
        qas.append("Are there forceps being used here? | Yes, forceps are mentioned.")
        qas.append(f"What type of forceps are used? | {types_str}")
    else:
        qas.append("Are there forceps being used here? | No, forceps are not mentioned.")
    return qas

def build_task_qas(df: pd.DataFrame) -> list[str]:
    try:
        task_col_actual = get_column_name(df, "Task")
    except ValueError:
        return []
    task_val_col = find_adjacent_value_column(df, task_col_actual)
    df[task_val_col] = coerce_numeric_01(df[task_val_col])

    flagged_tasks = []
    for _, row in df.iterrows():
        task_name = normalize_string(row.get(task_col_actual, ""))
        if task_name and is_positive(row.get(task_val_col, 0)):
            flagged_tasks.append(task_name)

    unique_tasks = list(dict.fromkeys(flagged_tasks))
    answer = ", ".join(unique_tasks) if unique_tasks else "None"
    return [f"What task is being performed in this clip? | {answer}"]

# ---------- Processing ----------

def process_case_folder(case_dir: Path):
    """
    Process all CSV files inside a single case folder.
    QA files are renamed as qa_case_<folder_number>_<i>.txt
    Old .txt files are deleted before generating new ones.
    """
    # 🔹 Delete old .txt files first
    for old_txt in case_dir.glob("*.txt"):
        try:
            old_txt.unlink()
            print(f"[DEL] Removed old file: {old_txt}")
        except Exception as e:
            print(f"[WARN] Could not delete {old_txt}: {e}")

    csvs = sorted(case_dir.glob("*.csv"))
    if not csvs:
        print(f"[WARN] No CSV files in {case_dir}")
        return

    # Extract padded case number from folder name
    match = re.search(r"case[_\-]?(\d+)", case_dir.name, re.IGNORECASE)
    if not match:
        print(f"[WARN] Cannot parse case number from {case_dir.name}, skipping.")
        return
    case_num = int(match.group(1))
    padded_case = f"{case_num:03d}"

    for i, csv_path in enumerate(csvs, start=1):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            if csv_path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(csv_path)
            else:
                raise
        df.columns = [str(c).strip() for c in df.columns]

        lines: list[str] = []
        try:
            lines.extend(build_tool_qas(df))
            lines.extend(build_forceps_qas(df))
        except ValueError as e:
            print(f"[WARN] {csv_path.name}: {e} (skipping tool/forceps QAs)")
        try:
            lines.extend(build_task_qas(df))
        except ValueError as e:
            print(f"[WARN] {csv_path.name}: {e} (skipping task QAs)")

        # Add the fixed question at the end
        lines.append("What procedure is this summary describing? | The summary is describing endoscopic or laparoscopic surgery.")

        # Write to new filename
        out_name = f"qa_case_{padded_case}_{i}.txt"
        out_path = case_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        print(f"[OK] Wrote {out_path}")

def main():
    if not ROOT_DIR.exists():
        raise SystemExit(f"Root directory not found: {ROOT_DIR}")

    if RUN_SINGLE_CASE:
        case_dir = ROOT_DIR / SINGLE_CASE_NAME
        if not case_dir.is_dir():
            raise SystemExit(f"Case folder not found for test: {case_dir}")
        print(f"[TEST MODE] Processing single case folder: {case_dir}")
        process_case_folder(case_dir)
        return

    case_dirs = sorted(p for p in ROOT_DIR.iterdir() if p.is_dir() and p.name.lower().startswith("case"))
    if not case_dirs:
        print(f"[WARN] No case folders found under {ROOT_DIR}")
        return

    for case_dir in case_dirs:
        print(f"[CASE] {case_dir}")
        process_case_folder(case_dir)

if __name__ == "__main__":
    main()





