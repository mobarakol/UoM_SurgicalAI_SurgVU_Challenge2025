#!/usr/bin/env python3
import os
import re
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

def _looks_like_labels(s: pd.Series, thresh: float = 0.8) -> bool:
    if s.dtype != "object":
        return False
    vals = s.dropna().astype(str).str.strip()
    if vals.empty:
        return False
    # proportion of entries that are NOT numeric-looking
    non_numeric = ~vals.str.fullmatch(r"-?\d+(\.\d+)?")
    return (non_numeric.mean() >= thresh)

_BAD_NAME_HEADERS = re.compile(r"^(yes(_?no)?|yes/?no|flag|present|absent|y/?n|true|false)$", re.I)

def _is_binary_flags(col: pd.Series) -> bool:
    s = pd.to_numeric(col, errors="coerce")
    s = s.dropna()
    if s.empty:
        return False
    return set(s.unique()).issubset({0, 1})

def _find_case_folder(csv_path: Path) -> str:
    """
    Walk up from the CSV file to find a directory named like 'case_###'.
    If none found, return the CSV's immediate parent directory name.
    """
    for ancestor in [csv_path.parent, *csv_path.parents]:
        name = ancestor.name
        if re.fullmatch(r"case[_-]?\d{1,}", name, flags=re.IGNORECASE):
            return name
    return csv_path.parent.name

def count_tools_and_tasks(root_dir: str):
    tools = Counter()
    tasks = Counter()
    # NEW: mapping from label -> set(case folders)
    tools_to_cases = defaultdict(set)
    tasks_to_cases = defaultdict(set)

    bad_files = []

    for subdir, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            fp = os.path.join(subdir, fn)
            csv_path = Path(fp)
            case_folder = _find_case_folder(csv_path)

            try:
                df = pd.read_csv(fp)
            except Exception as e:
                bad_files.append((fp, f"read error: {e}"))
                continue

            cols = list(df.columns)
            used_pairs = False

            # --- try semantic pairs: (label column, binary flag column) ---
            for i in range(0, len(cols) - 1, 2):
                name_col = cols[i]
                flag_col = cols[i + 1]

                # skip if left column header obviously a flag
                if _BAD_NAME_HEADERS.match(str(name_col)):
                    continue

                flags_raw = df[flag_col]
                if not _is_binary_flags(flags_raw):
                    continue

                names_raw = df[name_col].astype(str).str.strip()
                if not _looks_like_labels(names_raw):
                    continue

                used_pairs = True
                flags = pd.to_numeric(flags_raw, errors="coerce").fillna(0).astype(int)

                col_lower = str(name_col).lower()
                if "tool" in col_lower:
                    bucket = tools
                    mapping = tools_to_cases
                elif "task" in col_lower:
                    bucket = tasks
                    mapping = tasks_to_cases
                else:
                    # If ambiguous, default to tasks (keeps behavior close to original)
                    bucket = tasks
                    mapping = tasks_to_cases

                for n, f in zip(names_raw, flags):
                    n_clean = n.strip()
                    if f == 1 and n_clean and n_clean.lower() != "nan":
                        bucket[n_clean] += 1
                        mapping[n_clean].add(case_folder)

            # --- fallback: wide one-hot format (treat as tasks) ---
            if not used_pairs:
                for c in df.columns:
                    cl = str(c).lower()
                    if cl.startswith("unnamed") or cl in {"index", "id"}:
                        continue
                    if _BAD_NAME_HEADERS.match(str(c)):
                        continue
                    s = pd.to_numeric(df[c], errors="coerce")
                    if s.isna().all():
                        continue
                    vals = set(s.dropna().unique().tolist())
                    if vals.issubset({0, 1}):
                        total = int(s.sum())
                        if total > 0:
                            cname = str(c).strip()
                            tasks[cname] += total
                            tasks_to_cases[cname].add(case_folder)

    # --- print results ---
    def _print_with_cases(title: str, counter: Counter, mapping: dict):
        print(title)
        if not counter:
            print("  (none)")
            return
        for name, cnt in counter.most_common():
            print(f"{name}: {cnt}")
            cases = sorted(mapping.get(name, []))
            if cases:
                print("   cases: " + ", ".join(cases))

    _print_with_cases("== Tools ==", tools, tools_to_cases)
    print()
    _print_with_cases("== Tasks ==", tasks, tasks_to_cases)

    if bad_files:
        print("\nFiles skipped due to errors:")
        for fp, msg in bad_files[:10]:
            print(f"  {fp}: {msg}")
        if len(bad_files) > 10:
            print(f"  ... and {len(bad_files) - 10} more")

# Example usage
if __name__ == "__main__":
    root_directory = ""
    # root_directory = ""
    count_tools_and_tasks(root_directory)
