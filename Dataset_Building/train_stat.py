import os
import re
import pandas as pd
from collections import Counter

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

def count_tools_and_tasks(root_dir: str):
    tools = Counter()
    tasks = Counter()
    bad_files = []

    for subdir, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            fp = os.path.join(subdir, fn)

            try:
                df = pd.read_csv(fp)
            except Exception as e:
                bad_files.append((fp, f"read error: {e}"))
                continue

            cols = list(df.columns)
            used_pairs = False

            # --- try semantic pairs, but validate left is labels & right is binary
            for i in range(0, len(cols) - 1, 2):
                name_col = cols[i]
                flag_col = cols[i + 1]

                # skip if left column header screams "this is a flag"
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
                elif "task" in col_lower:
                    bucket = tasks
                else:
                    # If header doesn't say, infer by the OTHER column header if possible
                    bucket = tasks

                for n, f in zip(names_raw, flags):
                    if f == 1 and n and n.lower() != "nan":
                        bucket[n] += 1

            # --- fallback: wide one-hot format
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
                            # Put unknowns under tasks so they still show up
                            tasks[c] += total

    # --- print results ---
    print("== Tools ==")
    for name, cnt in tools.most_common():
        print(f"{name}: {cnt}")

    print("\n== Tasks ==")
    for name, cnt in tasks.most_common():
        print(f"{name}: {cnt}")

    if bad_files:
        print("\nFiles skipped due to errors:")
        for fp, msg in bad_files[:10]:
            print(f"  {fp}: {msg}")
        if len(bad_files) > 10:
            print(f"  ... and {len(bad_files) - 10} more")


# Example usage
if __name__ == "__main__":
    root_directory = ""
    count_tools_and_tasks(root_directory)



