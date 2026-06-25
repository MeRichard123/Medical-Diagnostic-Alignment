import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "preference_data.csv"
OUTPUT_PATH = BASE_DIR / "preference_data_cleaned.csv"


def _first_short_candidate(lines):
    for l in lines:
        if not l:
            continue
        # prefer very short explicit labels
        if len(l.split()) <= 4:
            return l
    return None


def clean_response(text: str) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Normalize whitespace and split into lines
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return ""

    # If the entire output is just the same line repeated, collapse to single instance
    uniq = list(OrderedDict.fromkeys(lines))
    if len(uniq) == 1:
        return uniq[0]

    # Try to extract explicit labels after common markers
    candidates = []
    for l in lines:
        m = re.search(r"(?:Diagnosis|Answer|Label)\s*[:\-]?\s*(.+)", l, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            # strip trailing punctuation
            cand = re.sub(r"[\.\,\;\:]+$", "", cand).strip()
            if cand:
                candidates.append(cand)
    if candidates:
        # prefer the shortest explicit candidate
        candidates = sorted(candidates, key=lambda s: (len(s.split()), len(s)))
        return candidates[0]

    # Remove long repeated sections: keep unique lines in order
    if len(uniq) > 0:
        # Prefer first short line among unique lines
        short = _first_short_candidate(uniq)
        if short:
            return short
        # Otherwise, return the first unique line truncated to 6 words
        first = uniq[0]
        words = first.split()
        return " ".join(words[:6])


def main(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH):
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    for col in ("chosen", "rejected"):
        if col in df.columns:
            cleaned = []
            for v in df[col].fillna(""):
                cleaned.append(clean_response(v))
            df[col] = cleaned

    df.to_csv(output_path, index=False)
    print(f"Wrote cleaned preferences to: {output_path}")


if __name__ == "__main__":
    main()
