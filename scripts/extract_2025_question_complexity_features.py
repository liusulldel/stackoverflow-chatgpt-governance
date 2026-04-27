from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw" / "stackexchange_20251231" / "stackoverflow.com_extracted"
PROCESSED_DIR = BASE_DIR / "processed"

POSTS_XML = RAW_DIR / "Posts.xml"
FOCAL_QUESTIONS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_questions.parquet"

OUTPUT_PARQUET = PROCESSED_DIR / "stackexchange_20251231_question_complexity_features.parquet"
OUTPUT_CSV = PROCESSED_DIR / "stackexchange_20251231_question_complexity_features.csv"
SUMMARY_JSON = PROCESSED_DIR / "stackexchange_20251231_question_complexity_summary.json"

TAG_RE = re.compile(r"<[^>]+>")
CODE_BLOCK_RE = re.compile(r"<pre(?:\\s[^>]*)?>(.*?)</pre>", flags=re.IGNORECASE | re.DOTALL)
INLINE_CODE_RE = re.compile(r"<code(?:\\s[^>]*)?>(.*?)</code>", flags=re.IGNORECASE | re.DOTALL)
ERROR_RE = re.compile(
    r"\\b(?:error|exception|traceback|stack trace|segmentation fault|segfault|syntaxerror|typeerror|valueerror|keyerror|nullpointer|undefined|failed|cannot|can't)\\b",
    flags=re.IGNORECASE,
)


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    text = TAG_RE.sub(" ", html.unescape(value))
    return re.sub(r"\\s+", " ", text).strip()


def parse_tags(tags_raw: str | None) -> list[str]:
    if not tags_raw:
        return []
    stripped = tags_raw.strip("<>")
    if not stripped:
        return []
    return [tag for tag in stripped.split("><") if tag]


def compute_metrics(body_html: str | None, comment_count_raw: str | None, tags_raw: str | None, last_edit_raw: str | None, title_raw: str | None) -> dict[str, object]:
    body_html = body_html or ""
    title_text = normalize_text(title_raw)
    body_text = normalize_text(body_html)
    code_blocks = CODE_BLOCK_RE.findall(body_html)
    inline_codes = INLINE_CODE_RE.findall(body_html)
    code_char_count = sum(len(normalize_text(block)) for block in code_blocks) + sum(len(normalize_text(code)) for code in inline_codes)
    body_word_count = len(body_text.split())
    error_keyword_count = len(ERROR_RE.findall(body_text))
    tags = parse_tags(tags_raw)

    return {
        "title_length_chars": len(title_text),
        "body_length_chars": len(body_text),
        "body_word_count": body_word_count,
        "code_block_count": len(code_blocks),
        "inline_code_count": len(inline_codes),
        "code_char_count": code_char_count,
        "error_keyword_count": error_keyword_count,
        "error_keyword_density": (error_keyword_count / body_word_count) if body_word_count > 0 else 0.0,
        "comment_count": int(comment_count_raw) if comment_count_raw else 0,
        "has_edit": 1 if last_edit_raw else 0,
        "tag_count_full": len(tags),
    }


def main() -> None:
    if not POSTS_XML.exists():
        raise FileNotFoundError(f"Posts.xml not found: {POSTS_XML}")

    focal_questions = pd.read_parquet(FOCAL_QUESTIONS_PARQUET, columns=["question_id"])
    focal_ids = set(focal_questions["question_id"].astype(int).tolist())

    rows: list[dict[str, object]] = []
    rows_seen = 0
    matched = 0

    context = ET.iterparse(POSTS_XML, events=("end",))
    for _, elem in context:
        if elem.tag != "row":
            continue
        rows_seen += 1
        attrib = elem.attrib
        if attrib.get("PostTypeId") != "1":
            elem.clear()
            continue
        post_id = attrib.get("Id")
        if not post_id:
            elem.clear()
            continue
        post_id_int = int(post_id)
        if post_id_int not in focal_ids:
            elem.clear()
            continue

        matched += 1
        metrics = compute_metrics(
            attrib.get("Body"),
            attrib.get("CommentCount"),
            attrib.get("Tags"),
            attrib.get("LastEditDate"),
            attrib.get("Title"),
        )
        metrics["question_id"] = post_id_int
        rows.append(metrics)

        if matched % 200_000 == 0:
            print(f"Matched {matched:,} / {len(focal_ids):,} focal questions after scanning {rows_seen:,} rows.", flush=True)
        elem.clear()

    features = pd.DataFrame(rows).sort_values("question_id").reset_index(drop=True)
    features.to_parquet(OUTPUT_PARQUET, index=False)
    features.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "rows_seen": rows_seen,
        "n_focal_questions_requested": len(focal_ids),
        "n_features_rows": int(len(features)),
        "coverage_rate": float(len(features) / len(focal_ids)) if focal_ids else 0.0,
        "output_parquet": str(OUTPUT_PARQUET),
        "output_csv": str(OUTPUT_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
