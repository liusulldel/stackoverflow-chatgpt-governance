from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

POSTS_XML = RAW_DIR / "stackexchange_20251231" / "stackoverflow.com_extracted" / "Posts.xml"

QUESTION_LEVEL_PARQUET = PROCESSED_DIR / "stackexchange_20251231_question_level_enriched.parquet"

QUESTION_LEVEL_OUT_PARQUET = PROCESSED_DIR / "build_who_still_answers_moderation_question_level.parquet"
QUESTION_LEVEL_OUT_CSV = PROCESSED_DIR / "build_who_still_answers_moderation_question_level.csv"
PANEL_OUT_CSV = PROCESSED_DIR / "build_who_still_answers_moderation_panel.csv"
SUMMARY_JSON = PROCESSED_DIR / "build_who_still_answers_moderation_summary.json"

SHOCK_DATE = pd.Timestamp("2022-11-30T00:00:00Z")


def fast_attr(line: str, key: str) -> Optional[str]:
    needle = f'{key}="'
    start = line.find(needle)
    if start == -1:
        return None
    start += len(needle)
    end = line.find('"', start)
    if end == -1:
        return None
    return line[start:end]


def extract_closed_dates(posts_path: Path, question_ids: Iterable[int]) -> Dict[int, str]:
    question_id_set = set(int(qid) for qid in question_ids)
    closed_dates: Dict[int, str] = {}
    matched = 0
    closed_found = 0
    log_every = 5_000_000

    with posts_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f):
            if "<row" not in line or 'PostTypeId="1"' not in line:
                continue
            qid_raw = fast_attr(line, "Id")
            if qid_raw is None:
                continue
            qid = int(qid_raw)
            if qid not in question_id_set:
                continue
            matched += 1
            closed_date = fast_attr(line, "ClosedDate")
            if closed_date:
                closed_dates[qid] = closed_date
                closed_found += 1
            if matched % 100000 == 0:
                print(
                    f"Matched {matched:,} questions; closed dates {closed_found:,} "
                    f"(line {line_idx:,})."
                )
            if line_idx % log_every == 0 and line_idx > 0:
                print(f"Scanned {line_idx:,} lines; matched {matched:,}.")

    print(f"Finished scanning posts. Matched {matched:,} questions, closed dates {closed_found:,}.")
    return closed_dates


def compute_observation_cutoff(question_created_at: pd.Series, closed_at: pd.Series) -> pd.Timestamp:
    candidates = [question_created_at.max()]
    if closed_at.notna().any():
        candidates.append(closed_at.max())
    return max(pd.Timestamp(candidate) for candidate in candidates if pd.notna(candidate))


def add_closure_features(df: pd.DataFrame) -> pd.DataFrame:
    df["question_created_at"] = pd.to_datetime(df["question_created_at"], utc=True)
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    df["closed_archive"] = df["closed_at"].notna().astype(int)

    observation_cutoff = compute_observation_cutoff(df["question_created_at"], df["closed_at"])
    df["observation_cutoff_at"] = observation_cutoff

    for window_name, days in [("closed_7d", 7), ("closed_30d", 30)]:
        hours = 24 * days
        delta = (df["closed_at"] - df["question_created_at"]).dt.total_seconds() / 3600.0
        df[f"{window_name}_hours"] = delta
        df[window_name] = ((delta >= 0) & (delta <= hours)).fillna(False).astype(int)
        df[f"{window_name}_eligible"] = (df["question_created_at"] <= (observation_cutoff - pd.Timedelta(hours=hours))).astype(int)

    return df


def build_primary_panel(df: pd.DataFrame) -> pd.DataFrame:
    def eligible_rate(g: pd.DataFrame, event_col: str, eligible_col: str) -> tuple[float, float]:
        denom = float(g[eligible_col].sum())
        if denom <= 0:
            return float("nan"), 0.0
        numerator = float(g.loc[g[eligible_col] == 1, event_col].sum())
        return numerator / denom, denom

    sample = df.loc[df["keep_single_focal"] == 1].copy()
    rows: list[dict] = []
    for (tag, month_id), g in sample.groupby(["primary_tag", "month_id"], sort=True):
        closed_7d_rate, closed_7d_denom = eligible_rate(g, "closed_7d", "closed_7d_eligible")
        closed_30d_rate, closed_30d_denom = eligible_rate(g, "closed_30d", "closed_30d_eligible")
        rows.append(
            {
                "tag": tag,
                "month_id": month_id,
                "n_questions": int(len(g)),
                "high_tag": int(g["high_tag"].iloc[0]) if "high_tag" in g else 0,
                "exposure_index": float(g["exposure_index"].iloc[0]) if "exposure_index" in g else float("nan"),
                "closed_archive_rate": float(g["closed_archive"].mean()),
                "closed_7d_rate": closed_7d_rate,
                "closed_7d_denom": closed_7d_denom,
                "closed_30d_rate": closed_30d_rate,
                "closed_30d_denom": closed_30d_denom,
            }
        )

    grouped = pd.DataFrame(rows)
    grouped["group"] = grouped["high_tag"].map({1: "high", 0: "low"})
    grouped["post_chatgpt"] = (pd.to_datetime(grouped["month_id"] + "-01", utc=True) >= SHOCK_DATE).astype(int)
    grouped["high_post"] = grouped["high_tag"] * grouped["post_chatgpt"]
    grouped["exposure_post"] = grouped["exposure_index"] * grouped["post_chatgpt"]
    return grouped.sort_values(["tag", "month_id"]).reset_index(drop=True)


def summarize_diff_in_diff(df: pd.DataFrame, outcome: str, eligible_col: Optional[str]) -> dict:
    sample = df.loc[df["keep_single_focal"] == 1].copy()
    if eligible_col is not None:
        sample = sample.loc[sample[eligible_col] == 1].copy()
    sample["group"] = sample["high_tag"].map({1: "high", 0: "low"})
    sample["post_chatgpt"] = (sample["question_created_at"] >= SHOCK_DATE).astype(int)

    summary = (
        sample.groupby(["group", "post_chatgpt"], as_index=False)[outcome]
        .mean()
        .pivot(index="group", columns="post_chatgpt", values=outcome)
    )
    high_pre = float(summary.loc["high", 0]) if 0 in summary.columns else float("nan")
    high_post = float(summary.loc["high", 1]) if 1 in summary.columns else float("nan")
    low_pre = float(summary.loc["low", 0]) if 0 in summary.columns else float("nan")
    low_post = float(summary.loc["low", 1]) if 1 in summary.columns else float("nan")

    return {
        "high_pre": high_pre,
        "high_post": high_post,
        "low_pre": low_pre,
        "low_post": low_post,
        "diff_in_diff": (high_post - high_pre) - (low_post - low_pre),
        "eligible_filter": eligible_col,
    }


def main() -> None:
    if not POSTS_XML.exists():
        raise FileNotFoundError(f"Posts XML not found: {POSTS_XML}")

    question_level = pd.read_parquet(QUESTION_LEVEL_PARQUET)
    closed_dates = extract_closed_dates(POSTS_XML, question_level["question_id"].tolist())
    question_level = question_level.copy()
    question_level["closed_at"] = question_level["question_id"].map(closed_dates)

    question_level = add_closure_features(question_level)
    panel = build_primary_panel(question_level)

    QUESTION_LEVEL_OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    question_level.to_parquet(QUESTION_LEVEL_OUT_PARQUET, index=False)
    question_level.to_csv(QUESTION_LEVEL_OUT_CSV, index=False)
    panel.to_csv(PANEL_OUT_CSV, index=False)

    summary = {
        "observation_cutoff": question_level["observation_cutoff_at"].iloc[0].isoformat(),
        "n_question_level_rows": int(len(question_level)),
        "n_closed_archive": int(question_level["closed_archive"].sum()),
        "closed_archive_rate": float(question_level["closed_archive"].mean()),
        "closed_7d_mean_eligible": float(question_level.loc[question_level["closed_7d_eligible"] == 1, "closed_7d"].mean()),
        "closed_30d_mean_eligible": float(question_level.loc[question_level["closed_30d_eligible"] == 1, "closed_30d"].mean()),
        "min_question_created_at": question_level["question_created_at"].min().isoformat(),
        "max_question_created_at": question_level["question_created_at"].max().isoformat(),
        "diff_in_diff_closed_7d": summarize_diff_in_diff(question_level, "closed_7d", "closed_7d_eligible"),
        "diff_in_diff_closed_30d": summarize_diff_in_diff(question_level, "closed_30d", "closed_30d_eligible"),
        "diff_in_diff_closed_archive": summarize_diff_in_diff(question_level, "closed_archive", None),
        "notes": [
            "Deletion events are not present in Posts.xml; closure relies on ClosedDate for currently-closed questions.",
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
