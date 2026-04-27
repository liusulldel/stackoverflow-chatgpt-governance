from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"

QUESTIONS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_questions.parquet"
ANSWERS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_answers.parquet"
ACCEPT_VOTES_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_accept_votes.parquet"
EXPOSURE_TAG_CSV = PROCESSED_DIR / "strengthened_exposure_tag_scores.csv"

QUESTION_LEVEL_PARQUET = PROCESSED_DIR / "stackexchange_20251231_question_level_enriched.parquet"
QUESTION_LEVEL_CSV = PROCESSED_DIR / "stackexchange_20251231_question_level_enriched.csv"
PRIMARY_PANEL_CSV = PROCESSED_DIR / "stackexchange_20251231_primary_panel.csv"
FRACTIONAL_PANEL_CSV = PROCESSED_DIR / "stackexchange_20251231_fractional_panel.csv"
SUMMARY_JSON = PROCESSED_DIR / "stackexchange_20251231_panel_summary.json"

SELECTED_TAGS = [
    "apache-spark",
    "android",
    "bash",
    "docker",
    "excel",
    "firebase",
    "javascript",
    "kubernetes",
    "linux",
    "memory-management",
    "multithreading",
    "numpy",
    "pandas",
    "python",
    "regex",
    "sql",
]

HIGH_EXPOSURE_TAGS = {
    "bash",
    "excel",
    "javascript",
    "numpy",
    "pandas",
    "python",
    "regex",
    "sql",
}

SHOCK_DATE = pd.Timestamp("2022-11-30T00:00:00Z")


def parse_selected_tags(tag_string: str | None) -> list[str]:
    if tag_string is None or pd.isna(tag_string):
        return []
    return [tag for tag in str(tag_string).split(";") if tag in SELECTED_TAGS]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    questions = pd.read_parquet(QUESTIONS_PARQUET)
    answers = pd.read_parquet(ANSWERS_PARQUET)
    accept_votes = pd.read_parquet(ACCEPT_VOTES_PARQUET) if ACCEPT_VOTES_PARQUET.exists() else pd.DataFrame(columns=["answer_id", "accept_vote_date"])
    exposure = pd.read_csv(EXPOSURE_TAG_CSV)

    questions["question_created_at"] = pd.to_datetime(questions["question_created_at"], utc=True)
    answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], utc=True)
    if not accept_votes.empty:
        accept_votes["accept_vote_date"] = pd.to_datetime(accept_votes["accept_vote_date"], utc=True, errors="coerce")

    return questions, answers, accept_votes, exposure


def compute_observation_cutoff(
    questions: pd.DataFrame,
    answers: pd.DataFrame,
    accept_votes: pd.DataFrame,
) -> pd.Timestamp:
    candidates = [questions["question_created_at"].max(), answers["answer_created_at"].max()]
    if not accept_votes.empty and accept_votes["accept_vote_date"].notna().any():
        candidates.append(accept_votes["accept_vote_date"].max())
    return max(pd.Timestamp(candidate) for candidate in candidates if pd.notna(candidate))


def compute_question_level(
    questions: pd.DataFrame,
    answers: pd.DataFrame,
    accept_votes: pd.DataFrame,
    exposure: pd.DataFrame,
    observation_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    first_answers = (
        answers.groupby("question_id", as_index=False)["answer_created_at"]
        .min()
        .rename(columns={"answer_created_at": "first_answer_at"})
    )

    accept_dates = pd.DataFrame(columns=["accepted_answer_id", "accept_vote_date"])
    if not accept_votes.empty:
        accept_dates = (
            accept_votes.groupby("answer_id", as_index=False)["accept_vote_date"]
            .min()
            .rename(columns={"answer_id": "accepted_answer_id"})
        )

    df = questions.merge(first_answers, on="question_id", how="left").merge(accept_dates, on="accepted_answer_id", how="left")
    df["question_tags_list"] = df["question_tags"].apply(parse_selected_tags)
    df["selected_tags_list"] = df["selected_tags"].apply(parse_selected_tags)
    df["selected_tag_overlap"] = df["selected_tags_list"].str.len()
    df["keep_single_focal"] = (df["selected_tag_overlap"] == 1).astype(int)
    df["primary_tag"] = df["selected_tags_list"].apply(lambda x: x[0] if len(x) == 1 else None)
    df["month_id"] = df["question_created_at"].dt.strftime("%Y-%m")
    df["post_chatgpt"] = (df["question_created_at"] >= SHOCK_DATE).astype(int)
    df["accepted_archive"] = df["accepted_answer_id"].notna().astype(int)
    df["observation_cutoff_at"] = observation_cutoff

    for window_name, days in [("first_answer_1d", 1), ("first_answer_7d", 7)]:
        hours = 24 * days
        delta = (df["first_answer_at"] - df["question_created_at"]).dt.total_seconds() / 3600.0
        df[f"{window_name}_hours"] = delta
        df[window_name] = ((delta >= 0) & (delta <= hours)).fillna(False).astype(int)
        df[f"{window_name}_eligible"] = (df["question_created_at"] <= (observation_cutoff - pd.Timedelta(hours=hours))).astype(int)

    for window_name, days in [("accepted_7d", 7), ("accepted_30d", 30)]:
        hours = 24 * days
        delta = (df["accept_vote_date"] - df["question_created_at"]).dt.total_seconds() / 3600.0
        df[f"{window_name}_hours"] = delta
        df[window_name] = ((delta >= 0) & (delta <= hours)).fillna(False).astype(int)
        df[f"{window_name}_eligible"] = (df["question_created_at"] <= (observation_cutoff - pd.Timedelta(hours=hours))).astype(int)

    exposure_lookup = exposure[["tag", "exposure_index"]].copy()
    exposure_lookup["high_tag"] = exposure_lookup["tag"].isin(HIGH_EXPOSURE_TAGS).astype(int)
    df = df.merge(exposure_lookup.rename(columns={"tag": "primary_tag"}), on="primary_tag", how="left")

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
        accepted_7d_rate, accepted_7d_denom = eligible_rate(g, "accepted_7d", "accepted_7d_eligible")
        accepted_30d_rate, accepted_30d_denom = eligible_rate(g, "accepted_30d", "accepted_30d_eligible")
        first_answer_1d_rate, first_answer_1d_denom = eligible_rate(g, "first_answer_1d", "first_answer_1d_eligible")
        first_answer_7d_rate, first_answer_7d_denom = eligible_rate(g, "first_answer_7d", "first_answer_7d_eligible")
        rows.append(
            {
                "tag": tag,
                "month_id": month_id,
                "n_questions": int(len(g)),
                "high_tag": int(g["high_tag"].iloc[0]),
                "exposure_index": float(g["exposure_index"].iloc[0]),
                "accepted_archive_rate": float(g["accepted_archive"].mean()),
                "accepted_7d_rate": accepted_7d_rate,
                "accepted_7d_denom": accepted_7d_denom,
                "accepted_30d_rate": accepted_30d_rate,
                "accepted_30d_denom": accepted_30d_denom,
                "first_answer_1d_rate": first_answer_1d_rate,
                "first_answer_1d_denom": first_answer_1d_denom,
                "first_answer_7d_rate": first_answer_7d_rate,
                "first_answer_7d_denom": first_answer_7d_denom,
            }
        )
    grouped = pd.DataFrame(rows)
    grouped["group"] = grouped["high_tag"].map({1: "high", 0: "low"})
    grouped["post_chatgpt"] = (pd.to_datetime(grouped["month_id"] + "-01", utc=True) >= SHOCK_DATE).astype(int)
    grouped["high_post"] = grouped["high_tag"] * grouped["post_chatgpt"]
    grouped["exposure_post"] = grouped["exposure_index"] * grouped["post_chatgpt"]
    return grouped.sort_values(["tag", "month_id"]).reset_index(drop=True)


def build_fractional_panel(df: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    exposure_map = exposure.set_index("tag")["exposure_index"].to_dict()
    high_map = {tag: int(tag in HIGH_EXPOSURE_TAGS) for tag in SELECTED_TAGS}

    for row in df.itertuples(index=False):
        selected_tags = list(row.selected_tags_list)
        if not selected_tags:
            continue
        weight = 1.0 / len(selected_tags)
        for tag in selected_tags:
            rows.append(
                {
                    "question_id": row.question_id,
                    "tag": tag,
                    "month_id": row.month_id,
                    "weight": weight,
                    "accepted_archive": row.accepted_archive,
                    "accepted_7d": row.accepted_7d,
                    "accepted_7d_eligible": row.accepted_7d_eligible,
                    "accepted_30d": row.accepted_30d,
                    "accepted_30d_eligible": row.accepted_30d_eligible,
                    "first_answer_1d": row.first_answer_1d,
                    "first_answer_1d_eligible": row.first_answer_1d_eligible,
                    "first_answer_7d": row.first_answer_7d,
                    "first_answer_7d_eligible": row.first_answer_7d_eligible,
                    "high_tag": high_map.get(tag, 0),
                    "exposure_index": exposure_map.get(tag),
                    "post_chatgpt": row.post_chatgpt,
                }
            )

    fractional = pd.DataFrame(rows)
    grouped = (
        fractional.groupby(["tag", "month_id"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_questions_weighted": g["weight"].sum(),
                    "high_tag": g["high_tag"].iloc[0],
                    "exposure_index": g["exposure_index"].iloc[0],
                    "accepted_archive_rate": (g["accepted_archive"] * g["weight"]).sum() / g["weight"].sum(),
                    "accepted_7d_denom": (g["accepted_7d_eligible"] * g["weight"]).sum(),
                    "accepted_7d_rate": ((g["accepted_7d"] * g["weight"]).sum() / (g["accepted_7d_eligible"] * g["weight"]).sum()) if (g["accepted_7d_eligible"] * g["weight"]).sum() > 0 else float("nan"),
                    "accepted_30d_denom": (g["accepted_30d_eligible"] * g["weight"]).sum(),
                    "accepted_30d_rate": ((g["accepted_30d"] * g["weight"]).sum() / (g["accepted_30d_eligible"] * g["weight"]).sum()) if (g["accepted_30d_eligible"] * g["weight"]).sum() > 0 else float("nan"),
                    "first_answer_1d_denom": (g["first_answer_1d_eligible"] * g["weight"]).sum(),
                    "first_answer_1d_rate": ((g["first_answer_1d"] * g["weight"]).sum() / (g["first_answer_1d_eligible"] * g["weight"]).sum()) if (g["first_answer_1d_eligible"] * g["weight"]).sum() > 0 else float("nan"),
                    "first_answer_7d_denom": (g["first_answer_7d_eligible"] * g["weight"]).sum(),
                    "first_answer_7d_rate": ((g["first_answer_7d"] * g["weight"]).sum() / (g["first_answer_7d_eligible"] * g["weight"]).sum()) if (g["first_answer_7d_eligible"] * g["weight"]).sum() > 0 else float("nan"),
                }
            )
        )
        .reset_index(drop=True)
    )
    grouped["group"] = grouped["high_tag"].map({1: "high", 0: "low"})
    grouped["post_chatgpt"] = (pd.to_datetime(grouped["month_id"] + "-01", utc=True) >= SHOCK_DATE).astype(int)
    grouped["high_post"] = grouped["high_tag"] * grouped["post_chatgpt"]
    grouped["exposure_post"] = grouped["exposure_index"] * grouped["post_chatgpt"]
    return grouped.sort_values(["tag", "month_id"]).reset_index(drop=True)


def main() -> None:
    questions, answers, accept_votes, exposure = load_inputs()
    observation_cutoff = compute_observation_cutoff(questions, answers, accept_votes)
    question_level = compute_question_level(questions, answers, accept_votes, exposure, observation_cutoff)
    primary_panel = build_primary_panel(question_level)
    fractional_panel = build_fractional_panel(question_level, exposure)

    question_level.to_parquet(QUESTION_LEVEL_PARQUET, index=False)
    question_level.to_csv(QUESTION_LEVEL_CSV, index=False)
    primary_panel.to_csv(PRIMARY_PANEL_CSV, index=False)
    fractional_panel.to_csv(FRACTIONAL_PANEL_CSV, index=False)

    summary = {
        "observation_cutoff": observation_cutoff.isoformat(),
        "n_question_level_rows": int(len(question_level)),
        "n_primary_panel_rows": int(len(primary_panel)),
        "n_fractional_panel_rows": int(len(fractional_panel)),
        "min_question_created_at": question_level["question_created_at"].min().isoformat(),
        "max_question_created_at": question_level["question_created_at"].max().isoformat(),
        "accepted_archive_mean": float(question_level["accepted_archive"].mean()),
        "accepted_7d_mean_eligible": float(question_level.loc[question_level["accepted_7d_eligible"] == 1, "accepted_7d"].mean()),
        "accepted_30d_mean_eligible": float(question_level.loc[question_level["accepted_30d_eligible"] == 1, "accepted_30d"].mean()),
        "first_answer_1d_mean_eligible": float(question_level.loc[question_level["first_answer_1d_eligible"] == 1, "first_answer_1d"].mean()),
        "first_answer_7d_mean_eligible": float(question_level.loc[question_level["first_answer_7d_eligible"] == 1, "first_answer_7d"].mean()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
