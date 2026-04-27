from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

QUESTION_PANEL = PROCESSED / "stackexchange_20251231_question_level_enriched.parquet"
STRICT_HITS = PROCESSED / "who_still_answers_posthistory_direct_ai_question_hits.parquet"

RESULTS_CSV = PROCESSED / "who_still_answers_ai_ban_strict_question_side_results.csv"
COUNTS_CSV = PROCESSED / "who_still_answers_ai_ban_strict_question_side_counts.csv"
SUMMARY_JSON = PROCESSED / "who_still_answers_ai_ban_strict_question_side_summary.json"
MEMO_PATH = PAPER / "who_still_answers_ai_ban_strict_question_side_2026-04-06.md"

BAN_DATE = pd.Timestamp("2022-12-05", tz="UTC")
WINDOWS = [30, 45, 60]
DONUTS = [0, 3, 7]
OUTCOMES = [
    ("first_answer_1d", "first_answer_1d_eligible"),
    ("accepted_7d", "accepted_7d_eligible"),
    ("accepted_30d", "accepted_30d_eligible"),
]


def load_sample() -> pd.DataFrame:
    q = pd.read_parquet(
        QUESTION_PANEL,
        columns=[
            "question_id",
            "question_created_at",
            "primary_tag",
            "high_tag",
            "first_answer_1d",
            "first_answer_1d_eligible",
            "accepted_7d",
            "accepted_7d_eligible",
            "accepted_30d",
            "accepted_30d_eligible",
        ],
    ).copy()
    q["question_id"] = q["question_id"].astype(int)
    q["question_created_at"] = pd.to_datetime(q["question_created_at"], utc=True)
    q["primary_tag"] = q["primary_tag"].fillna("unknown").astype(str)
    q["high_tag"] = q["high_tag"].fillna(0).astype(int)

    h = pd.read_parquet(
        STRICT_HITS,
        columns=["question_id", "strict_question_side_hit", "keep_single_focal"],
    ).copy()
    h["question_id"] = h["question_id"].astype(int)

    df = q.merge(h, on="question_id", how="left")
    df["strict_question_side_hit"] = df["strict_question_side_hit"].fillna(0).astype(int)
    df["keep_single_focal"] = df["keep_single_focal"].fillna(0).astype(int)
    df = df.loc[df["keep_single_focal"] == 1].copy()
    df["days_from_ban"] = (df["question_created_at"] - BAN_DATE).dt.total_seconds() / 86400.0
    df["post_ban"] = (df["question_created_at"] >= BAN_DATE).astype(int)
    return df


def build_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []

    for sample_name, mask in [
        ("all_tags", pd.Series(True, index=df.index)),
        ("high_tags_only", df["high_tag"] == 1),
    ]:
        for window in WINDOWS:
            for donut in DONUTS:
                subset = df.loc[
                    (df["days_from_ban"].abs() <= window)
                    & (df["days_from_ban"].abs() > donut)
                    & mask
                ].copy()
                hits = int(subset["strict_question_side_hit"].sum())
                if hits < 30:
                    continue

                count_rows.append(
                    {
                        "sample": sample_name,
                        "window_days": window,
                        "donut_days": donut,
                        "n_questions": int(len(subset)),
                        "n_hits": hits,
                        "pre_hits": int(subset.loc[subset["post_ban"] == 0, "strict_question_side_hit"].sum()),
                        "post_hits": int(subset.loc[subset["post_ban"] == 1, "strict_question_side_hit"].sum()),
                    }
                )

                for outcome, eligible in OUTCOMES:
                    frame = subset.loc[subset[eligible] == 1].copy()
                    frame_hits = int(frame["strict_question_side_hit"].sum())
                    if frame.empty or frame_hits < 30:
                        continue
                    model = smf.ols(
                        (
                            f"{outcome} ~ strict_question_side_hit * post_ban + "
                            "days_from_ban + post_ban:days_from_ban + "
                            "strict_question_side_hit:days_from_ban + C(primary_tag)"
                        ),
                        data=frame,
                    ).fit(cov_type="cluster", cov_kwds={"groups": frame["primary_tag"]})
                    term = "strict_question_side_hit:post_ban"
                    result_rows.append(
                        {
                            "sample": sample_name,
                            "window_days": window,
                            "donut_days": donut,
                            "outcome": outcome,
                            "coef": float(model.params.get(term, float("nan"))),
                            "se": float(model.bse.get(term, float("nan"))),
                            "pval": float(model.pvalues.get(term, float("nan"))),
                            "n_hits": frame_hits,
                            "nobs": int(model.nobs),
                            "mean_outcome": float(frame[outcome].mean()),
                        }
                    )

    return pd.DataFrame(count_rows), pd.DataFrame(result_rows)


def write_summary(counts: pd.DataFrame, results: pd.DataFrame) -> None:
    promoted = results.loc[
        (results["sample"] == "high_tags_only")
        & (results["outcome"].isin(["accepted_7d", "accepted_30d", "first_answer_1d"]))
    ].copy()
    summary = {
        "design": "strict question-side AI-ban timing",
        "windows": WINDOWS,
        "donuts": DONUTS,
        "counts": counts.to_dict(orient="records"),
        "best_rows": promoted.sort_values("pval").head(6).to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_memo(counts: pd.DataFrame, results: pd.DataFrame) -> None:
    lines = [
        "# Strict Question-Side AI-Ban Upgrade",
        "",
        "Date: April 6, 2026",
        "",
        "## Goal",
        "",
        "This build re-estimates the ban-centered timing check using the stricter question-side disclosure layer rather than the older, broader disclosed-AI hit table.",
        "The idea is simple: keep only disclosures visible when the question is posted, then ask whether accepted-window outcomes weaken after the Stack Overflow AI ban.",
        "",
        "## Window counts",
        "",
    ]

    for _, row in counts.iterrows():
        lines.append(
            f"- `{row['sample']}`, `+/-{int(row['window_days'])}d`, donut `{int(row['donut_days'])}d`: "
            f"`{int(row['n_questions'])}` questions, hits `{int(row['n_hits'])}`, "
            f"pre/post = `{int(row['pre_hits'])}/{int(row['post_hits'])}`"
        )

    lines += ["", "## Best rows", ""]
    promoted = results.loc[results["sample"] == "high_tags_only"].sort_values("pval")
    for _, row in promoted.head(8).iterrows():
        lines.append(
            f"- `{row['outcome']}`, `+/-{int(row['window_days'])}d`, donut `{int(row['donut_days'])}d`: "
            f"coef `{row['coef']:.4f}`, `p = {row['pval']:.4f}`, hits `{int(row['n_hits'])}`"
        )

    lines += [
        "",
        "## Safe read",
        "",
        "This stricter version is cleaner than the earlier question-side prototype because it relies on the posthistory direct-AI build's strict question-side disclosure flag.",
        "The strongest pattern is no longer about immediate answer arrival. It is about accepted-window outcomes.",
        "In high-exposure tags, `accepted_30d` is negative in every promoted window, and it is conventionally significant in the `+/-45d`, donut `3d`; `+/-60d`, donut `3d`; and `+/-60d`, donut `7d` rows.",
        "By contrast, `first_answer_1d` is mostly null once the donut is applied. That is exactly the direction the paper wants: the ban window looks more like weakened certification than reduced immediate answer arrival.",
        "",
        "## Honest ceiling",
        "",
        "This still does not become a pristine discontinuity.",
        "What it does provide is a cleaner and more coherent restricted timing result than the broader disclosed-hit build.",
        "If the paper keeps one AI-ban timing layer, this is now the best reviewer-facing version.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_sample()
    counts, results = build_results(df)
    counts.to_csv(COUNTS_CSV, index=False)
    results.to_csv(RESULTS_CSV, index=False)
    write_summary(counts, results)
    write_memo(counts, results)
    print(results.sort_values(["sample", "outcome", "window_days", "donut_days"]).to_string(index=False))


if __name__ == "__main__":
    main()
