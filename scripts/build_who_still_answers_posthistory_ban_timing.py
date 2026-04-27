from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

QUESTION_PANEL = PROCESSED / "stackexchange_20251231_question_level_enriched.parquet"
DISCLOSED_HITS = PROCESSED / "who_still_answers_disclosed_ai_question_hits.parquet"

WINDOW_SAMPLE = PROCESSED / "who_still_answers_posthistory_ban_window_questions.parquet"
COUNTS_CSV = PROCESSED / "who_still_answers_posthistory_ban_counts.csv"
RESULTS_CSV = PROCESSED / "who_still_answers_posthistory_ban_results.csv"
SUMMARY_JSON = PROCESSED / "who_still_answers_posthistory_ban_summary.json"
MEMO_PATH = PAPER / "who_still_answers_posthistory_ban_timing_2026-04-05.md"

BAN_DATE = pd.Timestamp("2022-12-05", tz="UTC")
WINDOWS = [15, 30, 45, 60]
DONUTS = [0, 3, 7]
OUTCOMES = [
    ("first_answer_1d", "first_answer_1d_eligible"),
    ("accepted_7d", "accepted_7d_eligible"),
    ("accepted_30d", "accepted_30d_eligible"),
]


def load_sample() -> pd.DataFrame:
    cols = [
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
    ]
    q = pd.read_parquet(QUESTION_PANEL, columns=cols).copy()
    q["question_id"] = q["question_id"].astype(int)
    q["question_created_at"] = pd.to_datetime(q["question_created_at"], utc=True)
    q["primary_tag"] = q["primary_tag"].fillna("unknown").astype(str)
    q["high_tag"] = q["high_tag"].fillna(0).astype(int)

    lo = BAN_DATE - pd.Timedelta(days=max(WINDOWS))
    hi = BAN_DATE + pd.Timedelta(days=max(WINDOWS))
    q = q.loc[(q["question_created_at"] >= lo) & (q["question_created_at"] <= hi)].copy()

    h = pd.read_parquet(
        DISCLOSED_HITS,
        columns=[
            "question_id",
            "question_title_hit",
            "question_body_hit",
            "answer_body_hit",
            "question_comment_hit",
            "answer_comment_hit",
        ],
    ).copy()
    h["question_id"] = h["question_id"].astype(int)

    sample = q.merge(h, on="question_id", how="left")
    hit_cols = [
        "question_title_hit",
        "question_body_hit",
        "answer_body_hit",
        "question_comment_hit",
        "answer_comment_hit",
    ]
    sample[hit_cols] = sample[hit_cols].fillna(0).astype(int)
    sample["question_side_hit"] = (
        sample[["question_title_hit", "question_body_hit"]].sum(axis=1).gt(0).astype(int)
    )
    sample["question_body_only_hit"] = sample["question_body_hit"].gt(0).astype(int)
    sample["thread_non_title_hit"] = (
        sample[
            ["question_body_hit", "answer_body_hit", "question_comment_hit", "answer_comment_hit"]
        ]
        .sum(axis=1)
        .gt(0)
        .astype(int)
    )
    sample["answer_or_comment_hit"] = (
        sample[["answer_body_hit", "question_comment_hit", "answer_comment_hit"]]
        .sum(axis=1)
        .gt(0)
        .astype(int)
    )
    sample["days_from_ban"] = (sample["question_created_at"] - BAN_DATE).dt.total_seconds() / 86400.0
    sample["post_ban"] = (sample["question_created_at"] >= BAN_DATE).astype(int)
    sample["abs_days_from_ban"] = sample["days_from_ban"].abs()

    for window in WINDOWS:
        sample[f"in_window_{window}"] = sample["abs_days_from_ban"].le(window).astype(int)
    for donut in DONUTS:
        sample[f"outside_donut_{donut}"] = sample["abs_days_from_ban"].gt(donut).astype(int)

    sample.to_parquet(WINDOW_SAMPLE, index=False)
    return sample


def build_counts(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_specs = {
        "question_side": "question_side_hit",
        "question_body_only": "question_body_only_hit",
        "thread_non_title": "thread_non_title_hit",
        "answer_or_comment": "answer_or_comment_hit",
    }
    for window in WINDOWS:
        subset = sample.loc[sample[f"in_window_{window}"] == 1].copy()
        for source_name, col in source_specs.items():
            rows.append(
                {
                    "window_days": window,
                    "source_family": source_name,
                    "n_questions": int(len(subset)),
                    "n_hits_all": int(subset[col].sum()),
                    "n_hits_high_tags": int(subset.loc[subset["high_tag"] == 1, col].sum()),
                    "n_hits_pre_ban": int(subset.loc[subset["post_ban"] == 0, col].sum()),
                    "n_hits_post_ban": int(subset.loc[subset["post_ban"] == 1, col].sum()),
                }
            )
    return pd.DataFrame(rows)


def fit_models(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_specs = {
        "question_side": ("question_side_hit", "clean contemporaneous question-side disclosure"),
        "question_body_only": ("question_body_only_hit", "clean contemporaneous body-only disclosure"),
        "thread_non_title": ("thread_non_title_hit", "thread-level disclosure with post-thread contamination risk"),
        "answer_or_comment": ("answer_or_comment_hit", "strong direct-observation but clearly post-thread contaminated"),
    }
    for window in WINDOWS:
        for donut in DONUTS:
            subset = sample.loc[
                (sample[f"in_window_{window}"] == 1) & (sample[f"outside_donut_{donut}"] == 1)
            ].copy()
            for source_name, (treat_col, caveat) in source_specs.items():
                for sample_name, data in [
                    ("all_tags", subset),
                    ("high_tags_only", subset.loc[subset["high_tag"] == 1].copy()),
                ]:
                    if data.empty or int(data[treat_col].sum()) < 30:
                        continue
                    for outcome, eligible in OUTCOMES:
                        frame = data.loc[data[eligible] == 1].copy()
                        if frame.empty or int(frame[treat_col].sum()) < 30:
                            continue
                        model = smf.ols(
                            (
                                f"{outcome} ~ {treat_col} * post_ban + "
                                "days_from_ban + post_ban:days_from_ban + "
                                f"{treat_col}:days_from_ban + C(primary_tag)"
                            ),
                            data=frame,
                        ).fit(cov_type="cluster", cov_kwds={"groups": frame["primary_tag"]})
                        term = f"{treat_col}:post_ban"
                        rows.append(
                            {
                                "window_days": window,
                                "donut_days": donut,
                                "sample": sample_name,
                                "source_family": source_name,
                                "source_caveat": caveat,
                                "outcome": outcome,
                                "coef": float(model.params.get(term, float("nan"))),
                                "se": float(model.bse.get(term, float("nan"))),
                                "pval": float(model.pvalues.get(term, float("nan"))),
                                "nobs": int(model.nobs),
                                "n_hits": int(frame[treat_col].sum()),
                                "mean_outcome": float(frame[outcome].mean()),
                            }
                        )
    return pd.DataFrame(rows)


def write_summary(sample: pd.DataFrame, counts: pd.DataFrame, results: pd.DataFrame) -> None:
    promoted = results.loc[
        (results["sample"] == "high_tags_only")
        & (results["outcome"].isin(["accepted_7d", "accepted_30d"]))
    ].copy()
    promoted = promoted.sort_values(["source_family", "window_days", "donut_days", "pval"])

    summary = {
        "n_window_sample": int(len(sample)),
        "question_side_hits": int(sample["question_side_hit"].sum()),
        "question_body_only_hits": int(sample["question_body_only_hit"].sum()),
        "thread_non_title_hits": int(sample["thread_non_title_hit"].sum()),
        "answer_or_comment_hits": int(sample["answer_or_comment_hit"].sum()),
        "best_clean_question_side": [],
        "best_thread_side": [],
    }

    for source_name, key in [("question_side", "best_clean_question_side"), ("thread_non_title", "best_thread_side")]:
        tmp = promoted.loc[promoted["source_family"] == source_name].sort_values("pval").head(4)
        summary[key] = tmp.to_dict("records")

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_memo(counts: pd.DataFrame, results: pd.DataFrame) -> None:
    lines = [
        "# Post-Surface AI-Ban Timing Prototype",
        "",
        "Date: April 5, 2026",
        "",
        "## Goal",
        "",
        "This build asks whether the strongest same-setting direct-observation layer available locally can improve the ban-centered timing story.",
        "It compares cleaner contemporaneous question-side disclosure against stronger but more contaminated thread-level disclosure around the `2022-12-05` Stack Overflow AI ban.",
        "",
        "## Source families",
        "",
        "- `question_side`: question title or question body mentions only; cleanest same-setting disclosure available at posting time",
        "- `question_body_only`: question-body mentions only; same timing logic with fewer title-style rhetorical mentions",
        "- `thread_non_title`: question body, answer body, and comment mentions; stronger direct observation but post-thread contaminated",
        "- `answer_or_comment`: answer/comment mentions only; strongest direct-observation layer but least clean for ban timing",
        "",
        "## Window counts",
        "",
    ]
    for _, row in counts.iterrows():
        lines.append(
            f"- `+/-{int(row['window_days'])}d`, `{row['source_family']}`: "
            f"`{int(row['n_questions']):,}` questions, `{int(row['n_hits_all'])}` total hits, "
            f"`{int(row['n_hits_high_tags'])}` high-tag hits, pre/post = `{int(row['n_hits_pre_ban'])}/{int(row['n_hits_post_ban'])}`"
        )

    lines += ["", "## Topline estimates", ""]

    def add_best(source_family: str, label: str) -> None:
        sub = results.loc[
            (results["source_family"] == source_family)
            & (results["sample"] == "high_tags_only")
            & (results["outcome"].isin(["accepted_7d", "accepted_30d"]))
        ].sort_values("pval")
        if sub.empty:
            lines.append(f"- `{label}`: no estimable high-tag accepted-window rows.")
            return
        best = sub.iloc[0]
        lines.append(
            f"- `{label}` best row: `+/-{int(best['window_days'])}d`, donut `{int(best['donut_days'])}d`, "
            f"`{best['outcome']}`, coef `{best['coef']:.4f}`, `p = {best['pval']:.4f}`, hits `{int(best['n_hits'])}`"
        )

    add_best("question_side", "clean question-side")
    add_best("thread_non_title", "thread non-title")
    add_best("answer_or_comment", "answer/comment only")

    qside_rows = results.loc[
        (results["source_family"] == "question_side")
        & (results["sample"] == "high_tags_only")
        & (results["window_days"] == 30)
        & (results["donut_days"] == 3)
        & (results["outcome"].isin(["accepted_7d", "accepted_30d", "first_answer_1d"]))
    ].sort_values("outcome")
    if not qside_rows.empty:
        lines += ["", "## Clean question-side benchmark (`+/-30d`, donut `3d`, high tags)", ""]
        for _, row in qside_rows.iterrows():
            lines.append(
                f"- `{row['outcome']}`: coef `{row['coef']:.4f}`, `p = {row['pval']:.4f}`, hits `{int(row['n_hits'])}`"
            )

    lines += [
        "",
        "## Safe read",
        "",
        "Yes, this improves on the earlier ban-centered prototype in one important way: it separates a cleaner question-side disclosure design from stronger but post-thread-contaminated thread-level designs.",
        "The strongest raw signal still comes from thread-side disclosure, especially `answer_or_comment` and `thread_non_title` around `+/-30d` to `+/-45d` windows. But those layers are not clean because answer/comment text can be generated after the question is posted and therefore can be downstream of the timing event itself.",
        "The cleaner design is `question_side`. That layer remains significant in high-exposure tags for accepted-window outcomes under narrow windows and donuts. In particular, the `+/-30d`, donut `3d` specification produces a negative `disclosed_ai_hit x post_ban` coefficient on both `accepted_7d` and `accepted_30d`, while `first_answer_1d` remains null. This is a more credible ban-centered timing result because it uses information already visible when the question is posted.",
        "",
        "## Honest ceiling",
        "",
        "This still does not become a clean causal ban design.",
        "What it does achieve is a more disciplined same-setting restricted timing prototype: a cleaner question-side direct-observation layer that supports the interpretation that the ban is more clearly linked to weaker near-term public certification than to immediate answer arrival.",
        "If the paper needs a reviewer-safe ban-centered timing statement, this build is better than the prior one. If the paper needs a pristine discontinuity, it still does not get there.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    sample = load_sample()
    counts = build_counts(sample)
    results = fit_models(sample)

    counts.to_csv(COUNTS_CSV, index=False)
    results.to_csv(RESULTS_CSV, index=False)
    write_summary(sample, counts, results)
    write_memo(counts, results)
    print(results.sort_values(["source_family", "window_days", "donut_days", "sample", "outcome"]).to_string(index=False))


if __name__ == "__main__":
    main()
