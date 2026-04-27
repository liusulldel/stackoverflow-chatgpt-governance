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

WINDOW_SAMPLE = PROCESSED / "who_still_answers_ai_ban_window_questions.parquet"
COUNTS_CSV = PROCESSED / "who_still_answers_ai_ban_counts.csv"
RESULTS_CSV = PROCESSED / "who_still_answers_ai_ban_results.csv"
DIAGNOSTICS_CSV = PROCESSED / "who_still_answers_ai_ban_diagnostics.csv"
SUMMARY_JSON = PROCESSED / "who_still_answers_ai_ban_summary.json"

BAN_DATE = pd.Timestamp("2022-12-05", tz="UTC")
WINDOWS = [7, 14, 21, 28, 35]
DONUTS = [0, 1, 3, 5]
EXPOSURE_BINS = 5
MAX_CONTROL_WEIGHT = 5.0


def load_window_sample() -> pd.DataFrame:
    cols = [
        "question_id",
        "question_created_at",
        "primary_tag",
        "high_tag",
        "exposure_index",
        "first_answer_1d",
        "first_answer_1d_eligible",
        "accepted_30d",
        "accepted_30d_eligible",
    ]
    q = pd.read_parquet(QUESTION_PANEL, columns=cols).copy()
    q["question_id"] = q["question_id"].astype(int)
    q["question_created_at"] = pd.to_datetime(q["question_created_at"], utc=True)

    lo = BAN_DATE - pd.Timedelta(days=max(WINDOWS))
    hi = BAN_DATE + pd.Timedelta(days=max(WINDOWS))
    q = q.loc[(q["question_created_at"] >= lo) & (q["question_created_at"] <= hi)].copy()

    hits = pd.read_parquet(
        DISCLOSED_HITS,
        columns=[
            "question_id",
            "question_title_hit",
            "question_body_hit",
            "answer_body_hit",
            "question_comment_hit",
            "answer_comment_hit",
            "any_comment_hit",
        ],
    ).copy()
    hits["question_id"] = hits["question_id"].astype(int)
    hits["disclosed_ai_hit"] = 1
    hits["thread_source_count"] = (
        hits[
            [
                "question_title_hit",
                "question_body_hit",
                "answer_body_hit",
                "question_comment_hit",
                "answer_comment_hit",
            ]
        ]
        .fillna(0)
        .sum(axis=1)
    )

    sample = q.merge(hits, on="question_id", how="left")
    fill_cols = [
        "question_title_hit",
        "question_body_hit",
        "answer_body_hit",
        "question_comment_hit",
        "answer_comment_hit",
        "any_comment_hit",
        "disclosed_ai_hit",
        "thread_source_count",
    ]
    sample[fill_cols] = sample[fill_cols].fillna(0)
    sample["days_from_ban"] = (sample["question_created_at"] - BAN_DATE).dt.total_seconds() / 86400.0
    sample["post_ban"] = (sample["question_created_at"] >= BAN_DATE).astype(int)
    sample["abs_days_from_ban"] = sample["days_from_ban"].abs()
    sample["high_tag"] = sample["high_tag"].fillna(0).astype(int)
    sample["primary_tag"] = sample["primary_tag"].fillna("unknown").astype(str)
    exposure_median = sample["exposure_index"].median()
    if pd.isna(exposure_median):
        exposure_median = 0.0
    sample["exposure_index"] = sample["exposure_index"].fillna(exposure_median)

    exposure_rank = sample["exposure_index"].rank(method="first")
    try:
        sample["exposure_bin"] = pd.qcut(
            exposure_rank,
            q=min(EXPOSURE_BINS, exposure_rank.nunique()),
            labels=False,
            duplicates="drop",
        ).astype(int)
    except ValueError:
        sample["exposure_bin"] = pd.cut(
            exposure_rank,
            bins=min(EXPOSURE_BINS, exposure_rank.nunique()),
            labels=False,
            include_lowest=True,
        ).astype(int)

    for window in WINDOWS:
        sample[f"in_window_{window}"] = sample["abs_days_from_ban"].le(window).astype(int)
    for donut in DONUTS:
        sample[f"outside_donut_{donut}"] = sample["abs_days_from_ban"].gt(donut).astype(int)

    sample.to_parquet(WINDOW_SAMPLE, index=False)
    return sample


def apply_matched_controls(subset: pd.DataFrame) -> pd.DataFrame:
    strata_cols = ["primary_tag", "exposure_bin"]
    strata = (
        subset.groupby(strata_cols)["disclosed_ai_hit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "treated", "count": "total"})
    )
    strata["control"] = strata["total"] - strata["treated"]
    valid = strata.loc[(strata["treated"] > 0) & (strata["control"] > 0)].reset_index()
    if valid.empty:
        return subset.iloc[0:0].copy()
    subset = subset.merge(valid[strata_cols], on=strata_cols, how="inner")
    weights = (
        subset.groupby(strata_cols)["disclosed_ai_hit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "treated", "count": "total"})
    )
    weights["control"] = weights["total"] - weights["treated"]
    weights["control_weight"] = (weights["treated"] / weights["control"]).clip(upper=MAX_CONTROL_WEIGHT)
    subset = subset.merge(weights[["control_weight"]], on=strata_cols, how="left")
    subset["match_weight"] = 1.0
    subset.loc[subset["disclosed_ai_hit"] == 0, "match_weight"] = subset.loc[
        subset["disclosed_ai_hit"] == 0, "control_weight"
    ]
    return subset


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    w = weights.astype(float)
    return float((series * w).sum() / w.sum()) if w.sum() else float("nan")


def weighted_var(series: pd.Series, weights: pd.Series) -> float:
    w = weights.astype(float)
    if w.sum() == 0:
        return float("nan")
    mean = weighted_mean(series, w)
    return float(((w * (series - mean) ** 2).sum()) / w.sum())


def standardized_mean_diff(treated: pd.Series, control: pd.Series, wt_t: pd.Series, wt_c: pd.Series) -> float:
    mean_t = weighted_mean(treated, wt_t)
    mean_c = weighted_mean(control, wt_c)
    var_t = weighted_var(treated, wt_t)
    var_c = weighted_var(control, wt_c)
    denom = (0.5 * (var_t + var_c)) ** 0.5
    if denom == 0 or pd.isna(denom):
        return float("nan")
    return float((mean_t - mean_c) / denom)


def fit_models(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        for donut in DONUTS:
            if donut >= window:
                continue
            subset = sample.loc[
                (sample[f"in_window_{window}"] == 1) & (sample[f"outside_donut_{donut}"] == 1)
            ].copy()
            for sample_name, data in [
                ("matched_all_tags", apply_matched_controls(subset)),
                ("matched_high_tags", apply_matched_controls(subset.loc[subset["high_tag"] == 1].copy())),
            ]:
                if data.empty or data["disclosed_ai_hit"].sum() < 25:
                    continue
                for outcome, eligible in [
                    ("first_answer_1d", "first_answer_1d_eligible"),
                    ("accepted_30d", "accepted_30d_eligible"),
                ]:
                    frame = data.loc[data[eligible] == 1].copy()
                    if frame.empty or frame["disclosed_ai_hit"].sum() < 25:
                        continue
                    model = smf.wls(
                        (
                            f"{outcome} ~ disclosed_ai_hit * post_ban + "
                            "days_from_ban + post_ban:days_from_ban + disclosed_ai_hit:days_from_ban + "
                            "C(primary_tag)"
                        ),
                        data=frame,
                        weights=frame["match_weight"],
                    ).fit(cov_type="cluster", cov_kwds={"groups": frame["primary_tag"]})
                    term = "disclosed_ai_hit:post_ban"
                    rows.append(
                        {
                            "window_days": window,
                            "donut_days": donut,
                            "sample": sample_name,
                            "outcome": outcome,
                            "coef": float(model.params.get(term, float("nan"))),
                            "se": float(model.bse.get(term, float("nan"))),
                            "pval": float(model.pvalues.get(term, float("nan"))),
                            "nobs": int(model.nobs),
                            "n_disclosed": int(frame["disclosed_ai_hit"].sum()),
                            "mean_outcome": float(frame[outcome].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def build_diagnostics(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        for donut in DONUTS:
            if donut >= window:
                continue
            subset = sample.loc[
                (sample[f"in_window_{window}"] == 1) & (sample[f"outside_donut_{donut}"] == 1)
            ].copy()
            for sample_name, data in [
                ("matched_all_tags", apply_matched_controls(subset)),
                ("matched_high_tags", apply_matched_controls(subset.loc[subset["high_tag"] == 1].copy())),
            ]:
                if data.empty:
                    continue
                treated = data.loc[data["disclosed_ai_hit"] == 1].copy()
                control = data.loc[data["disclosed_ai_hit"] == 0].copy()
                if treated.empty or control.empty:
                    continue
                wt_t = treated["match_weight"]
                wt_c = control["match_weight"]
                rows.append(
                    {
                        "window_days": window,
                        "donut_days": donut,
                        "sample": sample_name,
                        "n_questions": int(len(data)),
                        "n_disclosed": int(treated["disclosed_ai_hit"].sum()),
                        "n_controls": int(len(control)),
                        "mean_exposure_disclosed": weighted_mean(treated["exposure_index"], wt_t),
                        "mean_exposure_controls": weighted_mean(control["exposure_index"], wt_c),
                        "smd_exposure_index": standardized_mean_diff(
                            treated["exposure_index"], control["exposure_index"], wt_t, wt_c
                        ),
                        "mean_abs_days_disclosed": weighted_mean(treated["abs_days_from_ban"], wt_t),
                        "mean_abs_days_controls": weighted_mean(control["abs_days_from_ban"], wt_c),
                        "smd_abs_days": standardized_mean_diff(
                            treated["abs_days_from_ban"], control["abs_days_from_ban"], wt_t, wt_c
                        ),
                        "share_post_ban_disclosed": weighted_mean(treated["post_ban"], wt_t),
                        "share_post_ban_controls": weighted_mean(control["post_ban"], wt_c),
                        "smd_post_ban": standardized_mean_diff(
                            treated["post_ban"], control["post_ban"], wt_t, wt_c
                        ),
                        "avg_control_weight": float(control["match_weight"].mean()),
                        "max_control_weight": float(control["match_weight"].max()),
                    }
                )
    return pd.DataFrame(rows)


def build_counts(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        subset = sample.loc[sample[f"in_window_{window}"] == 1].copy()
        rows.append(
            {
                "window_days": window,
                "n_questions": int(len(subset)),
                "n_disclosed_ai": int(subset["disclosed_ai_hit"].sum()),
                "n_question_comment_hits": int(subset["question_comment_hit"].sum()),
                "n_answer_comment_hits": int(subset["answer_comment_hit"].sum()),
                "n_answer_body_hits": int(subset["answer_body_hit"].sum()),
                "avg_thread_source_count_disclosed": float(
                    subset.loc[subset["disclosed_ai_hit"] == 1, "thread_source_count"].mean()
                )
                if int(subset["disclosed_ai_hit"].sum()) > 0
                else 0.0,
                "share_disclosed_ai": float(subset["disclosed_ai_hit"].mean()),
                "share_post_ban": float(subset["post_ban"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    sample = load_window_sample()
    counts = build_counts(sample)
    results = fit_models(sample)
    diagnostics = build_diagnostics(sample)

    counts.to_csv(COUNTS_CSV, index=False)
    results.to_csv(RESULTS_CSV, index=False)
    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "n_sample": int(len(sample)),
                "n_disclosed_ai": int(sample["disclosed_ai_hit"].sum()),
                "n_any_comment_hit": int(sample["any_comment_hit"].sum()),
                "windows": counts.to_dict("records"),
                "diagnostics_rows": int(len(diagnostics)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
