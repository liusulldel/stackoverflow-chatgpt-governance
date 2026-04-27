from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

DURABILITY_ENTRANT_PANEL = PROCESSED / "who_still_answers_durability_entrant_panel.parquet"
SUBTYPE_CONSEQUENCE_PANEL = PROCESSED / "p1_jmis_subtype_consequence_panel.csv"

COND_PANEL_CSV = PROCESSED / "p1_established_repeat_core_conditional_panel.csv"
COND_RESULTS_CSV = PROCESSED / "p1_established_repeat_core_conditional_results.csv"
CORE_SPLIT_PANEL_CSV = PROCESSED / "p1_established_repeat_core_split_tag_month.csv"
CORE_SPLIT_RESULTS_CSV = PROCESSED / "p1_established_repeat_core_split_consequence_results.csv"
CORE_THRESHOLD_SUMMARY_CSV = PROCESSED / "p1_established_repeat_core_threshold_summary.csv"
READOUT_MD = PAPER / "p1_established_repeat_core_conditional_readout_2026-04-05.md"


def fit_wls(frame: pd.DataFrame, formula: str, weight_col: str, cluster_col: str = "primary_tag"):
    model_frame = frame.dropna(subset=[weight_col, cluster_col]).copy()
    y, X = dmatrices(formula, data=model_frame, return_type="dataframe", NA_action="drop")
    weights = model_frame.loc[X.index, weight_col].astype(float)
    groups = model_frame.loc[X.index, cluster_col]
    return sm.WLS(y, X, weights=weights).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True, "df_correction": True},
    )


def load_repeaters() -> tuple[pd.DataFrame, int, int]:
    entrants = pd.read_parquet(DURABILITY_ENTRANT_PANEL)
    entrants["entry_month"] = entrants["entry_month"].astype(str)
    month_order = {m: i + 1 for i, m in enumerate(sorted(entrants["entry_month"].dropna().unique()))}
    entrants["time_index"] = entrants["entry_month"].map(month_order)
    entrants["high_post"] = entrants["high_tag"] * entrants["post_chatgpt"]
    entrants["exposure_post"] = entrants["exposure_index"] * entrants["post_chatgpt"]

    established = entrants.loc[
        (entrants["entrant_type"] == "established_cross_tag") & (entrants["eligible_365d"] == 1)
    ].copy()
    repeaters = established.loc[established["one_shot_365d"] == 0].copy()

    answers_threshold = int(repeaters["answers_365d"].quantile(0.75))
    active_threshold = int(repeaters["active_months_365d"].quantile(0.75))
    repeaters["high_intensity_repeat_core"] = (
        (repeaters["answers_365d"] >= answers_threshold) & (repeaters["active_months_365d"] >= active_threshold)
    ).astype(int)
    repeaters["low_intensity_repeat_core"] = (1 - repeaters["high_intensity_repeat_core"]).astype(int)
    repeaters["log_answers_365d"] = np.log1p(repeaters["answers_365d"])
    return repeaters, answers_threshold, active_threshold


def build_conditional_panel(repeaters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = (
        repeaters.groupby(
            ["primary_tag", "entry_month", "time_index", "post_chatgpt", "exposure_index", "high_tag"],
            as_index=False,
        )
        .agg(
            n_repeaters=("high_intensity_repeat_core", "size"),
            high_intensity_repeat_core_share=("high_intensity_repeat_core", "mean"),
            mean_answers_365d_repeaters=("answers_365d", "mean"),
            mean_active_months_365d_repeaters=("active_months_365d", "mean"),
            mean_log_answers_365d_repeaters=("log_answers_365d", "mean"),
        )
    )
    panel["high_post"] = panel["high_tag"] * panel["post_chatgpt"]
    panel["exposure_post"] = panel["exposure_index"] * panel["post_chatgpt"]
    panel.to_csv(COND_PANEL_CSV, index=False)

    rows: list[dict[str, object]] = []
    for outcome in [
        "high_intensity_repeat_core_share",
        "mean_answers_365d_repeaters",
        "mean_active_months_365d_repeaters",
        "mean_log_answers_365d_repeaters",
    ]:
        for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
            formula = f"{outcome} ~ {term} + C(primary_tag):time_index + C(primary_tag) + C(entry_month)"
            model = fit_wls(panel, formula, "n_repeaters")
            rows.append(
                {
                    "family": family,
                    "outcome": outcome,
                    "term": term,
                    "coef": float(model.params.get(term, np.nan)),
                    "se": float(model.bse.get(term, np.nan)),
                    "pval": float(model.pvalues.get(term, np.nan)),
                    "nobs": int(model.nobs),
                    "mean_outcome": float(panel[outcome].mean()),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(COND_RESULTS_CSV, index=False)
    return panel, results


def build_core_split_panel(repeaters: pd.DataFrame) -> pd.DataFrame:
    entrants = pd.read_parquet(DURABILITY_ENTRANT_PANEL)
    entrants["entry_month"] = entrants["entry_month"].astype(str)
    month_order = {m: i + 1 for i, m in enumerate(sorted(entrants["entry_month"].dropna().unique()))}
    entrants["time_index"] = entrants["entry_month"].map(month_order)
    entrants["high_post"] = entrants["high_tag"] * entrants["post_chatgpt"]
    entrants["exposure_post"] = entrants["exposure_index"] * entrants["post_chatgpt"]
    eligible = entrants.loc[entrants["eligible_365d"] == 1].copy()

    repeater_flags = repeaters[
        [
            "primary_tag",
            "owner_user_id",
            "first_tag_answer_at",
            "high_intensity_repeat_core",
        ]
    ].copy()
    repeater_flags["owner_user_id"] = pd.to_numeric(repeater_flags["owner_user_id"], errors="coerce")
    eligible = eligible.merge(
        repeater_flags,
        on=["primary_tag", "owner_user_id", "first_tag_answer_at"],
        how="left",
    )
    eligible["high_intensity_repeat_core"] = eligible["high_intensity_repeat_core"].fillna(0).astype(int)

    split_panel = (
        eligible.groupby(
            ["primary_tag", "entry_month", "time_index", "post_chatgpt", "exposure_index", "high_tag"],
            as_index=False,
        )
        .agg(
            n_eligible_entrants=("owner_user_id", "size"),
            established_one_shot_share_365d=(
                "entrant_type",
                lambda s: np.mean(
                    (s == "established_cross_tag")
                    & (eligible.loc[s.index, "one_shot_365d"].fillna(0).astype(float) == 1.0)
                ),
            ),
            established_repeat_high_intensity_share_365d=(
                "entrant_type",
                lambda s: np.mean(
                    (s == "established_cross_tag")
                    & (eligible.loc[s.index, "one_shot_365d"].fillna(0).astype(float) == 0.0)
                    & (eligible.loc[s.index, "high_intensity_repeat_core"] == 1)
                ),
            ),
            established_repeat_low_intensity_share_365d=(
                "entrant_type",
                lambda s: np.mean(
                    (s == "established_cross_tag")
                    & (eligible.loc[s.index, "one_shot_365d"].fillna(0).astype(float) == 0.0)
                    & (eligible.loc[s.index, "high_intensity_repeat_core"] == 0)
                ),
            ),
        )
    )
    split_panel["established_repeat_total_share_365d"] = (
        split_panel["established_repeat_high_intensity_share_365d"]
        + split_panel["established_repeat_low_intensity_share_365d"]
    )
    split_panel["high_post"] = split_panel["high_tag"] * split_panel["post_chatgpt"]
    split_panel["exposure_post"] = split_panel["exposure_index"] * split_panel["post_chatgpt"]
    split_panel.to_csv(CORE_SPLIT_PANEL_CSV, index=False)
    return split_panel


def run_core_split_consequence(split_panel: pd.DataFrame) -> pd.DataFrame:
    consequence = pd.read_csv(SUBTYPE_CONSEQUENCE_PANEL)
    consequence["month_id"] = consequence["month_id"].astype(str)
    joined = consequence.merge(
        split_panel,
        left_on=["primary_tag", "month_id"],
        right_on=["primary_tag", "entry_month"],
        how="inner",
    )
    rows: list[dict[str, object]] = []
    outcomes = [
        ("any_answer_7d_rate", "any_answer_7d_denom"),
        ("first_positive_answer_latency_mean", "first_positive_answer_denom" if "first_positive_answer_denom" in joined.columns else "first_positive_answer_latency_denom"),
        ("accepted_cond_any_answer_30d_rate", "accepted_cond_any_answer_30d_denom"),
        ("accepted_vote_30d_rate", "accepted_vote_30d_denom"),
        ("first_answer_1d_rate_closure", "first_answer_1d_denom_closure"),
    ]
    for family, term in [("binary", "high_post_x"), ("continuous", "exposure_post_x")]:
        for outcome, weight_col in outcomes:
            formula = (
                f"{outcome} ~ {term} + established_one_shot_share_365d + "
                "established_repeat_high_intensity_share_365d + established_repeat_low_intensity_share_365d + "
                "C(primary_tag):time_index_x + C(primary_tag) + C(month_id)"
            )
            model = fit_wls(joined, formula, weight_col)
            for t in [
                term,
                "established_one_shot_share_365d",
                "established_repeat_high_intensity_share_365d",
                "established_repeat_low_intensity_share_365d",
            ]:
                rows.append(
                    {
                        "family": family,
                        "outcome": outcome,
                        "term": t.replace("_x", ""),
                        "coef": float(model.params.get(t, np.nan)),
                        "se": float(model.bse.get(t, np.nan)),
                        "pval": float(model.pvalues.get(t, np.nan)),
                        "nobs": int(model.nobs),
                        "mean_outcome": float(joined[outcome].mean()),
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(CORE_SPLIT_RESULTS_CSV, index=False)
    return results


def write_readout(
    answers_threshold: int,
    active_threshold: int,
    conditional_results: pd.DataFrame,
    split_results: pd.DataFrame,
) -> None:
    def pick(df: pd.DataFrame, **conds):
        mask = pd.Series(True, index=df.index)
        for k, v in conds.items():
            mask &= df[k] == v
        out = df.loc[mask]
        return None if out.empty else out.iloc[0]

    hi_share = pick(conditional_results, family="continuous", outcome="high_intensity_repeat_core_share", term="exposure_post")
    mean_answers = pick(conditional_results, family="continuous", outcome="mean_answers_365d_repeaters", term="exposure_post")
    mean_active = pick(conditional_results, family="continuous", outcome="mean_active_months_365d_repeaters", term="exposure_post")
    mean_log_answers = pick(conditional_results, family="continuous", outcome="mean_log_answers_365d_repeaters", term="exposure_post")

    cov_one = pick(split_results, family="continuous", outcome="any_answer_7d_rate", term="established_one_shot_share_365d")
    cov_hi = pick(split_results, family="continuous", outcome="any_answer_7d_rate", term="established_repeat_high_intensity_share_365d")
    cov_lo = pick(split_results, family="continuous", outcome="any_answer_7d_rate", term="established_repeat_low_intensity_share_365d")
    lat_one = pick(split_results, family="continuous", outcome="first_positive_answer_latency_mean", term="established_one_shot_share_365d")
    lat_hi = pick(split_results, family="continuous", outcome="first_positive_answer_latency_mean", term="established_repeat_high_intensity_share_365d")
    lat_lo = pick(split_results, family="continuous", outcome="first_positive_answer_latency_mean", term="established_repeat_low_intensity_share_365d")

    lines = [
        "# Established Repeat Core Conditional on Return",
        "",
        f"High-intensity repeat core is defined using fixed full-sample established-repeater thresholds: `answers_365d >= {answers_threshold}` and `active_months_365d >= {active_threshold}`.",
        "",
        "## Conditional-on-Return Surface",
        "",
        f"- `high_intensity_repeat_core_share`: coef `{hi_share['coef']:.4f}`, p `{hi_share['pval']:.4g}`.",
        f"- `mean_answers_365d_repeaters`: coef `{mean_answers['coef']:.4f}`, p `{mean_answers['pval']:.4g}`.",
        f"- `mean_active_months_365d_repeaters`: coef `{mean_active['coef']:.4f}`, p `{mean_active['pval']:.4g}`.",
        f"- `mean_log_answers_365d_repeaters`: coef `{mean_log_answers['coef']:.4f}`, p `{mean_log_answers['pval']:.4g}`.",
        "",
        "These coefficients answer the key conditional question: once established cross-tag entrants do return, do they become a more intense local core or not?",
        "",
        "## Consequence Allocation",
        "",
        f"- `any_answer_7d_rate`: one-shot `{cov_one['coef']:.4f}` (p `{cov_one['pval']:.4g}`), high-intensity repeat `{cov_hi['coef']:.4f}` (p `{cov_hi['pval']:.4g}`), low-intensity repeat `{cov_lo['coef']:.4f}` (p `{cov_lo['pval']:.4g}`).",
        f"- `first_positive_answer_latency_mean`: one-shot `{lat_one['coef']:.1f}` (p `{lat_one['pval']:.4g}`), high-intensity repeat `{lat_hi['coef']:.1f}` (p `{lat_hi['pval']:.4g}`), low-intensity repeat `{lat_lo['coef']:.1f}` (p `{lat_lo['pval']:.4g}`).",
        "",
        "## Read",
        "",
        "The right interpretation is bounded. This split asks whether the repeating established pool contains a more intense embedded core, and whether that core repairs coverage or only participates in slower deeper progression. It should not be read as causal local assimilation.",
    ]
    READOUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repeaters, answers_threshold, active_threshold = load_repeaters()
    threshold_summary = pd.DataFrame(
        [
            {
                "answers_365d_threshold": answers_threshold,
                "active_months_365d_threshold": active_threshold,
                "n_repeaters": int(len(repeaters)),
                "high_intensity_repeat_core_share": float(repeaters["high_intensity_repeat_core"].mean()),
            }
        ]
    )
    threshold_summary.to_csv(CORE_THRESHOLD_SUMMARY_CSV, index=False)
    _, conditional_results = build_conditional_panel(repeaters)
    split_panel = build_core_split_panel(repeaters)
    split_results = run_core_split_consequence(split_panel)
    write_readout(answers_threshold, active_threshold, conditional_results, split_results)


if __name__ == "__main__":
    main()
