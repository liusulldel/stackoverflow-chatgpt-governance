"""
Build minimal, submission-legible methods exhibits for the P1 JMIS package.

Goal: promote few-cluster and timing discipline into visible artifacts without
overclaiming causal identification.

Outputs (written under processed/):
- p1_jmis_leave_one_domain_out.csv
- p1_jmis_leave_one_domain_out_summary.csv
- p1_jmis_timing_series_high_low.csv
- p1_jmis_pretrend_slope_tests.csv

Outputs (written under paper/):
- p1_jmis_promoted_methods_exhibits_2026-04-04.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(r"D:/AI alignment/projects/stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    panel: str  # "residual" or "subtype"
    outcome: str
    formula: str
    term: str
    weight_col: str


def _wls_fit(df: pd.DataFrame, spec: ModelSpec):
    w = df[spec.weight_col].astype(float).values
    # Guard: weights must be positive; drop zero-weight rows.
    keep = np.isfinite(w) & (w > 0)
    df2 = df.loc[keep].copy()
    w2 = w[keep]
    model = smf.wls(spec.formula, data=df2, weights=w2)
    return model.fit()


def build_leave_one_domain_out(residual_panel: pd.DataFrame, subtype_panel: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ModelSpec(
            model_id="T2_residual_queue_complexity",
            panel="residual",
            outcome="residual_queue_complexity_index_mean",
            formula="residual_queue_complexity_index_mean ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="high_post",
            weight_col="n_questions",
        ),
        ModelSpec(
            model_id="T3_brand_new_share",
            panel="residual",
            outcome="brand_new_platform_share",
            formula="brand_new_platform_share ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="high_post",
            weight_col="n_new_answerers_profiles",
        ),
        ModelSpec(
            model_id="T3_low_tenure_share",
            panel="residual",
            outcome="low_tenure_existing_share",
            formula="low_tenure_existing_share ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="high_post",
            weight_col="n_new_answerers_profiles",
        ),
        ModelSpec(
            model_id="T4A_first_answer_1d",
            panel="subtype",
            outcome="first_answer_1d_rate_closure",
            formula="first_answer_1d_rate_closure ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="high_post",
            weight_col="first_answer_1d_denom_closure",
        ),
        ModelSpec(
            model_id="T4A_accepted_vote_30d",
            panel="subtype",
            outcome="accepted_vote_30d_rate",
            formula="accepted_vote_30d_rate ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="high_post",
            weight_col="accepted_vote_30d_denom",
        ),
        ModelSpec(
            model_id="T4B_brand_new_to_any_answer_7d",
            panel="subtype",
            outcome="any_answer_7d_rate",
            formula="any_answer_7d_rate ~ high_post + brand_new_platform_share + low_tenure_existing_share + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="brand_new_platform_share",
            weight_col="any_answer_7d_denom",
        ),
        ModelSpec(
            model_id="T4B_established_to_any_answer_7d",
            panel="subtype",
            outcome="any_answer_7d_rate",
            formula="any_answer_7d_rate ~ high_post + established_cross_tag_share + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="established_cross_tag_share",
            weight_col="any_answer_7d_denom",
        ),
    ]

    tags = sorted(residual_panel["primary_tag"].unique().tolist())

    rows = []
    for spec in specs:
        df = residual_panel if spec.panel == "residual" else subtype_panel

        # Full-sample fit as baseline row.
        fit_full = _wls_fit(df, spec)
        rows.append(
            {
                "model_id": spec.model_id,
                "panel": spec.panel,
                "outcome": spec.outcome,
                "term": spec.term,
                "leaveout_primary_tag": "__full__",
                "coef": float(fit_full.params.get(spec.term, np.nan)),
                "se": float(fit_full.bse.get(spec.term, np.nan)),
                "pval": float(fit_full.pvalues.get(spec.term, np.nan)),
                "nobs": int(fit_full.nobs),
                "weight_col": spec.weight_col,
                "formula": spec.formula,
            }
        )

        for tag in tags:
            df_lo = df[df["primary_tag"] != tag].copy()
            fit = _wls_fit(df_lo, spec)
            rows.append(
                {
                    "model_id": spec.model_id,
                    "panel": spec.panel,
                    "outcome": spec.outcome,
                    "term": spec.term,
                    "leaveout_primary_tag": tag,
                    "coef": float(fit.params.get(spec.term, np.nan)),
                    "se": float(fit.bse.get(spec.term, np.nan)),
                    "pval": float(fit.pvalues.get(spec.term, np.nan)),
                    "nobs": int(fit.nobs),
                    "weight_col": spec.weight_col,
                    "formula": spec.formula,
                }
            )

    return pd.DataFrame(rows)


def summarize_leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    def _summ(g):
        g2 = g[g["leaveout_primary_tag"] != "__full__"]
        coefs = g2["coef"].astype(float)
        return pd.Series(
            {
                "full_coef": float(g[g["leaveout_primary_tag"] == "__full__"]["coef"].iloc[0]),
                "n_leaveouts": int(len(g2)),
                "n_positive": int((coefs > 0).sum()),
                "n_negative": int((coefs < 0).sum()),
                "min_coef": float(coefs.min()),
                "median_coef": float(coefs.median()),
                "max_coef": float(coefs.max()),
            }
        )

    return df.groupby(["model_id", "panel", "outcome", "term"], as_index=False).apply(_summ).reset_index(drop=True)


def build_timing_series(residual_panel: pd.DataFrame, subtype_panel: pd.DataFrame) -> pd.DataFrame:
    # Merge high_tag and post_chatgpt into the subtype panel (it does not carry them).
    tag_map = residual_panel[["primary_tag", "high_tag"]].drop_duplicates()
    month_map = residual_panel[["month_id", "time_index", "post_chatgpt"]].drop_duplicates()
    subtype = subtype_panel.merge(tag_map, on="primary_tag", how="left").merge(month_map, on=["month_id", "time_index"], how="left")

    # Outcomes to plot/diagnose. Keep this minimal and promoted.
    timing_outcomes = [
        ("residual_queue_complexity_index_mean", residual_panel, "n_questions"),
        ("brand_new_platform_share", residual_panel, "n_new_answerers_profiles"),
        ("first_answer_1d_rate_closure", subtype, "first_answer_1d_denom_closure"),
        ("accepted_vote_30d_rate", subtype, "accepted_vote_30d_denom"),
    ]

    series_rows = []
    for outcome, df, wcol in timing_outcomes:
        for month_id, g in df.groupby("month_id"):
            for high_tag in [0, 1]:
                gg = g[g["high_tag"] == high_tag]
                if gg.empty:
                    continue
                w = gg[wcol].astype(float)
                y = gg[outcome].astype(float)
                mean = float(np.average(y, weights=w))
                series_rows.append(
                    {
                        "outcome": outcome,
                        "month_id": month_id,
                        "high_tag": int(high_tag),
                        "weighted_mean": mean,
                        "weight_col": wcol,
                        "n_domains": int(gg["primary_tag"].nunique()),
                    }
                )

    series = pd.DataFrame(series_rows)
    # Add a convenience high-minus-low diff.
    wide = series.pivot_table(index=["outcome", "month_id"], columns="high_tag", values="weighted_mean")
    wide = wide.rename(columns={0: "low_tag_mean", 1: "high_tag_mean"}).reset_index()
    wide["high_minus_low"] = wide["high_tag_mean"] - wide["low_tag_mean"]
    return wide.sort_values(["outcome", "month_id"]).reset_index(drop=True)


def build_pretrend_slope_tests(residual_panel: pd.DataFrame, subtype_panel: pd.DataFrame) -> pd.DataFrame:
    # Pretrend test: outcome ~ C(domain) + C(month) + high_tag:time_index (pre period only).
    # This is not a shock design. It's a minimal discipline check for differential pre slopes.
    tag_map = residual_panel[["primary_tag", "high_tag"]].drop_duplicates()
    month_map = residual_panel[["month_id", "time_index", "post_chatgpt"]].drop_duplicates()
    subtype = subtype_panel.merge(tag_map, on="primary_tag", how="left").merge(month_map, on=["month_id", "time_index"], how="left")

    tests = [
        ("residual_queue_complexity_index_mean", residual_panel, "n_questions"),
        ("brand_new_platform_share", residual_panel, "n_new_answerers_profiles"),
        ("first_answer_1d_rate_closure", subtype, "first_answer_1d_denom_closure"),
        ("accepted_vote_30d_rate", subtype, "accepted_vote_30d_denom"),
    ]

    rows = []
    for outcome, df, wcol in tests:
        pre = df[df["post_chatgpt"] == 0].copy()
        if pre.empty:
            continue
        # Ensure high_tag exists.
        if "high_tag" not in pre.columns:
            pre = pre.merge(tag_map, on="primary_tag", how="left")
        formula = f"{outcome} ~ C(primary_tag) + C(month_id) + high_tag:time_index"
        fit = _wls_fit(pre, ModelSpec("pretrend", "na", outcome, formula, "high_tag:time_index", wcol))
        term = "high_tag:time_index"
        rows.append(
            {
                "outcome": outcome,
                "term": term,
                "coef": float(fit.params.get(term, np.nan)),
                "se": float(fit.bse.get(term, np.nan)),
                "pval": float(fit.pvalues.get(term, np.nan)),
                "nobs": int(fit.nobs),
                "weight_col": wcol,
                "formula": formula,
            }
        )

    return pd.DataFrame(rows)


def main():
    residual_path = PROCESSED / "p1_jmis_residual_queue_panel.csv"
    subtype_path = PROCESSED / "p1_jmis_subtype_consequence_panel.csv"
    residual = pd.read_csv(residual_path)
    subtype = pd.read_csv(subtype_path)

    loo = build_leave_one_domain_out(residual, subtype)
    loo_summary = summarize_leave_one_out(loo)
    timing = build_timing_series(residual, subtype)
    pretrend = build_pretrend_slope_tests(residual, subtype)

    out_loo = PROCESSED / "p1_jmis_leave_one_domain_out.csv"
    out_loo_sum = PROCESSED / "p1_jmis_leave_one_domain_out_summary.csv"
    out_timing = PROCESSED / "p1_jmis_timing_series_high_low.csv"
    out_pretrend = PROCESSED / "p1_jmis_pretrend_slope_tests.csv"

    loo.to_csv(out_loo, index=False)
    loo_summary.to_csv(out_loo_sum, index=False)
    timing.to_csv(out_timing, index=False)
    pretrend.to_csv(out_pretrend, index=False)

    # Lightweight paper-facing memo.
    memo_path = PAPER / "p1_jmis_promoted_methods_exhibits_2026-04-04.md"
    with memo_path.open("w", encoding="utf-8") as f:
        f.write("# P1 Promoted Methods Exhibits (Few-Cluster + Timing Discipline)\n\n")
        f.write("Date: `2026-04-04`\n\n")
        f.write("This memo summarizes minimal, submission-legible methods exhibits generated from existing harmonized panels.\n\n")
        f.write("## Few-Cluster Sensitivity (Leave-One-Domain-Out)\n\n")
        f.write("Files:\n\n")
        f.write(f"- `{out_loo.as_posix()}`\n")
        f.write(f"- `{out_loo_sum.as_posix()}`\n\n")
        f.write("Summary (sign stability across 16 leaveouts):\n\n")
        f.write(loo_summary.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Timing Discipline (Descriptive High-Low Series)\n\n")
        f.write("File:\n\n")
        f.write(f"- `{out_timing.as_posix()}`\n\n")
        f.write("This series is descriptive. It is intended to discipline periodization language, not to justify a clean shock.\n\n")
        f.write("## Differential Pretrend Slopes (Pre-Period Only)\n\n")
        f.write("File:\n\n")
        f.write(f"- `{out_pretrend.as_posix()}`\n\n")
        if not pretrend.empty:
            f.write(pretrend.to_markdown(index=False))
            f.write("\n")


if __name__ == "__main__":
    main()

