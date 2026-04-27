from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

PANEL_CSV = PROCESSED / "p1_jmis_residual_queue_panel.csv"

FEW_CLUSTER_CSV = PROCESSED / "p1_jmis_few_cluster_surface.csv"
TIMING_PANEL_CSV = PROCESSED / "p1_jmis_timing_discipline_panel.csv"
TIMING_SERIES_CSV = PROCESSED / "p1_jmis_timing_series.csv"

READOUT_MD = PAPER / "p1_jmis_promoted_methods_exhibits_2026-04-04.md"


def fit_weighted(frame: pd.DataFrame, formula: str, weight_col: str):
    outcome = formula.split("~", 1)[0].strip()
    rhs_vars = []
    for token in formula.replace("~", "+").split("+"):
        token = token.strip()
        if not token or token == "1" or token.startswith("C("):
            continue
        rhs_vars.append(token)
    needed = [outcome, weight_col, "primary_tag"] + [v for v in rhs_vars if v in frame.columns]
    sample = frame.dropna(subset=[c for c in needed if c in frame.columns]).copy()
    sample = sample.loc[sample[weight_col].astype(float) > 0].copy()
    fit = smf.wls(formula, data=sample, weights=sample[weight_col].astype(float)).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["primary_tag"]},
    )
    return fit, sample


def leave_one_domain_out(
    frame: pd.DataFrame,
    formula: str,
    weight_col: str,
    term: str,
) -> dict[str, object]:
    tags = sorted(frame["primary_tag"].dropna().unique())
    rows: list[dict[str, object]] = []
    for tag in tags:
        sample = frame.loc[frame["primary_tag"] != tag].copy()
        fit, _ = fit_weighted(sample, formula, weight_col)
        coef = float(fit.params.get(term, np.nan))
        pval = float(fit.pvalues.get(term, np.nan))
        rows.append(
            {
                "dropped_tag": tag,
                "coef": coef,
                "pval": pval,
                "sign": np.sign(coef) if np.isfinite(coef) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    nonnull = out["coef"].dropna()
    positive = int((nonnull > 0).sum())
    negative = int((nonnull < 0).sum())
    return {
        "leave_one_out_runs": int(len(out)),
        "leave_one_out_positive": positive,
        "leave_one_out_negative": negative,
        "leave_one_out_same_sign_share": float(max(positive, negative) / len(out)) if len(out) else np.nan,
        "leave_one_out_min_coef": float(nonnull.min()) if not nonnull.empty else np.nan,
        "leave_one_out_max_coef": float(nonnull.max()) if not nonnull.empty else np.nan,
    }


def build_few_cluster_surface(panel: pd.DataFrame) -> pd.DataFrame:
    specs = [
        {
            "layer": "queue",
            "outcome": "residual_queue_complexity_index_mean",
            "term": "high_post",
            "weight_col": "n_questions",
            "formula": "residual_queue_complexity_index_mean ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
        },
        {
            "layer": "resorting",
            "outcome": "brand_new_platform_share",
            "term": "high_post",
            "weight_col": "n_new_answerers_profiles",
            "formula": "brand_new_platform_share ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
        },
        {
            "layer": "response",
            "outcome": "first_answer_1d_rate_closure",
            "term": "high_post",
            "weight_col": "first_answer_1d_denom_closure",
            "formula": "first_answer_1d_rate_closure ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
        },
        {
            "layer": "settlement",
            "outcome": "accepted_vote_30d_rate",
            "term": "high_post",
            "weight_col": "accepted_vote_30d_denom",
            "formula": "accepted_vote_30d_rate ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
        },
    ]
    rows: list[dict[str, object]] = []
    for spec in specs:
        fit, sample = fit_weighted(panel, spec["formula"], spec["weight_col"])
        loo = leave_one_domain_out(sample, spec["formula"], spec["weight_col"], spec["term"])
        rows.append(
            {
                "layer": spec["layer"],
                "outcome": spec["outcome"],
                "term": spec["term"],
                "coef": float(fit.params[spec["term"]]),
                "se": float(fit.bse[spec["term"]]),
                "pval": float(fit.pvalues[spec["term"]]),
                "clusters": int(sample["primary_tag"].nunique()),
                "nobs": int(fit.nobs),
                **loo,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(FEW_CLUSTER_CSV, index=False)
    return out


def build_timing_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = panel.copy()
    work["month_dt"] = pd.to_datetime(work["month_id"] + "-01", utc=True)
    pre = work.loc[work["post_chatgpt"] == 0].copy()
    post = work.loc[work["post_chatgpt"] == 1].copy()

    specs = [
        ("queue", "residual_queue_complexity_index_mean", "n_questions"),
        ("resorting", "brand_new_platform_share", "n_new_answerers_profiles"),
        ("response", "first_answer_1d_rate_closure", "first_answer_1d_denom_closure"),
        ("settlement", "accepted_vote_30d_rate", "accepted_vote_30d_denom"),
    ]

    rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    for layer, outcome, weight_col in specs:
        pre_formula = f"{outcome} ~ high_tag:time_index + C(primary_tag) + C(month_id)"
        pre_fit, pre_sample = fit_weighted(pre, pre_formula, weight_col)
        full_formula = f"{outcome} ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)"
        full_fit, full_sample = fit_weighted(work, full_formula, weight_col)
        rows.append(
            {
                "layer": layer,
                "outcome": outcome,
                "pretrend_term": "high_tag:time_index",
                "pretrend_coef": float(pre_fit.params.get("high_tag:time_index", np.nan)),
                "pretrend_pval": float(pre_fit.pvalues.get("high_tag:time_index", np.nan)),
                "promoted_term": "high_post",
                "promoted_coef": float(full_fit.params.get("high_post", np.nan)),
                "promoted_pval": float(full_fit.pvalues.get("high_post", np.nan)),
                "pre_clusters": int(pre_sample["primary_tag"].nunique()),
                "full_clusters": int(full_sample["primary_tag"].nunique()),
            }
        )

        grouped = (
            work.dropna(subset=[outcome, weight_col, "high_tag"])
            .groupby(["month_id", "month_dt", "high_tag"], as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "weighted_mean": np.average(
                            g[outcome].astype(float),
                            weights=g[weight_col].astype(float),
                        ),
                        "denom_sum": float(g[weight_col].astype(float).sum()),
                        "n_domains": int(g["primary_tag"].nunique()),
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        grouped["layer"] = layer
        grouped["outcome"] = outcome
        series_rows.append(grouped)

    timing = pd.DataFrame(rows)
    timing.to_csv(TIMING_PANEL_CSV, index=False)
    series = pd.concat(series_rows, ignore_index=True)
    series.to_csv(TIMING_SERIES_CSV, index=False)
    return timing, series


def write_readout(few_cluster: pd.DataFrame, timing: pd.DataFrame) -> None:
    lines: list[str] = [
        "# P1 Promoted Methods Exhibits",
        "",
        "## Purpose",
        "",
        "These exhibits promote the paper's methods discipline into visible package surfaces rather than leaving it only in prose.",
        "",
        "## Few-Cluster Surface",
        "",
        "Promoted coefficients tracked here:",
        "",
    ]
    for _, row in few_cluster.iterrows():
        lines.append(
            f"- `{row['layer']}` layer / `{row['outcome']}`: coef `{row['coef']:.4f}`, p `{row['pval']:.4g}`, "
            f"`same-sign leave-one-domain-out share = {row['leave_one_out_same_sign_share']:.3f}` "
            f"across `{int(row['leave_one_out_runs'])}` runs."
        )
    lines.extend(
        [
            "",
            "## Timing Surface",
            "",
            "Each promoted outcome reports a pre-period differential slope (`high_tag:time_index`) and the promoted post-period level term (`high_post`).",
            "",
        ]
    )
    for _, row in timing.iterrows():
        lines.append(
            f"- `{row['layer']}` / `{row['outcome']}`: pretrend coef `{row['pretrend_coef']:.4f}` "
            f"(p `{row['pretrend_pval']:.4g}`); promoted post coef `{row['promoted_coef']:.4f}` "
            f"(p `{row['promoted_pval']:.4g}`)."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- [{FEW_CLUSTER_CSV.name}]({FEW_CLUSTER_CSV.as_posix()})",
            f"- [{TIMING_PANEL_CSV.name}]({TIMING_PANEL_CSV.as_posix()})",
            f"- [{TIMING_SERIES_CSV.name}]({TIMING_SERIES_CSV.as_posix()})",
        ]
    )
    READOUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel = pd.read_csv(PANEL_CSV)
    few_cluster = build_few_cluster_surface(panel)
    timing, _ = build_timing_panel(panel)
    write_readout(few_cluster, timing)
    print(FEW_CLUSTER_CSV)
    print(TIMING_PANEL_CSV)
    print(TIMING_SERIES_CSV)
    print(READOUT_MD)


if __name__ == "__main__":
    main()
