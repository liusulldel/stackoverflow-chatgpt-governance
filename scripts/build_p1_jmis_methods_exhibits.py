"""
Build promoted methods exhibits for the P1 JMIS package.

Outputs are intentionally "submission-surface" oriented:
- few-cluster LODO sign-stability (domain = primary_tag, k=16)
- timing discipline (event-time means, placebo break ranks)

This script is designed to be conservative and auditable:
- it uses the same WLS + FE + tag-specific trend structure used in the existing
  harmonized results (where feasible)
- it clusters SE by primary_tag (few clusters; we report primarily sign stability)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


ROOT = str(Path(__file__).resolve().parent.parent)
PROCESSED = os.path.join(ROOT, "processed")
PAPER = os.path.join(ROOT, "paper")
FIGURES = os.path.join(ROOT, "figures")


def _ensure_dirs() -> None:
    for d in (PROCESSED, PAPER, FIGURES):
        os.makedirs(d, exist_ok=True)


@dataclass(frozen=True)
class ModelSpec:
    spec_id: str
    panel: str
    outcome: str
    term: str
    formula: str
    weight_col: str


def _read_panel(name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED, name)
    df = pd.read_csv(path)
    # Ensure consistent dtypes for FE terms
    df["primary_tag"] = df["primary_tag"].astype(str)
    df["month_id"] = df["month_id"].astype(str)
    if "time_index" in df.columns:
        df["time_index"] = pd.to_numeric(df["time_index"], errors="coerce")
    if "high_tag" in df.columns:
        df["high_tag"] = pd.to_numeric(df["high_tag"], errors="coerce")
    if "high_post" in df.columns:
        df["high_post"] = pd.to_numeric(df["high_post"], errors="coerce")
    return df


def _fit_wls_cluster_term(df: pd.DataFrame, formula: str, weight_col: str, term: str) -> tuple[float, float, float, int]:
    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    dfx = df.copy()
    dfx["_w"] = w
    dfx = dfx.loc[dfx["_w"] > 0].copy()
    model = smf.wls(formula=formula, data=dfx, weights=dfx["_w"])
    # Fit first, then align cluster groups to the rows actually used by Patsy (it may drop NA rows).
    base = model.fit()
    nobs = int(base.nobs)
    coef = float(base.params.get(term, np.nan))

    # Cluster-robust SEs with few clusters are not a panacea; we compute them for completeness,
    # but the LODO exhibit is interpreted primarily via sign stability.
    se = np.nan
    pval = np.nan
    try:
        used_idx = base.model.data.row_labels
        groups = dfx.loc[used_idx, "primary_tag"]
        robust = base.get_robustcov_results(cov_type="cluster", groups=groups)
        se = float(robust.bse[robust.model.exog_names.index(term)]) if term in robust.model.exog_names else np.nan
        pval = float(robust.pvalues[robust.model.exog_names.index(term)]) if term in robust.model.exog_names else np.nan
    except Exception:
        # Keep coef + nobs; SE/pval are optional for the stability artifacts.
        pass
    return coef, se, pval, nobs


def build_lodo(specs: list[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    summary_rows: list[dict] = []

    panels: dict[str, pd.DataFrame] = {}
    for sp in specs:
        if sp.panel not in panels:
            panels[sp.panel] = _read_panel(sp.panel)

    for sp in specs:
        df = panels[sp.panel]
        full_coef, full_se, full_pval, full_nobs = _fit_wls_cluster_term(df, sp.formula, sp.weight_col, sp.term)

        coefs = []
        omitted = []
        for tag in sorted(df["primary_tag"].unique()):
            dfo = df.loc[df["primary_tag"] != tag].copy()
            coef, se, pval, nobs = _fit_wls_cluster_term(dfo, sp.formula, sp.weight_col, sp.term)
            rows.append(
                {
                    "spec_id": sp.spec_id,
                    "panel": sp.panel,
                    "outcome": sp.outcome,
                    "term": sp.term,
                    "omitted_primary_tag": tag,
                    "coef": coef,
                    "se": se,
                    "pval": pval,
                    "nobs": nobs,
                    "weight_col": sp.weight_col,
                    "formula": sp.formula,
                }
            )
            if not math.isnan(coef):
                coefs.append(coef)
                omitted.append(tag)

        same_sign = 0
        flips = []
        if not math.isnan(full_coef) and full_coef != 0 and coefs:
            full_sign = 1 if full_coef > 0 else -1
            for coef, tag in zip(coefs, omitted):
                if coef == 0 or math.isnan(coef):
                    continue
                s = 1 if coef > 0 else -1
                if s == full_sign:
                    same_sign += 1
                else:
                    flips.append(tag)
        stability_share = same_sign / max(1, len(coefs))

        summary_rows.append(
            {
                "spec_id": sp.spec_id,
                "panel": sp.panel,
                "outcome": sp.outcome,
                "term": sp.term,
                "full_coef": full_coef,
                "full_se": full_se,
                "full_pval": full_pval,
                "full_nobs": full_nobs,
                "sign_stability_share": stability_share,
                "min_coef_over_omissions": float(np.min(coefs)) if coefs else np.nan,
                "max_coef_over_omissions": float(np.max(coefs)) if coefs else np.nan,
                "any_sign_flip": bool(flips),
                "flip_tags": ";".join(flips) if flips else "",
                "weight_col": sp.weight_col,
                "formula": sp.formula,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def build_event_time_means(
    panel_name: str,
    cutoff_month: str,
    outcomes: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    outcomes: list of (outcome_col, weight_col)
    """
    df = _read_panel(panel_name)
    # month_id is YYYY-MM; convert to month index
    months = sorted(df["month_id"].unique())
    month_to_idx = {m: i for i, m in enumerate(months)}
    cutoff_idx = month_to_idx[cutoff_month]
    df["_midx"] = df["month_id"].map(month_to_idx)
    df["event_month"] = df["_midx"] - cutoff_idx

    out_rows: list[dict] = []
    for outcome, wcol in outcomes:
        dfx = df.copy()
        dfx["_w"] = pd.to_numeric(dfx[wcol], errors="coerce").fillna(0.0)
        dfx = dfx.loc[dfx["_w"] > 0].copy()
        for (ev, high), g in dfx.groupby(["event_month", "high_tag"], dropna=False):
            w = g["_w"].to_numpy()
            y = pd.to_numeric(g[outcome], errors="coerce").to_numpy()
            m = float(np.average(y, weights=w)) if len(y) else np.nan
            out_rows.append(
                {
                    "panel": panel_name,
                    "outcome": outcome,
                    "weight_col": wcol,
                    "event_month": int(ev),
                    "high_tag": int(high) if not pd.isna(high) else np.nan,
                    "value": m,
                    "weight_sum": float(np.sum(w)),
                    "n_rows": int(len(y)),
                }
            )
    return pd.DataFrame(out_rows).sort_values(["outcome", "event_month", "high_tag"])


def build_placebo_break_rank(
    panel_name: str,
    placebo_months: list[str],
    actual_cutoff: str,
    outcome_specs: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """
    outcome_specs: list of (outcome_col, weight_col, formula_rhs_template)
    RHS template should include a placeholder '{hp}' for the high_post term name.
    """
    df = _read_panel(panel_name)
    months = sorted(df["month_id"].unique())
    month_to_idx = {m: i for i, m in enumerate(months)}

    results = []

    for outcome, wcol, rhs_tmpl in outcome_specs:
        # actual
        df_actual = df.copy()
        df_actual["_post"] = (df_actual["month_id"].map(month_to_idx) >= month_to_idx[actual_cutoff]).astype(int)
        df_actual["_hp"] = df_actual["high_tag"].astype(int) * df_actual["_post"]
        formula = f"{outcome} ~ _hp + {rhs_tmpl.format(hp='_hp')}"
        actual_coef, actual_se, actual_pval, _ = _fit_wls_cluster_term(df_actual, formula, wcol, "_hp")

        placebo_rows = []
        for pm in placebo_months:
            dfp = df.copy()
            dfp["_post"] = (dfp["month_id"].map(month_to_idx) >= month_to_idx[pm]).astype(int)
            dfp["_hp"] = dfp["high_tag"].astype(int) * dfp["_post"]
            pcoef, pse, ppval, _ = _fit_wls_cluster_term(dfp, formula, wcol, "_hp")
            placebo_rows.append((pm, pcoef, pse, ppval))

        # rank by absolute magnitude among placebo coefs
        placebo_abs = [abs(c) for _, c, _, _ in placebo_rows if not math.isnan(c)]
        actual_abs = abs(actual_coef) if not math.isnan(actual_coef) else np.nan
        rank = 1 + sum(1 for a in placebo_abs if a >= actual_abs) if placebo_abs and not math.isnan(actual_abs) else np.nan

        results.append(
            {
                "panel": panel_name,
                "outcome": outcome,
                "actual_cutoff_month": actual_cutoff,
                "actual_coef": actual_coef,
                "actual_se": actual_se,
                "actual_pval": actual_pval,
                "rank_by_abs_vs_placebos": rank,
                "n_placebos": int(len(placebo_abs)),
                "placebo_months_min": min(placebo_months) if placebo_months else "",
                "placebo_months_max": max(placebo_months) if placebo_months else "",
                "weight_col": wcol,
                "formula": formula,
            }
        )

        for pm, pcoef, pse, ppval in placebo_rows:
            results.append(
                {
                    "panel": panel_name,
                    "outcome": outcome,
                    "actual_cutoff_month": actual_cutoff,
                    "actual_coef": actual_coef,
                    "actual_se": actual_se,
                    "actual_pval": actual_pval,
                    "rank_by_abs_vs_placebos": rank,
                    "n_placebos": int(len(placebo_abs)),
                    "placebo_month": pm,
                    "placebo_coef": pcoef,
                    "placebo_se": pse,
                    "placebo_pval": ppval,
                    "weight_col": wcol,
                    "formula": formula,
                }
            )

    return pd.DataFrame(results)


def write_md_exhibit_few_cluster(summary: pd.DataFrame, out_path: str) -> None:
    cols = [
        "outcome",
        "term",
        "full_coef",
        "full_pval",
        "sign_stability_share",
        "min_coef_over_omissions",
        "max_coef_over_omissions",
        "any_sign_flip",
        "flip_tags",
    ]
    d = summary[cols].copy()
    d["sign_stability_share"] = d["sign_stability_share"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    for c in ("full_coef", "min_coef_over_omissions", "max_coef_over_omissions"):
        d[c] = d[c].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
    d["full_pval"] = d["full_pval"].map(lambda x: f"{x:.3g}" if pd.notna(x) else "")
    d["any_sign_flip"] = d["any_sign_flip"].map(lambda x: "yes" if x else "no")

    lines = []
    lines.append("# Methods Exhibit: Few-Cluster LODO Sign Stability")
    lines.append("")
    lines.append("Unit of analysis: `domain-month` (`primary_tag 脳 month_id`). Effective clusters: `16` focal domains.")
    lines.append("")
    lines.append("This exhibit reports leave-one-domain-out (LODO) sign stability for the headline coefficients used in Tables 2鈥?.")
    lines.append("")
    lines.append(d.to_markdown(index=False))
    lines.append("")
    lines.append("Interpretation: this is a conservative sensitivity check (not CR2). If a coefficient flips sign when a single domain is omitted, the manuscript should narrow any strong wording attached to that coefficient.")
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_md_exhibit_timing(placebo_summary: pd.DataFrame, out_path: str) -> None:
    # Keep a compact summary: one row per outcome (actual only)
    df = placebo_summary.loc[placebo_summary["placebo_month"].isna()].copy() if "placebo_month" in placebo_summary.columns else placebo_summary.copy()
    keep = [
        "outcome",
        "actual_cutoff_month",
        "actual_coef",
        "actual_pval",
        "rank_by_abs_vs_placebos",
        "n_placebos",
        "placebo_months_min",
        "placebo_months_max",
    ]
    d = df[keep].copy()
    d["actual_coef"] = d["actual_coef"].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
    d["actual_pval"] = d["actual_pval"].map(lambda x: f"{x:.3g}" if pd.notna(x) else "")
    d["rank_by_abs_vs_placebos"] = d["rank_by_abs_vs_placebos"].map(lambda x: f"{int(x)}" if pd.notna(x) else "")

    lines = []
    lines.append("# Methods Exhibit: Timing Discipline (Placebo Break Ranks)")
    lines.append("")
    lines.append("This paper does not claim a clean shock. This exhibit exists to show that the periodization is not pure label: we compare the `2022-12` cutoff coefficient to a distribution of pre-period placebo cutoffs.")
    lines.append("")
    lines.append(d.to_markdown(index=False))
    lines.append("")
    lines.append("Note: `rank_by_abs_vs_placebos` counts how many placebo cutoffs produce an equal-or-larger absolute coefficient. A low rank indicates the cutoff is unusually large relative to pre-period breaks; a high rank indicates it is not exceptional.")
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    _ensure_dirs()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Headline LODO specs (aligned to the methods-surface memo)
    specs = [
        # Table 2 layer
        ModelSpec(
            spec_id="queue_residual_complexity",
            panel="p1_jmis_residual_queue_panel.csv",
            outcome="residual_queue_complexity_index_mean",
            term="high_post",
            formula="residual_queue_complexity_index_mean ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="n_questions",
        ),
        ModelSpec(
            spec_id="queue_body_word_count",
            panel="p1_jmis_residual_queue_panel.csv",
            outcome="body_word_count_mean",
            term="high_post",
            formula="body_word_count_mean ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="n_questions",
        ),
        ModelSpec(
            spec_id="queue_tag_breadth",
            panel="p1_jmis_residual_queue_panel.csv",
            outcome="tag_count_full_mean",
            term="high_post",
            formula="tag_count_full_mean ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="n_questions",
        ),
        # Table 3 layer
        ModelSpec(
            spec_id="resort_brand_new_share",
            panel="p1_jmis_residual_queue_panel.csv",
            outcome="brand_new_platform_share",
            term="high_post",
            formula="brand_new_platform_share ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="n_new_answerers_profiles",
        ),
        ModelSpec(
            spec_id="resort_low_tenure_share",
            panel="p1_jmis_residual_queue_panel.csv",
            outcome="low_tenure_existing_share",
            term="high_post",
            formula="low_tenure_existing_share ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="n_new_answerers_profiles",
        ),
        ModelSpec(
            spec_id="resort_established_cross_tag_share",
            panel="p1_jmis_residual_queue_panel.csv",
            outcome="established_cross_tag_share",
            term="high_post",
            formula="established_cross_tag_share ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="n_new_answerers_profiles",
        ),
        # Table 4 layer (exposure-by-period)
        ModelSpec(
            spec_id="consequence_first_answer_1d_closure",
            panel="p1_jmis_residual_queue_panel.csv",
            outcome="first_answer_1d_rate_closure",
            term="high_post",
            formula="first_answer_1d_rate_closure ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="first_answer_1d_denom_closure",
        ),
        ModelSpec(
            spec_id="consequence_accepted_vote_30d_rate",
            panel="p1_jmis_residual_queue_panel.csv",
            outcome="accepted_vote_30d_rate",
            term="high_post",
            formula="accepted_vote_30d_rate ~ high_post + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="accepted_vote_30d_denom",
        ),
        # Table 4 layer (entrant-share surfaces)
        ModelSpec(
            spec_id="assoc_brand_new_to_any_answer_7d",
            panel="p1_jmis_subtype_consequence_panel.csv",
            outcome="any_answer_7d_rate",
            term="brand_new_platform_share",
            formula="any_answer_7d_rate ~ high_post + brand_new_platform_share + low_tenure_existing_share + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="any_answer_7d_denom",
        ),
        ModelSpec(
            spec_id="assoc_established_to_any_answer_7d",
            panel="p1_jmis_subtype_consequence_panel.csv",
            outcome="any_answer_7d_rate",
            term="established_cross_tag_share",
            formula="any_answer_7d_rate ~ high_post + established_cross_tag_share + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="any_answer_7d_denom",
        ),
        ModelSpec(
            spec_id="assoc_established_to_first_positive_latency",
            panel="p1_jmis_subtype_consequence_panel.csv",
            outcome="first_positive_answer_latency_mean",
            term="established_cross_tag_share",
            formula="first_positive_answer_latency_mean ~ high_post + established_cross_tag_share + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            weight_col="first_positive_answer_latency_denom",
        ),
    ]

    lodo_est, lodo_sum = build_lodo(specs)
    lodo_est_path = os.path.join(PROCESSED, "p1_jmis_lodo_headline_estimates.csv")
    lodo_sum_path = os.path.join(PROCESSED, "p1_jmis_lodo_headline_summary.csv")
    lodo_est.to_csv(lodo_est_path, index=False)
    lodo_sum.to_csv(lodo_sum_path, index=False)

    few_cluster_md = os.path.join(PAPER, "p1_jmis_methods_exhibit_few_cluster_2026-04-04.md")
    write_md_exhibit_few_cluster(lodo_sum, few_cluster_md)

    # Timing: event-time means from the harmonized tag-month panel
    event_outcomes = [
        ("residual_queue_complexity_index_mean", "n_questions"),
        ("brand_new_platform_share", "n_new_answerers_profiles"),
        ("first_answer_1d_rate_closure", "first_answer_1d_denom_closure"),
        ("accepted_vote_30d_rate", "accepted_vote_30d_denom"),
    ]
    ev = build_event_time_means(
        panel_name="p1_jmis_residual_queue_panel.csv",
        cutoff_month="2022-12",
        outcomes=event_outcomes,
    )
    ev_path = os.path.join(PROCESSED, "p1_jmis_event_time_means.csv")
    ev.to_csv(ev_path, index=False)

    # Plot event-time panels (high vs low)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for i, (outcome, _wcol) in enumerate(event_outcomes):
        ax = axes[i]
        d = ev.loc[ev["outcome"] == outcome].copy()
        for high_val, label in [(0, "low-exposure"), (1, "high-exposure")]:
            g = d.loc[d["high_tag"] == high_val].sort_values("event_month")
            ax.plot(g["event_month"], g["value"], marker="o", linewidth=1.5, label=label)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(outcome)
        ax.set_xlabel("event_month (0 = 2022-12)")
        ax.set_ylabel("weighted mean")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(frameon=False, loc="best")
    fig.suptitle("Timing Discipline: Event-Time Means (High vs Low Exposure)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path_ev = os.path.join(FIGURES, "p1_jmis_event_time_panels.png")
    fig.savefig(fig_path_ev, dpi=200)
    plt.close(fig)

    # Timing: placebo break ranks
    # Pre-period placebo cutoffs
    df0 = _read_panel("p1_jmis_residual_queue_panel.csv")
    months = sorted(df0["month_id"].unique())
    placebo_months = [m for m in months if "2020-06" <= m <= "2022-06"]
    rhs = "C(primary_tag):time_index + C(primary_tag) + C(month_id)"
    placebo_specs = [
        ("residual_queue_complexity_index_mean", "n_questions", rhs),
        ("brand_new_platform_share", "n_new_answerers_profiles", rhs),
        ("first_answer_1d_rate_closure", "first_answer_1d_denom_closure", rhs),
        ("accepted_vote_30d_rate", "accepted_vote_30d_denom", rhs),
    ]
    placebo = build_placebo_break_rank(
        panel_name="p1_jmis_residual_queue_panel.csv",
        placebo_months=placebo_months,
        actual_cutoff="2022-12",
        outcome_specs=placebo_specs,
    )
    placebo_path = os.path.join(PROCESSED, "p1_jmis_placebo_break_rank.csv")
    placebo.to_csv(placebo_path, index=False)

    timing_md = os.path.join(PAPER, "p1_jmis_methods_exhibit_timing_2026-04-04.md")
    write_md_exhibit_timing(placebo, timing_md)

    # Plot placebo coefficient distributions
    # The saved file includes both an "actual-only" row and placebo rows; filter placebo rows.
    dfp = placebo.copy()
    if "placebo_month" not in dfp.columns:
        dfp["placebo_month"] = np.nan
    outcomes_u = [o for o, *_ in placebo_specs]
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    axes2 = axes2.flatten()
    for i, outcome in enumerate(outcomes_u):
        ax = axes2[i]
        sub = dfp.loc[(dfp["outcome"] == outcome) & (dfp["placebo_month"].notna())].copy()
        actual_row = dfp.loc[(dfp["outcome"] == outcome) & (dfp["placebo_month"].isna())].head(1)
        actual_coef = float(actual_row["actual_coef"].iloc[0]) if len(actual_row) else np.nan
        coefs = pd.to_numeric(sub["placebo_coef"], errors="coerce").dropna().to_numpy()
        ax.hist(coefs, bins=12, color="#cccccc", edgecolor="#666666")
        ax.axvline(actual_coef, color="#c0392b", linewidth=2, label="actual cutoff (2022-12)")
        ax.set_title(outcome)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(frameon=False, loc="best")
    fig2.suptitle("Timing Discipline: Placebo Cutoff Coefficient Distributions")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path_pb = os.path.join(FIGURES, "p1_jmis_placebo_rank_panels.png")
    fig2.savefig(fig_path_pb, dpi=200)
    plt.close(fig2)

    # Emit a small run log
    log_path = os.path.join(PAPER, "p1_jmis_methods_exhibits_runlog_2026-04-04.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Ran at: {stamp}\n")
        f.write("Outputs:\n")
        for p in [
            lodo_est_path,
            lodo_sum_path,
            few_cluster_md,
            ev_path,
            fig_path_ev,
            placebo_path,
            timing_md,
            fig_path_pb,
            log_path,
        ]:
            f.write(f"- {p}\n")


if __name__ == "__main__":
    main()
