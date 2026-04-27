from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper"

TAG_MONTH_PANEL_CSV = PROCESSED_DIR / "who_still_answers_tag_month_entry_panel.csv"

TIMING_AUDIT_CSV = PROCESSED_DIR / "who_still_answers_timing_acceleration_audit.csv"
TIMING_PERMUTATION_CSV = PROCESSED_DIR / "who_still_answers_timing_acceleration_permutations.csv"
STUDENTIZED_RI_CSV = PROCESSED_DIR / "who_still_answers_studentized_randomization.csv"
COLLAPSED_TAG_CSV = PROCESSED_DIR / "who_still_answers_collapsed_tag_inference.csv"
ROUND2_SUMMARY_JSON = PROCESSED_DIR / "who_still_answers_round2_extensions_summary.json"
ROUND2_MEMO_MD = PAPER_DIR / "who_still_answers_round2_extensions.md"

SHOCK_MONTH = "2022-12"
PRE_MIN_MONTHS = 6
TIMING_RANDOMIZATION_REPS = 999
INFERENCE_RANDOMIZATION_REPS = 3999
SEED = 20260404


def fit_weighted(formula: str, data: pd.DataFrame, weight_col: str | None, cluster_col: str = "primary_tag"):
    frame = data.copy()
    if weight_col is not None:
        frame = frame.loc[frame[weight_col].fillna(0) > 0].copy()
    y, x = patsy.dmatrices(formula, data=frame, return_type="dataframe", NA_action="drop")
    fit_frame = frame.loc[y.index].copy()
    if cluster_col in fit_frame.columns:
        fit_frame = fit_frame.loc[fit_frame[cluster_col].notna()].copy()
        y = y.loc[fit_frame.index]
        x = x.loc[fit_frame.index]
    groups = fit_frame[cluster_col]
    if weight_col is None:
        return sm.OLS(y, x).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True, "df_correction": True},
        )
    weights = fit_frame[weight_col].astype(float)
    return sm.WLS(y, x, weights=weights).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True, "df_correction": True},
    )


def prepare_panel() -> pd.DataFrame:
    panel = pd.read_csv(TAG_MONTH_PANEL_CSV)
    panel = panel.dropna(subset=["novice_entry_share", "exposure_index"]).copy()
    panel["post_chatgpt"] = panel["post_chatgpt"].astype(int)
    all_months = sorted(panel["month_id"].unique())
    month_map = {month: idx for idx, month in enumerate(all_months)}
    panel["month_order"] = panel["month_id"].map(month_map).astype(int)
    panel["exposure_time"] = panel["exposure_index"] * panel["month_order"]
    return panel


def prepare_break_frame(frame: pd.DataFrame, break_month: str) -> pd.DataFrame:
    temp = frame.copy()
    break_idx = int(temp.loc[temp["month_id"] == break_month, "month_order"].iloc[0])
    temp["break_post"] = (temp["month_order"] >= break_idx).astype(int)
    temp["break_slope"] = np.maximum(temp["month_order"] - break_idx, 0)
    temp["exposure_break_post"] = temp["exposure_index"] * temp["break_post"]
    temp["exposure_break_slope"] = temp["exposure_index"] * temp["break_slope"]
    return temp


def timing_acceleration_audit(
    panel: pd.DataFrame,
    run_permutations: bool = True,
    permutation_reps: int = TIMING_RANDOMIZATION_REPS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    all_months = sorted(panel["month_id"].unique())
    shock_idx = all_months.index(SHOCK_MONTH)
    candidate_months = all_months[PRE_MIN_MONTHS : shock_idx + 1]
    formula = (
        "novice_entry_share ~ exposure_time + exposure_break_post + "
        "exposure_break_slope + C(primary_tag) + C(month_id)"
    )
    rows = []
    for break_month in candidate_months:
        frame = prepare_break_frame(panel, break_month)
        model = fit_weighted(formula, frame, "n_new_answerers")
        rows.append(
            {
                "break_month": break_month,
                "coef_level": float(model.params.get("exposure_break_post", np.nan)),
                "pval_level": float(model.pvalues.get("exposure_break_post", np.nan)),
                "coef_acceleration": float(model.params.get("exposure_break_slope", np.nan)),
                "se_acceleration": float(model.bse.get("exposure_break_slope", np.nan)),
                "pval_acceleration": float(model.pvalues.get("exposure_break_slope", np.nan)),
            }
        )
    audit = pd.DataFrame(rows).sort_values("break_month").reset_index(drop=True)
    actual_row = audit.loc[audit["break_month"] == SHOCK_MONTH].iloc[0]
    pre = audit.loc[audit["break_month"] < SHOCK_MONTH].copy()
    actual_coef = float(actual_row["coef_acceleration"])
    rank = int((pre["coef_acceleration"] >= actual_coef).sum() + 1)
    audit["actual_break_month"] = SHOCK_MONTH
    audit["actual_coef_acceleration"] = actual_coef
    audit["actual_rank_vs_pre"] = rank
    audit.to_csv(TIMING_AUDIT_CSV, index=False)

    permutations = pd.DataFrame(
        columns=["draw", "coef_acceleration", "tstat_acceleration", "actual_coef_acceleration", "actual_tstat_acceleration"]
    )
    permutation_p = np.nan
    permutation_t_p = np.nan
    actual_frame = prepare_break_frame(panel, SHOCK_MONTH)
    actual_model = fit_weighted(formula, actual_frame, "n_new_answerers")
    actual_t = float(
        actual_model.params.get("exposure_break_slope", np.nan)
        / actual_model.bse.get("exposure_break_slope", np.nan)
    )
    if run_permutations:
        rng = np.random.default_rng(SEED)
        observed_map = panel.groupby("primary_tag", as_index=False)["exposure_index"].first()
        exposures = observed_map["exposure_index"].to_numpy()
        permutation_rows = []
        for draw in range(permutation_reps):
            permuted = rng.permutation(exposures)
            exposure_map = dict(zip(observed_map["primary_tag"], permuted, strict=False))
            perm_frame = actual_frame.copy()
            perm_frame["exposure_index"] = perm_frame["primary_tag"].map(exposure_map).astype(float)
            perm_frame["exposure_time"] = perm_frame["exposure_index"] * perm_frame["month_order"]
            perm_frame["exposure_break_post"] = perm_frame["exposure_index"] * perm_frame["break_post"]
            perm_frame["exposure_break_slope"] = perm_frame["exposure_index"] * perm_frame["break_slope"]
            try:
                perm_model = fit_weighted(formula, perm_frame, "n_new_answerers")
            except Exception:
                continue
            coef = float(perm_model.params.get("exposure_break_slope", np.nan))
            se = float(perm_model.bse.get("exposure_break_slope", np.nan))
            tstat = np.nan if se == 0 or np.isnan(se) else float(coef / se)
            permutation_rows.append(
                {
                    "draw": draw,
                    "coef_acceleration": coef,
                    "tstat_acceleration": tstat,
                    "actual_coef_acceleration": actual_coef,
                    "actual_tstat_acceleration": actual_t,
                }
            )
        permutations = pd.DataFrame(permutation_rows)
        if not permutations.empty:
            permutation_p = float((permutations["coef_acceleration"].abs() >= abs(actual_coef)).mean())
            permutation_t_p = float((permutations["tstat_acceleration"].abs() >= abs(actual_t)).mean())
    permutations.to_csv(TIMING_PERMUTATION_CSV, index=False)

    summary = {
        "actual_break_month": SHOCK_MONTH,
        "actual_coef_acceleration": actual_coef,
        "actual_pval_acceleration": float(actual_row["pval_acceleration"]),
        "actual_rank_vs_pre": rank,
        "n_pre_breaks": int(len(pre)),
        "share_pre_breaks_positive": float((pre["coef_acceleration"] > 0).mean()) if not pre.empty else np.nan,
        "share_pre_breaks_significant": float((pre["pval_acceleration"] < 0.05).mean()) if not pre.empty else np.nan,
        "permutation_p_coef": permutation_p,
        "permutation_p_tstat": permutation_t_p,
    }
    return audit, permutations, summary


def studentized_randomization(panel: pd.DataFrame, permutation_reps: int = INFERENCE_RANDOMIZATION_REPS) -> tuple[pd.DataFrame, dict]:
    formula = "novice_entry_share ~ exposure_post + C(primary_tag) + C(month_id)"
    actual_model = fit_weighted(formula, panel, "n_new_answerers")
    actual_coef = float(actual_model.params.get("exposure_post", np.nan))
    actual_se = float(actual_model.bse.get("exposure_post", np.nan))
    actual_t = np.nan if actual_se == 0 or np.isnan(actual_se) else float(actual_coef / actual_se)

    observed_map = panel.groupby("primary_tag", as_index=False)["exposure_index"].first()
    exposures = observed_map["exposure_index"].to_numpy()
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for draw in range(permutation_reps):
        permuted = rng.permutation(exposures)
        exposure_map = dict(zip(observed_map["primary_tag"], permuted, strict=False))
        temp = panel.copy()
        temp["exposure_index"] = temp["primary_tag"].map(exposure_map).astype(float)
        temp["exposure_post"] = temp["exposure_index"] * temp["post_chatgpt"]
        try:
            perm_model = fit_weighted(formula, temp, "n_new_answerers")
        except Exception:
            continue
        coef = float(perm_model.params.get("exposure_post", np.nan))
        se = float(perm_model.bse.get("exposure_post", np.nan))
        tstat = np.nan if se == 0 or np.isnan(se) else float(coef / se)
        rows.append(
            {
                "draw": draw,
                "coef": coef,
                "tstat": tstat,
                "actual_coef": actual_coef,
                "actual_tstat": actual_t,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(STUDENTIZED_RI_CSV, index=False)
    summary = {
        "actual_coef": actual_coef,
        "actual_tstat": actual_t,
        "coef_randomization_pval": float((out["coef"].abs() >= abs(actual_coef)).mean()) if not out.empty else np.nan,
        "tstat_randomization_pval": float((out["tstat"].abs() >= abs(actual_t)).mean()) if not out.empty else np.nan,
        "draws": int(len(out)),
    }
    return out, summary


def collapsed_tag_inference(panel: pd.DataFrame, permutation_reps: int = INFERENCE_RANDOMIZATION_REPS) -> tuple[pd.DataFrame, dict]:
    grouped = (
        panel.assign(period=np.where(panel["month_id"] < SHOCK_MONTH, "pre", "post"))
        .groupby(["primary_tag", "period"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "weighted_mean": np.average(g["novice_entry_share"], weights=g["n_new_answerers"]),
                    "total_weight": g["n_new_answerers"].sum(),
                    "exposure_index": g["exposure_index"].iloc[0],
                }
            )
        )
        .reset_index(drop=True)
    )
    wide = (
        grouped.pivot(index="primary_tag", columns="period", values=["weighted_mean", "total_weight", "exposure_index"])
        .reset_index()
    )
    wide.columns = ["_".join(col).strip("_") for col in wide.columns.to_flat_index()]
    wide["delta_novice_entry_share"] = wide["weighted_mean_post"] - wide["weighted_mean_pre"]
    wide["combined_weight"] = wide["total_weight_post"].fillna(0) + wide["total_weight_pre"].fillna(0)
    wide["exposure_index"] = wide["exposure_index_post"].fillna(wide["exposure_index_pre"])
    wide = wide.dropna(subset=["delta_novice_entry_share", "exposure_index"]).copy()

    y, x = patsy.dmatrices("delta_novice_entry_share ~ exposure_index", data=wide, return_type="dataframe")
    weights = wide.loc[y.index, "combined_weight"].astype(float)
    model = sm.WLS(y.iloc[:, 0], x, weights=weights).fit()
    actual_coef = float(model.params.get("exposure_index", np.nan))
    actual_t = float(model.tvalues.get("exposure_index", np.nan))

    exposures = wide["exposure_index"].to_numpy()
    rng = np.random.default_rng(SEED + 2)
    permutation_rows = []
    for draw in range(permutation_reps):
        permuted = rng.permutation(exposures)
        temp = wide.copy()
        temp["perm_exposure_index"] = permuted
        y_perm, x_perm = patsy.dmatrices(
            "delta_novice_entry_share ~ perm_exposure_index", data=temp, return_type="dataframe"
        )
        perm_model = sm.WLS(y_perm.iloc[:, 0], x_perm, weights=temp.loc[y_perm.index, "combined_weight"].astype(float)).fit()
        permutation_rows.append(
            {
                "draw": draw,
                "coef": float(perm_model.params.get("perm_exposure_index", np.nan)),
                "tstat": float(perm_model.tvalues.get("perm_exposure_index", np.nan)),
                "actual_coef": actual_coef,
                "actual_tstat": actual_t,
            }
        )
    permutation_df = pd.DataFrame(permutation_rows)
    coef_p = float((permutation_df["coef"].abs() >= abs(actual_coef)).mean())
    t_p = float((permutation_df["tstat"].abs() >= abs(actual_t)).mean())

    tag_col = "primary_tag" if "primary_tag" in wide.columns else "primary_tag_"
    export = wide[[tag_col, "exposure_index", "delta_novice_entry_share", "combined_weight"]].rename(
        columns={tag_col: "primary_tag"}
    )
    export["actual_coef"] = actual_coef
    export["actual_tstat"] = actual_t
    export["coef_randomization_pval"] = coef_p
    export["tstat_randomization_pval"] = t_p
    export.to_csv(COLLAPSED_TAG_CSV, index=False)
    summary = {
        "actual_coef": actual_coef,
        "actual_tstat": actual_t,
        "coef_randomization_pval": coef_p,
        "tstat_randomization_pval": t_p,
        "n_tags": int(len(wide)),
    }
    return export, summary


def write_memo(timing_summary: dict, ri_summary: dict, collapsed_summary: dict) -> None:
    lines = [
        "# Round 2 Empirical Extensions",
        "",
        "Date: April 4, 2026",
        "",
    ]
    if timing_summary:
        lines.extend(
            [
                "## Timing Acceleration Audit",
                "",
                f"- actual break month: `{timing_summary['actual_break_month']}`",
                f"- acceleration coefficient: `{timing_summary['actual_coef_acceleration']:.6f}`",
                f"- cluster p-value on acceleration term: `{timing_summary['actual_pval_acceleration']:.4f}`",
                f"- rank versus pre-break candidates: `{timing_summary['actual_rank_vs_pre']} / {timing_summary['n_pre_breaks']}`",
                f"- share of pre-break acceleration terms positive: `{timing_summary['share_pre_breaks_positive']:.3f}`",
                f"- share of pre-break acceleration terms significant: `{timing_summary['share_pre_breaks_significant']:.3f}`",
                f"- exposure-permutation p-value on coefficient: `{timing_summary['permutation_p_coef']:.4f}`",
                f"- exposure-permutation p-value on t-stat: `{timing_summary['permutation_p_tstat']:.4f}`",
                "",
            ]
        )
    if ri_summary:
        lines.extend(
            [
                "## Studentized Randomization Inference",
                "",
                f"- draws: `{ri_summary['draws']}`",
                f"- actual coefficient: `{ri_summary['actual_coef']:.6f}`",
                f"- actual t-stat: `{ri_summary['actual_tstat']:.4f}`",
                f"- coefficient-based randomization p-value: `{ri_summary['coef_randomization_pval']:.4f}`",
                f"- t-stat-based randomization p-value: `{ri_summary['tstat_randomization_pval']:.4f}`",
                "",
            ]
        )
    if collapsed_summary:
        lines.extend(
            [
                "## Collapsed 16-Tag Check",
                "",
                f"- tags: `{collapsed_summary['n_tags']}`",
                f"- collapsed delta coefficient: `{collapsed_summary['actual_coef']:.6f}`",
                f"- collapsed delta t-stat: `{collapsed_summary['actual_tstat']:.4f}`",
                f"- collapsed coefficient permutation p-value: `{collapsed_summary['coef_randomization_pval']:.4f}`",
                f"- collapsed t-stat permutation p-value: `{collapsed_summary['tstat_randomization_pval']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Read",
            "",
            "- This file is an empirical extension note, not a claim rewrite.",
            "- If the timing acceleration audit still ranks poorly against pre-breaks, the honest timing sentence should remain `visible by the ChatGPT period`.",
            "- If studentized RI and the collapsed-tag check stay mixed, the entrant-side headline should still be treated as bounded under the full conservative stack.",
        ]
    )
    ROUND2_MEMO_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "timing", "inference"], default="all")
    parser.add_argument("--timing-reps", type=int, default=TIMING_RANDOMIZATION_REPS)
    parser.add_argument("--inference-reps", type=int, default=INFERENCE_RANDOMIZATION_REPS)
    parser.add_argument("--skip-timing-permutations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = prepare_panel()
    timing_summary = {}
    ri_summary = {}
    collapsed_summary = {}
    if args.mode in {"all", "timing"}:
        _, _, timing_summary = timing_acceleration_audit(
            panel,
            run_permutations=not args.skip_timing_permutations,
            permutation_reps=args.timing_reps,
        )
    if args.mode in {"all", "inference"}:
        _, ri_summary = studentized_randomization(panel, permutation_reps=args.inference_reps)
        _, collapsed_summary = collapsed_tag_inference(panel, permutation_reps=args.inference_reps)
    summary = {
        "timing_acceleration": timing_summary,
        "studentized_randomization": ri_summary,
        "collapsed_tag_check": collapsed_summary,
    }
    ROUND2_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_memo(timing_summary, ri_summary, collapsed_summary)
    print(ROUND2_SUMMARY_JSON)
    print(ROUND2_MEMO_MD)


if __name__ == "__main__":
    main()
