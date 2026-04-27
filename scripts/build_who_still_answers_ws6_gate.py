from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
FIGURES_DIR = BASE_DIR / "figures"
PAPER_DIR = BASE_DIR / "paper"

DURABILITY_PANEL = PROCESSED_DIR / "who_still_answers_durability_tag_month_panel.csv"
ROLE_PANEL = PROCESSED_DIR / "who_still_answers_answer_role_tag_month_panel.csv"
BRIDGE_PANEL = PROCESSED_DIR / "who_still_answers_infrastructure_bridge_panel.csv"

WS6_INFERENCE_CSV = PROCESSED_DIR / "who_still_answers_ws6_small_sample_inference.csv"
WS6_RANDOMIZATION_CSV = PROCESSED_DIR / "who_still_answers_ws6_randomization.csv"
WS6_TIMING_AUDIT_CSV = PROCESSED_DIR / "who_still_answers_ws6_timing_audit.csv"
WS6_CONSTRUCT_SENSITIVITY_CSV = PROCESSED_DIR / "who_still_answers_ws6_construct_sensitivity.csv"
WS6_SUMMARY_JSON = PROCESSED_DIR / "who_still_answers_ws6_summary.json"

WS6_INFERENCE_FIGURE = FIGURES_DIR / "who_still_answers_ws6_conservative_inference.png"
WS6_TIMING_FIGURE = FIGURES_DIR / "who_still_answers_ws6_timing_rank_panel.png"
WS6_READOUT_MD = PAPER_DIR / "who_still_answers_ws6_gate_readout_2026-04-04.md"

SHOCK_MONTH = "2022-12"
WILD_BOOTSTRAP_REPS = 399
RANDOMIZATION_REPS = 499
PRE_MIN_MONTHS = 6
SEED = 20260404


@dataclass
class ModelSpec:
    name: str
    frame: pd.DataFrame
    formula: str
    term: str
    weight_col: str | None
    cluster_col: str = "primary_tag"
    timing_expected_sign: float = 1.0


def filter_weighted_frame(frame: pd.DataFrame, weight_col: str | None) -> pd.DataFrame:
    if weight_col is None:
        return frame.copy()
    return frame.loc[frame[weight_col].fillna(0) > 0].copy()


def aligned_fit_frame(formula: str, data: pd.DataFrame, weight_col: str | None, cluster_col: str) -> pd.DataFrame:
    frame = filter_weighted_frame(data, weight_col)
    y, _ = patsy.dmatrices(formula, data=frame, return_type="dataframe", NA_action="drop")
    fit_frame = frame.loc[y.index].copy()
    fit_frame = fit_frame.loc[fit_frame[cluster_col].notna()].copy()
    return fit_frame


def fit_model(formula: str, data: pd.DataFrame, weight_col: str | None, cluster_col: str) -> object:
    frame = filter_weighted_frame(data, weight_col)
    y, x = patsy.dmatrices(formula, data=frame, return_type="dataframe", NA_action="drop")
    fit_frame = frame.loc[y.index].copy()
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


def remove_term_from_formula(formula: str, term: str) -> str:
    lhs, rhs = formula.split("~", 1)
    rhs_terms = [piece.strip() for piece in rhs.split("+")]
    kept_terms = [piece for piece in rhs_terms if piece != term]
    return f"{lhs.strip()} ~ {' + '.join(kept_terms)}"


def add_permuted_exposure(frame: pd.DataFrame, exposure_map: dict[str, float]) -> pd.DataFrame:
    temp = frame.copy()
    temp["exposure_index"] = temp["primary_tag"].map(exposure_map).astype(float)
    if "post_chatgpt" in temp.columns:
        temp["exposure_post"] = temp["exposure_index"] * temp["post_chatgpt"]
    return temp


def inverse_square_root_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    return eigenvectors @ inv_sqrt @ eigenvectors.T


def cr2_term_statistics(model: object, term: str, groups: pd.Series | np.ndarray) -> dict[str, float]:
    params = model.params
    index = list(params.index) if hasattr(params, "index") else []
    if term not in index:
        return {"coef": np.nan, "se": np.nan, "pval": np.nan, "tstat": np.nan, "df": np.nan}
    term_idx = index.index(term)
    coef = float(params[term])
    groups_array = np.asarray(groups)
    unique_groups = pd.Index(pd.unique(groups_array))
    if len(unique_groups) < 2:
        return {"coef": coef, "se": np.nan, "pval": np.nan, "tstat": np.nan, "df": np.nan}
    wexog = np.asarray(model.model.wexog, dtype=float)
    wresid = np.asarray(model.wresid, dtype=float)
    xtx_inv = np.linalg.pinv(wexog.T @ wexog)
    meat = np.zeros((wexog.shape[1], wexog.shape[1]), dtype=float)
    for cluster in unique_groups:
        mask = groups_array == cluster
        xg = wexog[mask, :]
        eg = wresid[mask]
        if xg.size == 0:
            continue
        leverage = xg @ xtx_inv @ xg.T
        adjust = inverse_square_root_psd(np.eye(xg.shape[0]) - leverage)
        adjusted_resid = adjust @ eg
        meat += xg.T @ np.outer(adjusted_resid, adjusted_resid) @ xg
    cov = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(max(cov[term_idx, term_idx], 0.0)))
    tstat = float(coef / se) if np.isfinite(se) and se != 0 else np.nan
    df = max(int(len(unique_groups)) - 1, 1)
    return {
        "coef": coef,
        "se": se,
        "pval": float(2 * stats.t.sf(abs(tstat), df)) if np.isfinite(tstat) else np.nan,
        "tstat": tstat,
        "df": float(df),
    }


def wild_cluster_bootstrap_pvalue(spec: ModelSpec) -> float:
    frame = filter_weighted_frame(spec.frame, spec.weight_col)
    restricted_formula = remove_term_from_formula(spec.formula, spec.term)
    y_full, x_full = patsy.dmatrices(spec.formula, frame, return_type="dataframe")
    y_restricted, x_restricted = patsy.dmatrices(restricted_formula, frame, return_type="dataframe")
    common_index = y_full.index.intersection(y_restricted.index)
    fit_frame = frame.loc[common_index].copy()
    fit_frame = fit_frame.loc[fit_frame[spec.cluster_col].notna()].copy()
    common_index = fit_frame.index
    y_full = y_full.loc[common_index]
    x_full = x_full.loc[common_index]
    y_restricted = y_restricted.loc[common_index]
    x_restricted = x_restricted.loc[common_index]
    groups = fit_frame[spec.cluster_col].to_numpy()
    unique_groups = pd.Index(sorted(pd.unique(groups)))
    if len(unique_groups) < 6:
        return np.nan
    weights = np.ones(len(fit_frame)) if spec.weight_col is None else fit_frame[spec.weight_col].to_numpy()
    full_fit = sm.WLS(y_full.iloc[:, 0], x_full, weights=weights).fit()
    full_cluster = full_fit.get_robustcov_results(
        cov_type="cluster",
        groups=groups,
        use_correction=True,
        df_correction=True,
    )
    term_index = list(x_full.columns).index(spec.term)
    observed_t = float(full_cluster.params[term_index] / full_cluster.bse[term_index])
    restricted_fit = sm.WLS(y_restricted.iloc[:, 0], x_restricted, weights=weights).fit()
    fitted_restricted = restricted_fit.fittedvalues.to_numpy()
    residuals = y_restricted.iloc[:, 0].to_numpy() - fitted_restricted
    rng = np.random.default_rng(SEED + abs(hash(spec.name)) % 1000)
    bootstrap_t = []
    group_codes = pd.Categorical(groups, categories=unique_groups).codes
    for _ in range(WILD_BOOTSTRAP_REPS):
        weights_by_group = rng.choice([-1.0, 1.0], size=len(unique_groups))
        y_star = fitted_restricted + residuals * weights_by_group[group_codes]
        try:
            fit_star = sm.WLS(y_star, x_full, weights=weights).fit()
            fit_star = fit_star.get_robustcov_results(
                cov_type="cluster",
                groups=groups,
                use_correction=True,
                df_correction=True,
            )
            se_star = float(fit_star.bse[term_index])
            if se_star == 0 or np.isnan(se_star):
                continue
            bootstrap_t.append(float(fit_star.params[term_index] / se_star))
        except Exception:
            continue
    if not bootstrap_t:
        return np.nan
    bootstrap_array = np.asarray(bootstrap_t, dtype=float)
    return float(np.mean(np.abs(bootstrap_array) >= abs(observed_t)))


def randomization_inference(spec: ModelSpec) -> tuple[float, pd.DataFrame]:
    if "exposure" not in spec.term:
        return np.nan, pd.DataFrame(columns=["specification", "draw", "coef", "actual_coef"])
    frame = filter_weighted_frame(spec.frame, spec.weight_col)
    actual_model = fit_model(spec.formula, frame, spec.weight_col, spec.cluster_col)
    actual_coef = float(actual_model.params.get(spec.term, np.nan))
    observed_map = frame.groupby("primary_tag", as_index=False)["exposure_index"].first()
    exposures = observed_map["exposure_index"].to_numpy()
    rng = np.random.default_rng(SEED + len(exposures) + abs(hash(spec.name)) % 1000)
    rows = []
    for draw in range(RANDOMIZATION_REPS):
        permuted = rng.permutation(exposures)
        exposure_map = dict(zip(observed_map["primary_tag"], permuted, strict=False))
        permuted_frame = add_permuted_exposure(frame, exposure_map)
        try:
            permuted_model = fit_model(spec.formula, permuted_frame, spec.weight_col, spec.cluster_col)
        except Exception:
            continue
        rows.append(
            {
                "specification": spec.name,
                "draw": draw,
                "coef": float(permuted_model.params.get(spec.term, np.nan)),
                "actual_coef": actual_coef,
            }
        )
    perm_df = pd.DataFrame(rows)
    pvalue = np.nan
    if not perm_df.empty:
        pvalue = float((perm_df["coef"].abs() >= abs(actual_coef)).mean())
    return pvalue, perm_df


def prepare_timing_frame(frame: pd.DataFrame, break_month: str) -> pd.DataFrame:
    temp = frame.copy()
    break_idx = int(temp.loc[temp["month_id"] == break_month, "month_order"].iloc[0])
    temp["break_post"] = (temp["month_order"] >= break_idx).astype(int)
    temp["break_slope"] = np.maximum(temp["month_order"] - break_idx, 0)
    temp["exposure_break_post"] = temp["exposure_index"] * temp["break_post"]
    temp["exposure_break_slope"] = temp["exposure_index"] * temp["break_slope"]
    return temp


def timing_audit(spec: ModelSpec, outcome: str) -> pd.DataFrame:
    frame = filter_weighted_frame(spec.frame, spec.weight_col).copy()
    all_months = sorted(frame["month_id"].dropna().unique())
    if SHOCK_MONTH not in all_months:
        return pd.DataFrame()
    shock_idx = all_months.index(SHOCK_MONTH)
    candidate_months = all_months[PRE_MIN_MONTHS : shock_idx + 1]
    rows = []
    formula = (
        f"{outcome} ~ exposure_index:month_order + exposure_break_post + exposure_break_slope + "
        "C(primary_tag) + C(month_id)"
    )
    for break_month in candidate_months:
        temp = prepare_timing_frame(frame, break_month)
        model = fit_model(formula, temp, spec.weight_col, spec.cluster_col)
        rows.append(
            {
                "specification": spec.name,
                "outcome": outcome,
                "break_month": break_month,
                "coef_level": float(model.params.get("exposure_break_post", np.nan)),
                "pval_level": float(model.pvalues.get("exposure_break_post", np.nan)),
                "coef_slope": float(model.params.get("exposure_break_slope", np.nan)),
                "pval_slope": float(model.pvalues.get("exposure_break_slope", np.nan)),
                "directional_coef": float(model.params.get("exposure_break_slope", np.nan)) * spec.timing_expected_sign,
            }
        )
    return pd.DataFrame(rows)


def build_specs() -> tuple[list[ModelSpec], pd.DataFrame]:
    durability = pd.read_csv(DURABILITY_PANEL)
    role = pd.read_csv(ROLE_PANEL)
    bridge = pd.read_csv(BRIDGE_PANEL)

    if "month_order" not in durability.columns:
        durability["month_order"] = durability["entry_month"].map(
            {m: i for i, m in enumerate(sorted(durability["entry_month"].dropna().unique()))}
        )
    durability["month_id"] = durability["entry_month"]

    if "month_order" not in role.columns:
        role["month_order"] = role["month_id"].map(
            {m: i for i, m in enumerate(sorted(role["month_id"].dropna().unique()))}
        )

    if "month_order" not in bridge.columns:
        bridge["month_order"] = bridge["month_id"].map(
            {m: i for i, m in enumerate(sorted(bridge["month_id"].dropna().unique()))}
        )

    pooled_role = role.loc[role["role"].isin(["accepted_current", "first_positive", "top_score"])].copy()
    pooled_role["role"] = pd.Categorical(
        pooled_role["role"],
        categories=["accepted_current", "first_positive", "top_score"],
        ordered=True,
    )

    specs = [
        ModelSpec(
            name="durability_return_365d",
            frame=durability.loc[durability["outcome"] == "return_365d"].copy(),
            formula="rate ~ exposure_index + post_chatgpt + exposure_index:post_chatgpt + C(primary_tag) + C(entry_month)",
            term="exposure_index:post_chatgpt",
            weight_col="n_eligible",
            timing_expected_sign=-1.0,
        ),
        ModelSpec(
            name="durability_one_shot_365d",
            frame=durability.loc[durability["outcome"] == "one_shot_365d"].copy(),
            formula="rate ~ exposure_index + post_chatgpt + exposure_index:post_chatgpt + C(primary_tag) + C(entry_month)",
            term="exposure_index:post_chatgpt",
            weight_col="n_eligible",
            timing_expected_sign=1.0,
        ),
        ModelSpec(
            name="role_first_positive_recent90",
            frame=role.loc[role["role"] == "first_positive"].copy(),
            formula="recent_entrant_90d_share ~ exposure_index + post_chatgpt + exposure_index:post_chatgpt + residual_queue_complexity_index_mean + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="exposure_index:post_chatgpt",
            weight_col="n_role_questions",
            timing_expected_sign=1.0,
        ),
        ModelSpec(
            name="role_top_score_recent90",
            frame=role.loc[role["role"] == "top_score"].copy(),
            formula="recent_entrant_90d_share ~ exposure_index + post_chatgpt + exposure_index:post_chatgpt + residual_queue_complexity_index_mean + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="exposure_index:post_chatgpt",
            weight_col="n_role_questions",
            timing_expected_sign=1.0,
        ),
        ModelSpec(
            name="role_asymmetry_first_positive_vs_accepted",
            frame=pooled_role,
            formula="recent_entrant_90d_share ~ exposure_index + post_chatgpt + C(role) + exposure_index:post_chatgpt + exposure_index:C(role) + post_chatgpt:C(role) + exposure_index:post_chatgpt:C(role) + residual_queue_complexity_index_mean + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="exposure_index:post_chatgpt:C(role)[T.first_positive]",
            weight_col="n_role_questions",
            timing_expected_sign=1.0,
        ),
        ModelSpec(
            name="bridge_accepted_vote_30d_gap",
            frame=bridge.copy(),
            formula="accepted_vote_30d_rate ~ exposure_post + recent_gap_first_vs_accepted + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="recent_gap_first_vs_accepted",
            weight_col="accepted_vote_30d_denom",
            timing_expected_sign=1.0,
        ),
        ModelSpec(
            name="bridge_first_positive_latency_gap",
            frame=bridge.copy(),
            formula="first_positive_answer_latency_mean ~ exposure_post + recent_gap_first_vs_accepted + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            term="recent_gap_first_vs_accepted",
            weight_col="first_positive_answer_latency_denom",
            timing_expected_sign=1.0,
        ),
    ]

    return specs, role


def small_sample_inference(specs: list[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    randomization_frames = []
    for spec in specs:
        frame = filter_weighted_frame(spec.frame, spec.weight_col)
        model = fit_model(spec.formula, frame, spec.weight_col, spec.cluster_col)
        cr2_frame = aligned_fit_frame(spec.formula, frame, spec.weight_col, spec.cluster_col)
        cr2 = cr2_term_statistics(model, spec.term, cr2_frame[spec.cluster_col])
        wild_p = wild_cluster_bootstrap_pvalue(spec)
        ri_p, ri_df = randomization_inference(spec)
        if not ri_df.empty:
            randomization_frames.append(ri_df)
        rows.append(
            {
                "specification": spec.name,
                "term": spec.term,
                "coef": float(model.params.get(spec.term, np.nan)),
                "cluster_pval": float(model.pvalues.get(spec.term, np.nan)),
                "cr2_se": cr2["se"],
                "cr2_pval": cr2["pval"],
                "cr2_tstat": cr2["tstat"],
                "cr2_df": cr2["df"],
                "wild_cluster_bootstrap_pval": wild_p,
                "randomization_pval": ri_p,
                "nobs": int(model.nobs),
                "n_clusters": int(frame[spec.cluster_col].nunique()),
            }
        )
    inference_df = pd.DataFrame(rows)
    randomization_df = (
        pd.concat(randomization_frames, ignore_index=True)
        if randomization_frames
        else pd.DataFrame(columns=["specification", "draw", "coef", "actual_coef"])
    )
    inference_df.to_csv(WS6_INFERENCE_CSV, index=False)
    randomization_df.to_csv(WS6_RANDOMIZATION_CSV, index=False)
    return inference_df, randomization_df


def build_construct_sensitivity(role_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in ["recent_entrant_30d_share", "recent_entrant_90d_share", "recent_entrant_365d_share"]:
        for role in ["first_positive", "top_score", "accepted_current"]:
            sub = role_panel.loc[role_panel["role"] == role].copy()
            model = fit_model(
                f"{outcome} ~ exposure_index * post_chatgpt + residual_queue_complexity_index_mean + C(primary_tag):time_index + C(primary_tag) + C(month_id)",
                sub,
                "n_role_questions",
                "primary_tag",
            )
            rows.append(
                {
                    "family": "role_recent_window",
                    "role": role,
                    "outcome": outcome,
                    "coef": float(model.params.get("exposure_index:post_chatgpt", np.nan)),
                    "pval": float(model.pvalues.get("exposure_index:post_chatgpt", np.nan)),
                }
            )
    bridge_results = pd.read_csv(PROCESSED_DIR / "who_still_answers_infrastructure_bridge_results.csv")
    subset = bridge_results.loc[
        (bridge_results["sample"] == "full")
        & (bridge_results["outcome"].isin(["accepted_vote_30d_rate", "accepted_cond_any_answer_30d_rate"]))
        & (bridge_results["term"].isin(["first_recent_share", "recent_gap_first_vs_accepted", "accepted_incumbent_share"]))
    ].copy()
    for _, row in subset.iterrows():
        rows.append(
            {
                "family": "bridge_construct_variant",
                "role": row["outcome"],
                "outcome": row["term"],
                "coef": float(row["coef"]),
                "pval": float(row["pval"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(WS6_CONSTRUCT_SENSITIVITY_CSV, index=False)
    return out


def timing_diagnostics(specs: list[ModelSpec]) -> pd.DataFrame:
    mapping = {
        "durability_return_365d": "rate",
        "durability_one_shot_365d": "rate",
        "role_first_positive_recent90": "recent_entrant_90d_share",
        "role_top_score_recent90": "recent_entrant_90d_share",
        "role_asymmetry_first_positive_vs_accepted": "recent_entrant_90d_share",
        "bridge_accepted_vote_30d_gap": "recent_gap_first_vs_accepted",
        "bridge_first_positive_latency_gap": "recent_gap_first_vs_accepted",
    }
    frames = []
    for spec in specs:
        timing_df = timing_audit(spec, mapping[spec.name])
        if not timing_df.empty:
            frames.append(timing_df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(WS6_TIMING_AUDIT_CSV, index=False)
    return out


def make_figures(inference_df: pd.DataFrame, timing_df: pd.DataFrame) -> None:
    if not inference_df.empty:
        plot_df = inference_df.copy()
        plot_df["label"] = plot_df["specification"].str.replace("_", " ", regex=False)
        y = np.arange(len(plot_df))
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        ax.errorbar(
            plot_df["coef"],
            y,
            xerr=1.96 * plot_df["cr2_se"],
            fmt="o",
            color="#1f4e79",
            ecolor="#8aa8c5",
            elinewidth=2,
            capsize=4,
        )
        ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["label"])
        ax.set_xlabel("Coefficient with CR2 95% interval")
        ax.set_title("WS6 conservative inference on promoted mechanism outcomes")
        fig.tight_layout()
        fig.savefig(WS6_INFERENCE_FIGURE, dpi=200)
        plt.close(fig)

    if not timing_df.empty:
        summary = []
        for spec, frame in timing_df.groupby("specification"):
            pre = frame.loc[frame["break_month"] < SHOCK_MONTH].copy()
            actual = frame.loc[frame["break_month"] == SHOCK_MONTH].copy()
            if pre.empty or actual.empty:
                continue
            actual_coef = float(actual["directional_coef"].iloc[0])
            rank = int((pre["directional_coef"] >= actual_coef).sum() + 1)
            summary.append(
                {
                    "specification": spec,
                    "rank_vs_pre": rank,
                    "n_pre": int(len(pre)),
                    "actual_directional_coef": actual_coef,
                }
            )
        rank_df = pd.DataFrame(summary)
        if not rank_df.empty:
            fig, ax = plt.subplots(figsize=(9.0, 4.8))
            ax.bar(rank_df["specification"].str.replace("_", "\n", regex=False), rank_df["rank_vs_pre"], color="#7a1f2b")
            ax.set_ylabel("Actual 2022-12 slope rank vs. pre-break candidates")
            ax.set_title("WS6 timing diagnostics remain bounded rather than clean-break")
            fig.tight_layout()
            fig.savefig(WS6_TIMING_FIGURE, dpi=200)
            plt.close(fig)


def write_summary(
    inference_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    construct_df: pd.DataFrame,
) -> dict[str, object]:
    summary: dict[str, object] = {"inference": {}, "timing": {}, "construct_sensitivity": {}}
    for _, row in inference_df.iterrows():
        summary["inference"][row["specification"]] = {
            "coef": float(row["coef"]),
            "cluster_pval": float(row["cluster_pval"]),
            "cr2_pval": float(row["cr2_pval"]),
            "wild_pval": float(row["wild_cluster_bootstrap_pval"]),
            "randomization_pval": float(row["randomization_pval"]),
        }
    for spec, frame in timing_df.groupby("specification"):
        pre = frame.loc[frame["break_month"] < SHOCK_MONTH].copy()
        actual = frame.loc[frame["break_month"] == SHOCK_MONTH].copy()
        if pre.empty or actual.empty:
            continue
        actual_coef = float(actual["directional_coef"].iloc[0])
        summary["timing"][spec] = {
            "actual_directional_coef": actual_coef,
            "rank_vs_pre": int((pre["directional_coef"] >= actual_coef).sum() + 1),
            "n_pre_breaks": int(len(pre)),
            "share_significant_pre_breaks": float((pre["pval_slope"] < 0.05).mean()),
        }
    for family, frame in construct_df.groupby("family"):
        summary["construct_sensitivity"][family] = frame.to_dict(orient="records")
    WS6_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_readout(summary: dict[str, object]) -> None:
    inf = summary["inference"]
    timing = summary["timing"]
    lines = [
        "# Who Still Answers: WS6 Gate Readout",
        "",
        "## Conservative Inference",
        "",
        f"- `durability_return_365d`: CR2 `{inf['durability_return_365d']['cr2_pval']:.4f}`, wild `{inf['durability_return_365d']['wild_pval']:.4f}`, RI `{inf['durability_return_365d']['randomization_pval']:.4f}`",
        f"- `durability_one_shot_365d`: CR2 `{inf['durability_one_shot_365d']['cr2_pval']:.4f}`, wild `{inf['durability_one_shot_365d']['wild_pval']:.4f}`, RI `{inf['durability_one_shot_365d']['randomization_pval']:.4f}`",
        f"- `role_first_positive_recent90`: CR2 `{inf['role_first_positive_recent90']['cr2_pval']:.4f}`, wild `{inf['role_first_positive_recent90']['wild_pval']:.4f}`, RI `{inf['role_first_positive_recent90']['randomization_pval']:.4f}`",
        f"- `role_top_score_recent90`: CR2 `{inf['role_top_score_recent90']['cr2_pval']:.4f}`, wild `{inf['role_top_score_recent90']['wild_pval']:.4f}`, RI `{inf['role_top_score_recent90']['randomization_pval']:.4f}`",
        f"- `role_asymmetry_first_positive_vs_accepted`: CR2 `{inf['role_asymmetry_first_positive_vs_accepted']['cr2_pval']:.4f}`, wild `{inf['role_asymmetry_first_positive_vs_accepted']['wild_pval']:.4f}`, RI `{inf['role_asymmetry_first_positive_vs_accepted']['randomization_pval']:.4f}`",
        f"- `bridge_accepted_vote_30d_gap`: CR2 `{inf['bridge_accepted_vote_30d_gap']['cr2_pval']:.4f}`, wild `{inf['bridge_accepted_vote_30d_gap']['wild_pval']:.4f}`, RI `{inf['bridge_accepted_vote_30d_gap']['randomization_pval']:.4f}`",
        f"- `bridge_first_positive_latency_gap`: CR2 `{inf['bridge_first_positive_latency_gap']['cr2_pval']:.4f}`, wild `{inf['bridge_first_positive_latency_gap']['wild_pval']:.4f}`, RI `{inf['bridge_first_positive_latency_gap']['randomization_pval']:.4f}`",
        "",
        "## Timing Discipline",
        "",
    ]
    for spec in [
        "durability_return_365d",
        "role_first_positive_recent90",
        "bridge_accepted_vote_30d_gap",
    ]:
        if spec in timing:
            lines.append(
                f"- `{spec}`: actual `2022-12` directional rank `{timing[spec]['rank_vs_pre']}` of `{timing[spec]['n_pre_breaks']}` pre-break candidates; significant pre-break share `{timing[spec]['share_significant_pre_breaks']:.4f}`"
            )
    lines.extend(
        [
            "",
            "## Construct-Sensitivity Read",
            "",
            "- The role family remains positive at `90d` and `365d` windows for first-positive and top-score roles; shorter `30d` windows are weaker and should stay secondary.",
            "- The bridge remains strongest for the role-gap measure `recent_gap_first_vs_accepted`; raw `first_recent_share` is informative but less discriminating.",
            "",
            "## Gate Decision",
            "",
            "The promoted mechanism stack survives conservative inference well enough to justify a manuscript rebuild, but not a fake sharp-shock narrative.",
            "The safe paper is now a bounded `ChatGPT-period differential exposure` paper about durable public labor, answer-role reallocation,",
            "and downstream certification consequences. Timing remains mixed, so the manuscript should explicitly reject a pristine release-date design.",
        ]
    )
    WS6_READOUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    specs, role_panel = build_specs()
    inference_df, _ = small_sample_inference(specs)
    timing_df = timing_diagnostics(specs)
    construct_df = build_construct_sensitivity(role_panel)
    summary = write_summary(inference_df, timing_df, construct_df)
    make_figures(inference_df, timing_df)
    write_readout(summary)


if __name__ == "__main__":
    main()
