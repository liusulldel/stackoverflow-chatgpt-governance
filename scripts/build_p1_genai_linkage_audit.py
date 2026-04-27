from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"

# Inputs produced by earlier build steps (we reuse rather than recompute).
INTERNAL_TAG_MONTH = PROCESSED / "who_still_answers_internal_ai_title_trace_tag_month.csv"
EXTERNAL_PAGEVIEWS = PROCESSED / "who_still_answers_external_ai_pageviews.csv"

# Outputs for this linkage-audit build.
OUT_SERIES = PROCESSED / "p1_genai_linkage_audit_monthly_series.csv"
OUT_R1_R4 = PROCESSED / "p1_genai_linkage_audit_R1_R4_results.csv"
OUT_SUMMARY = PROCESSED / "p1_genai_linkage_audit_summary.json"

FIG_LA1 = FIGURES / "p1_genai_linkage_audit_LA1_trace.png"
READOUT = PAPER / "p1_genai_linkage_audit_build_readout_2026-04-04.md"


EVENT_CHATGPT_RELEASE = pd.Timestamp("2022-11-30")
EVENT_CHATGPT_LAUNCH_WEEK = pd.Timestamp("2022-12-05")
POST_START_MONTH = "2022-12"

SEED = 42
N_PERMUTATIONS = 2000


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def parse_month_dt(month_id: pd.Series) -> pd.Series:
    # month_id is "YYYY-MM"
    return pd.to_datetime(month_id.astype(str) + "-01", errors="coerce")


def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return np.nan, np.nan, int(len(df))
    r, p = stats.pearsonr(df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float))
    return float(r), float(p), int(len(df))


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return np.nan, np.nan, int(len(df))
    rho, p = stats.spearmanr(df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float))
    return float(rho), float(p), int(len(df))


@dataclass(frozen=True)
class SampleWindow:
    name: str
    start_month: str | None = None  # inclusive, "YYYY-MM"
    end_month: str | None = None  # inclusive, "YYYY-MM"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(True, index=df.index)
        if self.start_month is not None:
            out &= df["month_id"] >= self.start_month
        if self.end_month is not None:
            out &= df["month_id"] <= self.end_month
        return out


def load_monthly_series() -> pd.DataFrame:
    tag_month = pd.read_csv(INTERNAL_TAG_MONTH)
    external = pd.read_csv(EXTERNAL_PAGEVIEWS)

    # Aggregate internal title traces to month totals/rates.
    hit_cols = [c for c in tag_month.columns if c.endswith("_hits") and c != "ai_title_hits"]
    keep_cols = ["month_id", "questions", "ai_title_hits"] + hit_cols
    month_agg = (
        tag_month[keep_cols]
        .groupby("month_id", as_index=False)
        .sum(numeric_only=True)
        .sort_values("month_id")
        .reset_index(drop=True)
    )
    month_agg["internal_ai_title_rate"] = month_agg["ai_title_hits"] / month_agg["questions"].replace(0, np.nan)
    month_agg["internal_ai_title_log_hits"] = np.log1p(month_agg["ai_title_hits"])

    # Explicit internal ChatGPT-title counts (subset of all AI patterns).
    if "chatgpt_hits" in month_agg.columns:
        month_agg["internal_chatgpt_title_rate"] = month_agg["chatgpt_hits"] / month_agg["questions"].replace(
            0, np.nan
        )
        month_agg["internal_chatgpt_title_log_hits"] = np.log1p(month_agg["chatgpt_hits"])
    else:
        month_agg["internal_chatgpt_title_rate"] = np.nan
        month_agg["internal_chatgpt_title_log_hits"] = np.nan

    df = month_agg.merge(external, on="month_id", how="left")
    df["month_dt"] = parse_month_dt(df["month_id"])
    df = df.sort_values("month_id").reset_index(drop=True)
    df["time_index"] = np.arange(len(df), dtype=int)
    df["post"] = (df["month_id"] >= POST_START_MONTH).astype(int)

    # Full-sample z-scores (useful for plotting and consistent scaling).
    df["internal_ai_title_rate_z_full"] = zscore(df["internal_ai_title_rate"])
    df["internal_ai_title_log_hits_z_full"] = zscore(df["internal_ai_title_log_hits"])
    df["internal_chatgpt_title_rate_z_full"] = zscore(df["internal_chatgpt_title_rate"])
    df["internal_chatgpt_title_log_hits_z_full"] = zscore(df["internal_chatgpt_title_log_hits"])
    df["chatgpt_log_views_z_full"] = zscore(df["chatgpt_log_views"])

    return df


def compute_correlations(series: pd.DataFrame, windows: list[SampleWindow]) -> pd.DataFrame:
    out_rows: list[dict] = []
    y_vars = [
        "internal_ai_title_rate",
        "internal_ai_title_log_hits",
        "internal_chatgpt_title_rate",
        "internal_chatgpt_title_log_hits",
    ]
    x_vars = ["chatgpt_views", "chatgpt_log_views", "chatgpt_log_views_z_full"]
    for w in windows:
        sub = series.loc[w.mask(series)].copy()

        # Sample-specific z (so "post" correlations are not dominated by the all-zero pre period).
        sub["chatgpt_z_sample"] = zscore(sub["chatgpt_log_views"])
        for y in y_vars:
            sub[f"{y}_z_sample"] = zscore(sub[y])

        for x in x_vars + ["chatgpt_z_sample"]:
            for y in y_vars + [f"{y}_z_sample" for y in y_vars]:
                pear_r, pear_p, n = safe_corr(sub[x], sub[y])
                spr_r, spr_p, _ = safe_spearman(sub[x], sub[y])
                out_rows.append(
                    {
                        "sample": w.name,
                        "x": x,
                        "y": y,
                        "pearson_r": pear_r,
                        "pearson_p": pear_p,
                        "spearman_rho": spr_r,
                        "spearman_p": spr_p,
                        "n_months": n,
                    }
                )
    return pd.DataFrame(out_rows)


def compute_lag_correlations(
    series: pd.DataFrame, windows: list[SampleWindow], max_lag: int = 6
) -> pd.DataFrame:
    rows: list[dict] = []
    base = series.copy()
    base["chatgpt_z_full"] = base["chatgpt_log_views_z_full"]

    y_vars = [
        "internal_ai_title_log_hits_z_full",
        "internal_chatgpt_title_log_hits_z_full",
        "internal_ai_title_rate_z_full",
        "internal_chatgpt_title_rate_z_full",
    ]
    x = "chatgpt_z_full"

    # lag > 0 means external leads internal by `lag` months: corr(y_t, x_{t-lag}).
    for w in windows:
        sub = base.loc[w.mask(base)].copy()
        for y in y_vars:
            for lag in range(-max_lag, max_lag + 1):
                x_l = sub[x].shift(lag)
                r, p, n = safe_corr(sub[y], x_l)
                rows.append(
                    {
                        "sample": w.name,
                        "y": y,
                        "x": x,
                        "lag_months_shift": lag,
                        "pearson_r": r,
                        "pearson_p": p,
                        "n_months": n,
                    }
                )
    return pd.DataFrame(rows)


def fit_time_series_specs(series: pd.DataFrame, windows: list[SampleWindow]) -> pd.DataFrame:
    rows: list[dict] = []
    maxlags = 6  # Newey-West HAC lag (months)

    y_vars = [
        "internal_ai_title_log_hits",
        "internal_chatgpt_title_log_hits",
        "internal_ai_title_rate",
        "internal_chatgpt_title_rate",
    ]
    specs = [
        ("ts_a", "{y} ~ chatgpt_z_sample"),
        ("ts_b", "{y} ~ chatgpt_z_sample + time_index"),
    ]
    for w in windows:
        sub = series.loc[w.mask(series)].copy()
        sub["chatgpt_z_sample"] = zscore(sub["chatgpt_log_views"])
        for y in y_vars:
            if sub[y].notna().sum() < 10:
                continue
            for spec_name, form in specs:
                model = smf.ols(form.format(y=y), data=sub).fit(
                    cov_type="HAC", cov_kwds={"maxlags": maxlags}
                )
                coef = float(model.params.get("chatgpt_z_sample", np.nan))
                se = float(model.bse.get("chatgpt_z_sample", np.nan))
                pval = float(model.pvalues.get("chatgpt_z_sample", np.nan))
                rows.append(
                    {
                        "sample": w.name,
                        "spec": spec_name,
                        "y": y,
                        "x": "chatgpt_z_sample",
                        "coef": coef,
                        "se_hac": se,
                        "pval_hac": pval,
                        "r2": float(model.rsquared),
                        "nobs": int(model.nobs),
                        "maxlags_hac": maxlags,
                    }
                )
    return pd.DataFrame(rows)


def fit_tag_month_panel_specs(series: pd.DataFrame, windows: list[SampleWindow]) -> pd.DataFrame:
    tag_month = pd.read_csv(INTERNAL_TAG_MONTH)
    external = series[["month_id", "chatgpt_log_views"]].copy()
    df = tag_month.merge(external, on="month_id", how="left")
    df["month_dt"] = parse_month_dt(df["month_id"])
    df = df.sort_values(["primary_tag", "month_id"]).reset_index(drop=True)

    # Time index within the full timeline (not within tag), so the slope is interpretable.
    month_index = (
        df[["month_id"]]
        .drop_duplicates()
        .sort_values("month_id")
        .reset_index(drop=True)
        .assign(time_index=lambda d: np.arange(len(d), dtype=int))
    )
    df = df.merge(month_index, on="month_id", how="left")

    # Rates/logs at tag-month level.
    df["internal_ai_title_rate"] = df["ai_title_hits"] / df["questions"].replace(0, np.nan)
    df["internal_ai_title_log_hits"] = np.log1p(df["ai_title_hits"])
    df["internal_chatgpt_title_rate"] = df["chatgpt_hits"] / df["questions"].replace(0, np.nan)
    df["internal_chatgpt_title_log_hits"] = np.log1p(df["chatgpt_hits"])

    # Sample-specific external z to avoid pre-period zeros dominating scale.
    rows: list[dict] = []
    y_vars = [
        "internal_ai_title_rate",
        "internal_ai_title_log_hits",
        "internal_chatgpt_title_rate",
        "internal_chatgpt_title_log_hits",
    ]
    for w in windows:
        sub = df.loc[w.mask(df)].copy()
        sub["chatgpt_z_sample"] = zscore(sub["chatgpt_log_views"])
        for y in y_vars:
            use = sub[[y, "chatgpt_z_sample", "time_index", "primary_tag", "questions"]].dropna().copy()
            if len(use) < 50:
                continue

            # Tag FE + global time trend; cluster at tag.
            model = smf.wls(
                f"{y} ~ chatgpt_z_sample + time_index + C(primary_tag)",
                data=use,
                weights=use["questions"].clip(lower=1),
            ).fit(cov_type="cluster", cov_kwds={"groups": use["primary_tag"]})

            coef = float(model.params.get("chatgpt_z_sample", np.nan))
            se = float(model.bse.get("chatgpt_z_sample", np.nan))
            pval = float(model.pvalues.get("chatgpt_z_sample", np.nan))
            rows.append(
                {
                    "sample": w.name,
                    "spec": "panel_tag_fe_trend_cluster_tag",
                    "y": y,
                    "x": "chatgpt_z_sample",
                    "coef": coef,
                    "se_cluster_tag": se,
                    "pval_cluster_tag": pval,
                    "r2": float(model.rsquared),
                    "nobs": int(model.nobs),
                    "n_tags": int(use["primary_tag"].nunique()),
                    "weight": "questions",
                }
            )
    return pd.DataFrame(rows)


def load_tag_month_panel(monthly_series: pd.DataFrame) -> pd.DataFrame:
    tag_month = pd.read_csv(INTERNAL_TAG_MONTH).copy()
    monthly = monthly_series[["month_id", "chatgpt_z", "time_index", "post"]].copy()
    df = tag_month.merge(monthly, on="month_id", how="left").copy()

    df["high_tag"] = pd.to_numeric(df["high_tag"], errors="coerce").fillna(0).astype(int)
    df["post"] = pd.to_numeric(df["post"], errors="coerce").fillna(0).astype(int)
    df["time_index"] = pd.to_numeric(df["time_index"], errors="coerce")
    df["chatgpt_z"] = pd.to_numeric(df["chatgpt_z"], errors="coerce")
    df["exposure_index"] = pd.to_numeric(df["exposure_index"], errors="coerce")
    df["questions"] = pd.to_numeric(df["questions"], errors="coerce")
    df["ai_title_hits"] = pd.to_numeric(df["ai_title_hits"], errors="coerce")
    if "cursor_hits" in df.columns:
        df["cursor_hits"] = pd.to_numeric(df["cursor_hits"], errors="coerce").fillna(0)
    else:
        df["cursor_hits"] = 0

    denom = df["questions"].replace(0, np.nan)
    df["internal_ai_title_rate"] = df["ai_title_hits"] / denom
    df["internal_ai_title_rate_excl_cursor"] = (df["ai_title_hits"] - df["cursor_hits"]).clip(lower=0) / denom
    return df


def two_way_cluster_summary(
    formula: str, data: pd.DataFrame, weight_col: str, term: str
) -> tuple[float, float, float, int, int, int]:
    model = smf.wls(formula, data=data, weights=data[weight_col]).fit()
    used = data.loc[model.model.data.row_labels]
    tag_codes = used["primary_tag"].astype("category").cat.codes
    month_codes = used["month_id"].astype("category").cat.codes
    cov = cov_cluster_2groups(model, tag_codes, month_codes)[0]
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    idx = model.model.exog_names.index(term)
    coef = float(model.params.iloc[idx])
    se_term = float(se[idx])
    pval = float(2 * stats.norm.sf(abs(coef / se_term))) if se_term > 0 else np.nan
    return coef, se_term, pval, int(model.nobs), int(used["primary_tag"].nunique()), int(used["month_id"].nunique())


def permutation_p_value(
    data: pd.DataFrame,
    outcome: str,
    weight_col: str,
    exposure_by_tag: pd.Series,
    x_modifier: pd.Series,
    seed: int = SEED,
    n_permutations: int = N_PERMUTATIONS,
) -> tuple[float, float]:
    """
    Few-cluster-friendly diagnostic: shuffle tag-level exposure values and recompute the FWL coefficient for
    x = exposure(tag) * x_modifier(row).
    """
    use = data[[outcome, weight_col, "primary_tag", "month_id", "time_index"]].copy()
    use[outcome] = pd.to_numeric(use[outcome], errors="coerce")
    use[weight_col] = pd.to_numeric(use[weight_col], errors="coerce")
    use = use.dropna()
    if len(use) == 0:
        return np.nan, np.nan

    month_d = pd.get_dummies(use["month_id"], drop_first=True)
    tag_d = pd.get_dummies(use["primary_tag"], drop_first=True)
    time_index = use["time_index"].to_numpy(dtype=float)
    trend_parts = []
    for col in tag_d.columns:
        trend_parts.append(tag_d[col].to_numpy(dtype=float) * time_index)
    trend = np.column_stack(trend_parts) if trend_parts else np.empty((len(use), 0))
    controls = np.column_stack(
        [np.ones(len(use), dtype=float), tag_d.to_numpy(dtype=float), month_d.to_numpy(dtype=float), trend]
    )

    weights = np.sqrt(use[weight_col].clip(lower=0).to_numpy(dtype=float))
    cw = controls * weights[:, None]

    # Use a projection via QR once; avoids solving a least-squares system per permutation.
    q, _ = np.linalg.qr(cw, mode="reduced")
    qt = q.T

    y = use[outcome].to_numpy(dtype=float)
    yw = y * weights
    ry = yw - q.dot(qt.dot(yw))

    tags = exposure_by_tag.index.to_list()
    exposure_vals = pd.to_numeric(exposure_by_tag, errors="coerce").to_numpy(dtype=float)
    tag_to_idx = {tag: i for i, tag in enumerate(tags)}
    tag_idx = use["primary_tag"].map(tag_to_idx).to_numpy(dtype=int)
    modifier = pd.to_numeric(x_modifier.loc[use.index], errors="coerce").to_numpy(dtype=float)

    def coef_for_exposure_vec(exposure_vec: np.ndarray) -> float:
        x = exposure_vec[tag_idx] * modifier
        xw = x * weights
        rx = xw - q.dot(qt.dot(xw))
        denom = rx @ rx
        return float((rx @ ry) / denom) if denom > 0 else np.nan

    actual_coef = coef_for_exposure_vec(exposure_vals)

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        perm_coef = coef_for_exposure_vec(rng.permutation(exposure_vals))
        if np.isfinite(perm_coef) and np.isfinite(actual_coef) and abs(perm_coef) >= abs(actual_coef):
            extreme += 1
    perm_p = (extreme + 1) / (n_permutations + 1)
    return float(actual_coef), float(perm_p)


def fit_r1_r4_linkage_regressions(panel: pd.DataFrame) -> pd.DataFrame:
    formula_template = "{y} ~ {term} + C(primary_tag) + C(month_id) + C(primary_tag):time_index"
    weight_col = "questions"

    exposure_high = panel.groupby("primary_tag")["high_tag"].first()
    exposure_cont = panel.groupby("primary_tag")["exposure_index"].first()

    specs = [
        ("R1", "internal_ai_title_rate", "high_tag:post", exposure_high, panel["post"]),
        ("R2", "internal_ai_title_rate", "exposure_index:post", exposure_cont, panel["post"]),
        ("R3", "internal_ai_title_rate", "high_tag:chatgpt_z", exposure_high, panel["chatgpt_z"]),
        ("R4_R1_excl_cursor", "internal_ai_title_rate_excl_cursor", "high_tag:post", exposure_high, panel["post"]),
        ("R4_R2_excl_cursor", "internal_ai_title_rate_excl_cursor", "exposure_index:post", exposure_cont, panel["post"]),
        ("R4_R3_excl_cursor", "internal_ai_title_rate_excl_cursor", "high_tag:chatgpt_z", exposure_high, panel["chatgpt_z"]),
    ]

    rows: list[dict] = []
    for result_id, outcome, term, exposure_by_tag, modifier in specs:
        cols = [
            outcome,
            "primary_tag",
            "month_id",
            "time_index",
            "questions",
            "high_tag",
            "post",
            "exposure_index",
            "chatgpt_z",
        ]
        data = panel[cols].dropna().copy()
        if len(data) == 0:
            continue

        coef, se, pval, nobs, n_tags, n_months = two_way_cluster_summary(
            formula_template.format(y=outcome, term=term),
            data=data,
            weight_col=weight_col,
            term=term,
        )
        perm_coef, perm_p = permutation_p_value(
            data=data,
            outcome=outcome,
            weight_col=weight_col,
            exposure_by_tag=exposure_by_tag,
            x_modifier=modifier,
        )

        rows.append(
            {
                "result_id": result_id,
                "outcome": outcome,
                "term": term,
                "weight": weight_col,
                "coef": coef,
                "cluster_2way_se": se,
                "cluster_2way_pval": pval,
                "permutation_coef": perm_coef,
                "permutation_pval": perm_p,
                "nobs": nobs,
                "n_tags": n_tags,
                "n_months": n_months,
            }
        )
    return pd.DataFrame(rows)


def build_figure_la1(series: pd.DataFrame, out_path: Path) -> None:
    tag_month = pd.read_csv(INTERNAL_TAG_MONTH).copy()
    external = pd.read_csv(EXTERNAL_PAGEVIEWS).copy()
    external = external.sort_values("month_id").reset_index(drop=True)
    external["month_dt"] = parse_month_dt(external["month_id"])

    tag_month["month_dt"] = parse_month_dt(tag_month["month_id"])
    tag_month["high_tag"] = pd.to_numeric(tag_month["high_tag"], errors="coerce").fillna(0).astype(int)

    # Weighted monthly rates: sum(hits)/sum(questions), not the mean of row-level rates.
    def weighted_rate(g: pd.DataFrame) -> float:
        q = pd.to_numeric(g["questions"], errors="coerce").fillna(0).to_numpy(dtype=float)
        h = pd.to_numeric(g["ai_title_hits"], errors="coerce").fillna(0).to_numpy(dtype=float)
        denom = q.sum()
        return float(h.sum() / denom) if denom > 0 else np.nan

    overall = (
        tag_month.groupby("month_id", as_index=False)
        .apply(lambda g: pd.Series({"overall_internal_rate": weighted_rate(g)}))
        .reset_index(drop=True)
        .merge(external[["month_id", "chatgpt_z"]], on="month_id", how="left")
    )
    overall["month_dt"] = parse_month_dt(overall["month_id"])

    group_rates = (
        tag_month.groupby(["month_id", "high_tag"], as_index=False)
        .apply(lambda g: pd.Series({"rate": weighted_rate(g)}))
        .reset_index(drop=True)
    )
    group_rates["month_dt"] = parse_month_dt(group_rates["month_id"])

    # Correlation annotation (timing triangulation only; not causal).
    corr_df = overall[["overall_internal_rate", "chatgpt_z"]].dropna()
    corr_val = float(stats.pearsonr(corr_df["overall_internal_rate"], corr_df["chatgpt_z"])[0]) if len(corr_df) >= 3 else np.nan

    post_start_dt = pd.Timestamp(f"{POST_START_MONTH}-01")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)

    # Panel A: internal rate (left axis) vs external chatgpt_z (right axis).
    ax_l = axes[0]
    ax_r = ax_l.twinx()
    ax_l.plot(
        overall["month_dt"],
        overall["overall_internal_rate"],
        color="#2ca02c",
        linewidth=2,
        label="Internal: AI-title mention rate (sum hits / sum questions)",
    )
    ax_r.plot(
        overall["month_dt"],
        overall["chatgpt_z"],
        color="#1f77b4",
        linewidth=2,
        alpha=0.85,
        label="External: ChatGPT wiki pageviews (chatgpt_z)",
    )
    ax_l.axvspan(post_start_dt, overall["month_dt"].max(), color="gray", alpha=0.12, linewidth=0)
    ax_l.axvline(EVENT_CHATGPT_RELEASE, color="black", linestyle="--", linewidth=1)
    ax_l.axvline(EVENT_CHATGPT_LAUNCH_WEEK, color="gray", linestyle=":", linewidth=1)
    ax_l.set_title(f"Panel A: External ChatGPT attention vs internal AI-title salience (corr={corr_val:.3f})")
    ax_l.set_ylabel("Internal AI-title mention rate")
    ax_r.set_ylabel("External ChatGPT attention (z)")
    h_l, l_l = ax_l.get_legend_handles_labels()
    h_r, l_r = ax_r.get_legend_handles_labels()
    ax_l.legend(h_l + h_r, l_l + l_r, frameon=False, fontsize=9, loc="upper left")
    ax_l.grid(True, axis="y", alpha=0.25)

    # Panel B: internal rate by exposure group.
    ax = axes[1]
    for high_tag, label, color in [(1, "High-exposure tags", "#d62728"), (0, "Low-exposure tags", "#2ca02c")]:
        sub = group_rates[group_rates["high_tag"] == high_tag].sort_values("month_dt")
        ax.plot(sub["month_dt"], sub["rate"], linewidth=2, color=color, label=label)
    ax.axvspan(post_start_dt, group_rates["month_dt"].max(), color="gray", alpha=0.12, linewidth=0)
    ax.axvline(EVENT_CHATGPT_RELEASE, color="black", linestyle="--", linewidth=1)
    ax.axvline(EVENT_CHATGPT_LAUNCH_WEEK, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Panel B: Internal AI-title salience by exposure group")
    ax.set_ylabel("AI-title mention rate")
    ax.set_xlabel("Month")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Figure LA1: GenAI linkage audit (existing assets only)", y=0.98, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_readout(
    series: pd.DataFrame,
    r1_r4: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# P1 GenAI Linkage Audit: Implementation Memo (2026-04-04)")
    lines.append("")
    lines.append("Ownership lane: linkage validation / triangulation only (not causal identification).")
    lines.append("")
    lines.append("## Inputs (Existing Processed Assets)")
    lines.append(f"- {INTERNAL_TAG_MONTH.as_posix()}")
    lines.append(f"- {EXTERNAL_PAGEVIEWS.as_posix()}")
    lines.append("")
    lines.append("## Outputs Written For Integration")
    lines.append(f"- {OUT_SERIES.as_posix()}")
    lines.append(f"- {OUT_R1_R4.as_posix()}")
    lines.append(f"- {OUT_SUMMARY.as_posix()}")
    lines.append(f"- {FIG_LA1.as_posix()}")
    lines.append("")
    lines.append("## Figure LA1 Spec (Main-Text Candidate)")
    lines.append("Two panels (monthly, 2020-01 to 2025-12):")
    lines.append("Panel A (calibration / timing):")
    lines.append("- External: `chatgpt_z` from Wikipedia monthly ChatGPT pageviews.")
    lines.append("- Internal: overall AI-title mention rate aggregated as `sum(ai_title_hits_it)/sum(questions_it)`.")
    lines.append(f"- Post shading: months >= `{POST_START_MONTH}`.")
    lines.append("Panel B (exposure alignment):")
    lines.append("- Internal: AI-title mention rate for `high_tag==1` vs `high_tag==0` (each as `sum(hits)/sum(questions)` by month).")
    lines.append(f"- Post shading: months >= `{POST_START_MONTH}`.")
    lines.append("")
    lines.append("## R1-R4 Regression Specs (Appendix Table)")
    lines.append("Design: tag-month panel; WLS weights=`questions_it`; include tag FE, month FE, and tag-specific linear trends:")
    lines.append("- `C(primary_tag)` + `C(month_id)` + `C(primary_tag):time_index`")
    lines.append("")
    lines.append("(R1) DID-style linkage check (binary exposure):")
    lines.append("`internal_ai_title_rate_it ~ high_tag_i:post_t + FE + trends`")
    lines.append("(R2) Continuous exposure variant:")
    lines.append("`internal_ai_title_rate_it ~ exposure_index_i:post_t + FE + trends`")
    lines.append("(R3) External-timing interaction (diffusion-aligned linkage):")
    lines.append("`internal_ai_title_rate_it ~ high_tag_i:chatgpt_z_t + FE + trends`")
    lines.append("(R4) Conservative numerator robustness:")
    lines.append("Repeat R1-R3 using `internal_ai_title_rate_excl_cursor_it = (ai_title_hits_it - cursor_hits_it)/questions_it`.")
    lines.append("")
    lines.append("Inference reported in the build table:")
    lines.append("- Two-way clustered SE (tag + month) via `cov_cluster_2groups`.")
    lines.append(f"- Permutation p-values (shuffle tag-level exposure values across tags; seed={SEED}; n={N_PERMUTATIONS}).")
    lines.append("")
    lines.append("## Key Caveats / Claim-Control")
    lines.append(
        "- External series is a single Wikipedia page (ChatGPT) and is a proxy for salience, not usage; it may "
        "move due to media cycles unrelated to developer adoption."
    )
    lines.append(
        "- Internal series is based on title string matches (not verified tool usage), so it is sensitive to naming "
        "conventions and spillovers into generic AI discussions."
    )
    lines.append(
        "- Correlation/regression results here validate timing co-movement only; they do not identify causal effects "
        "and should not be interpreted as such."
    )

    READOUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    series = load_monthly_series()
    panel = load_tag_month_panel(series)
    r1_r4 = fit_r1_r4_linkage_regressions(panel)

    series.to_csv(OUT_SERIES, index=False)
    r1_r4.to_csv(OUT_R1_R4, index=False)

    build_figure_la1(series, FIG_LA1)

    overall = (
        panel.groupby("month_id", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "overall_internal_rate": float(g["ai_title_hits"].sum() / g["questions"].sum())
                    if g["questions"].sum() > 0
                    else np.nan
                }
            )
        )
        .reset_index(drop=True)
        .merge(series[["month_id", "chatgpt_z"]], on="month_id", how="left")
    )
    corr_df = overall[["overall_internal_rate", "chatgpt_z"]].dropna()
    corr_overall = float(stats.pearsonr(corr_df["overall_internal_rate"], corr_df["chatgpt_z"])[0]) if len(corr_df) >= 3 else np.nan

    summary = {
        "inputs": {
            "internal_tag_month_csv": INTERNAL_TAG_MONTH.name,
            "external_pageviews_csv": EXTERNAL_PAGEVIEWS.name,
        },
        "outputs": {
            "monthly_series_csv": OUT_SERIES.name,
            "r1_r4_results_csv": OUT_R1_R4.name,
            "figure_la1_png": FIG_LA1.name,
            "readout_md": READOUT.name,
        },
        "event_markers": {
            "chatgpt_release": str(EVENT_CHATGPT_RELEASE.date()),
            "chatgpt_launch_week": str(EVENT_CHATGPT_LAUNCH_WEEK.date()),
            "post_start_month": POST_START_MONTH,
        },
        "n_months_total": int(series["month_id"].nunique()),
        "month_range": {
            "min": str(series["month_id"].min()),
            "max": str(series["month_id"].max()),
        },
        "n_tags_internal_trace": int(pd.read_csv(INTERNAL_TAG_MONTH, usecols=["primary_tag"])["primary_tag"].nunique()),
        "corr_overall_internal_rate_with_chatgpt_z": corr_overall,
        "regression_table_rows": int(len(r1_r4)),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_readout(series, r1_r4)


if __name__ == "__main__":
    main()
