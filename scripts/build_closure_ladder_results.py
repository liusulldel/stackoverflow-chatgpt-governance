from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper" / "staged_public_resolution"
FIGURES_DIR = BASE_DIR / "figures"

QUESTIONS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_questions.parquet"
ANSWERS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_answers.parquet"
ACCEPT_VOTES_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_accept_votes.parquet"
QUESTION_LEVEL_PARQUET = PROCESSED_DIR / "stackexchange_20251231_question_level_enriched.parquet"
EXPOSURE_CSV = PROCESSED_DIR / "strengthened_exposure_tag_scores.csv"

RESULTS_JSON = PROCESSED_DIR / "closure_ladder_results.json"
RESULTS_CSV = PROCESSED_DIR / "closure_ladder_model_results.csv"
WINDOW_CSV = PROCESSED_DIR / "closure_ladder_window_trajectory.csv"
QUESTION_LEVEL_EXTRA_PARQUET = PROCESSED_DIR / "closure_ladder_question_level.parquet"
PRIMARY_PANEL_CSV = PROCESSED_DIR / "closure_ladder_primary_panel.csv"
FRACTIONAL_PANEL_CSV = PROCESSED_DIR / "closure_ladder_fractional_panel.csv"
SUMMARY_MD = PAPER_DIR / "closure_ladder_results.md"
FIGURE_PNG = FIGURES_DIR / "closure_ladder_trajectory.png"

SHOCK_DATE = pd.Timestamp("2022-11-30T00:00:00Z")
WILD_BOOTSTRAP_REPS = 399
SELECTED_TAGS = [
    "apache-spark",
    "android",
    "bash",
    "docker",
    "excel",
    "firebase",
    "javascript",
    "kubernetes",
    "linux",
    "memory-management",
    "multithreading",
    "numpy",
    "pandas",
    "python",
    "regex",
    "sql",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    question_level = pd.read_parquet(QUESTION_LEVEL_PARQUET)
    answers = pd.read_parquet(ANSWERS_PARQUET)
    accept_votes = pd.read_parquet(ACCEPT_VOTES_PARQUET)
    questions = pd.read_parquet(QUESTIONS_PARQUET, columns=["question_id", "question_created_at", "selected_tags", "accepted_answer_id"])
    exposure = pd.read_csv(EXPOSURE_CSV)[["tag", "exposure_index"]]

    question_level["question_created_at"] = pd.to_datetime(question_level["question_created_at"], utc=True)
    answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], utc=True)
    accept_votes["accept_vote_date"] = pd.to_datetime(accept_votes["accept_vote_date"], utc=True, errors="coerce")
    questions["question_created_at"] = pd.to_datetime(questions["question_created_at"], utc=True)
    return question_level, answers, accept_votes, questions, exposure


def parse_selected_tags(tag_string: str | None) -> list[str]:
    if tag_string is None or pd.isna(tag_string):
        return []
    return [tag for tag in str(tag_string).split(";") if tag in SELECTED_TAGS]


def build_question_level(
    question_level: pd.DataFrame,
    questions: pd.DataFrame,
    answers: pd.DataFrame,
    accept_votes: pd.DataFrame,
    exposure: pd.DataFrame,
) -> pd.DataFrame:
    observation_cutoff = max(
        question_level["question_created_at"].max(),
        answers["answer_created_at"].max(),
        accept_votes["accept_vote_date"].max(),
    )

    questions = questions.copy()
    questions["selected_tags_list"] = questions["selected_tags"].apply(parse_selected_tags)
    answers = answers.merge(questions[["question_id", "question_created_at"]], on="question_id", how="inner")
    answers["delta_hours"] = (answers["answer_created_at"] - answers["question_created_at"]).dt.total_seconds() / 3600.0
    answers["in_7d"] = (answers["delta_hours"] >= 0) & (answers["delta_hours"] <= 24.0 * 7.0)
    answers["in_30d"] = (answers["delta_hours"] >= 0) & (answers["delta_hours"] <= 24.0 * 30.0)
    answers["positive_score"] = answers["score"] > 0
    answers["positive_in_7d"] = answers["in_7d"] & answers["positive_score"]
    answers["positive_delta_hours"] = answers["delta_hours"].where(answers["positive_score"])
    answers["score_30d_zeroed"] = np.where(answers["in_30d"], answers["score"].clip(lower=0), np.nan)

    positive_latency = (
        answers.loc[answers["positive_score"] & (answers["delta_hours"] >= 0), ["question_id", "delta_hours"]]
        .groupby("question_id", as_index=False)["delta_hours"]
        .min()
        .rename(columns={"delta_hours": "first_positive_answer_latency_hours"})
    )

    answer_agg = (
        answers.groupby("question_id", as_index=False)
        .agg(
            any_answer_7d=("in_7d", "max"),
            any_pos_answer_7d=("positive_in_7d", "max"),
            any_answer_30d=("in_30d", "max"),
            max_answer_score_30d_answered=("score_30d_zeroed", "max"),
        )
        .merge(positive_latency, on="question_id", how="left")
    )

    accept_dates = (
        accept_votes.groupby("answer_id", as_index=False)["accept_vote_date"]
        .min()
        .rename(columns={"answer_id": "accepted_answer_id"})
    )

    df = question_level.copy()
    df = df.merge(answer_agg, on="question_id", how="left")
    df = df.merge(accept_dates, on="accepted_answer_id", how="left", suffixes=("", "_recomputed"))

    for col in ["any_answer_7d", "any_pos_answer_7d", "any_answer_30d"]:
        df[col] = df[col].fillna(False).astype(int)

    df["first_positive_answer_latency_hours"] = df["first_positive_answer_latency_hours"].astype(float)
    df["first_positive_answer_7d"] = (
        (df["first_positive_answer_latency_hours"] >= 0)
        & (df["first_positive_answer_latency_hours"] <= 24.0 * 7.0)
    ).fillna(False).astype(int)
    df["first_positive_answer_latency_hours_capped"] = df["first_positive_answer_latency_hours"].clip(upper=24.0 * 30.0)

    df["max_answer_score_30d_answered"] = df["max_answer_score_30d_answered"].astype(float)
    df["max_answer_score_30d_zero"] = df["max_answer_score_30d_answered"].fillna(0.0)
    df["accepted_cond_any_answer_30d"] = (
        ((df["accept_vote_date_recomputed"] - df["question_created_at"]).dt.total_seconds() / 3600.0 <= 24.0 * 30.0)
        & (df["any_answer_30d"] == 1)
    ).fillna(False).astype(int)

    df["any_answer_7d_eligible"] = (df["question_created_at"] <= (observation_cutoff - pd.Timedelta(days=7))).astype(int)
    df["any_pos_answer_7d_eligible"] = df["any_answer_7d_eligible"]
    df["first_positive_answer_7d_eligible"] = df["any_answer_7d_eligible"]
    df["first_positive_answer_latency_answered_eligible"] = df["first_positive_answer_latency_hours"].notna().astype(int)
    df["max_answer_score_30d_zero_eligible"] = (df["question_created_at"] <= (observation_cutoff - pd.Timedelta(days=30))).astype(int)
    df["max_answer_score_30d_answered_eligible"] = df["max_answer_score_30d_answered"].notna().astype(int)
    df["accepted_cond_any_answer_30d_eligible"] = (
        (df["question_created_at"] <= (observation_cutoff - pd.Timedelta(days=30)))
        & (df["any_answer_30d"] == 1)
    ).astype(int)

    exposure = exposure.copy()
    exposure["high_tag"] = exposure["tag"].isin({"bash", "excel", "javascript", "numpy", "pandas", "python", "regex", "sql"}).astype(int)
    exposure_lookup = exposure.rename(columns={"tag": "primary_tag"})
    keep_cols = ["primary_tag", "exposure_index", "high_tag"]
    drop_cols = [col for col in ["exposure_index", "high_tag"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.merge(exposure_lookup[keep_cols], on="primary_tag", how="left")
    return df


def eligible_mean(frame: pd.DataFrame, value_col: str, eligible_col: str) -> tuple[float, float]:
    eligible = frame.loc[frame[eligible_col] == 1, value_col].dropna()
    if eligible.empty:
        return float("nan"), 0.0
    return float(eligible.mean()), float(len(eligible))


def eligible_rate(frame: pd.DataFrame, event_col: str, eligible_col: str) -> tuple[float, float]:
    eligible = frame.loc[frame[eligible_col] == 1, event_col]
    if eligible.empty:
        return float("nan"), 0.0
    return float(eligible.mean()), float(len(eligible))


def build_primary_panel(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    sample = df.loc[df["keep_single_focal"] == 1].copy()
    for (tag, month_id), g in sample.groupby(["primary_tag", "month_id"], sort=True):
        any_answer_7d_rate, any_answer_7d_denom = eligible_rate(g, "any_answer_7d", "any_answer_7d_eligible")
        any_pos_answer_7d_rate, any_pos_answer_7d_denom = eligible_rate(g, "any_pos_answer_7d", "any_pos_answer_7d_eligible")
        first_positive_answer_7d_rate, first_positive_answer_7d_denom = eligible_rate(g, "first_positive_answer_7d", "first_positive_answer_7d_eligible")
        first_positive_latency_mean, first_positive_latency_denom = eligible_mean(g, "first_positive_answer_latency_hours", "first_positive_answer_latency_answered_eligible")
        max_score_zero_mean, max_score_zero_denom = eligible_mean(g, "max_answer_score_30d_zero", "max_answer_score_30d_zero_eligible")
        max_score_answered_mean, max_score_answered_denom = eligible_mean(g, "max_answer_score_30d_answered", "max_answer_score_30d_answered_eligible")
        accepted_cond_rate, accepted_cond_denom = eligible_rate(g, "accepted_cond_any_answer_30d", "accepted_cond_any_answer_30d_eligible")
        accepted_vote_30d_rate, accepted_vote_30d_denom = eligible_rate(g, "accepted_30d", "accepted_30d_eligible")
        first_answer_1d_rate, first_answer_1d_denom = eligible_rate(g, "first_answer_1d", "first_answer_1d_eligible")
        rows.append(
            {
                "tag": tag,
                "month_id": month_id,
                "n_questions": int(len(g)),
                "high_tag": int(g["high_tag"].iloc[0]),
                "exposure_index": float(g["exposure_index"].iloc[0]),
                "post_chatgpt": int(g["post_chatgpt"].iloc[0]),
                "first_answer_1d_rate": first_answer_1d_rate,
                "first_answer_1d_denom": first_answer_1d_denom,
                "any_answer_7d_rate": any_answer_7d_rate,
                "any_answer_7d_denom": any_answer_7d_denom,
                "any_pos_answer_7d_rate": any_pos_answer_7d_rate,
                "any_pos_answer_7d_denom": any_pos_answer_7d_denom,
                "first_positive_answer_7d_rate": first_positive_answer_7d_rate,
                "first_positive_answer_7d_denom": first_positive_answer_7d_denom,
                "first_positive_answer_latency_mean": first_positive_latency_mean,
                "first_positive_answer_latency_denom": first_positive_latency_denom,
                "max_answer_score_30d_zero_mean": max_score_zero_mean,
                "max_answer_score_30d_zero_denom": max_score_zero_denom,
                "max_answer_score_30d_answered_mean": max_score_answered_mean,
                "max_answer_score_30d_answered_denom": max_score_answered_denom,
                "accepted_cond_any_answer_30d_rate": accepted_cond_rate,
                "accepted_cond_any_answer_30d_denom": accepted_cond_denom,
                "accepted_vote_30d_rate": accepted_vote_30d_rate,
                "accepted_vote_30d_denom": accepted_vote_30d_denom,
            }
        )
    panel = pd.DataFrame(rows)
    panel["high_post"] = panel["high_tag"] * panel["post_chatgpt"]
    panel["time_index"] = pd.factorize(panel["month_id"], sort=True)[0] + 1
    return panel.sort_values(["tag", "month_id"]).reset_index(drop=True)


def build_fractional_panel(df: pd.DataFrame) -> pd.DataFrame:
    expanded_rows: list[dict] = []
    for row in df.itertuples(index=False):
        selected_tags = list(row.selected_tags_list)
        if not selected_tags:
            continue
        weight = 1.0 / len(selected_tags)
        for tag in selected_tags:
            expanded_rows.append(
                {
                    "question_id": row.question_id,
                    "tag": tag,
                    "month_id": row.month_id,
                    "weight": weight,
                    "high_tag": int(tag in {"bash", "excel", "javascript", "numpy", "pandas", "python", "regex", "sql"}),
                    "post_chatgpt": row.post_chatgpt,
                    "first_answer_1d": row.first_answer_1d,
                    "first_answer_1d_eligible": row.first_answer_1d_eligible,
                    "any_answer_7d": row.any_answer_7d,
                    "any_answer_7d_eligible": row.any_answer_7d_eligible,
                    "any_pos_answer_7d": row.any_pos_answer_7d,
                    "any_pos_answer_7d_eligible": row.any_pos_answer_7d_eligible,
                    "first_positive_answer_7d": row.first_positive_answer_7d,
                    "first_positive_answer_7d_eligible": row.first_positive_answer_7d_eligible,
                    "first_positive_answer_latency_hours": row.first_positive_answer_latency_hours,
                    "first_positive_answer_latency_answered_eligible": row.first_positive_answer_latency_answered_eligible,
                    "max_answer_score_30d_zero": row.max_answer_score_30d_zero,
                    "max_answer_score_30d_zero_eligible": row.max_answer_score_30d_zero_eligible,
                    "max_answer_score_30d_answered": row.max_answer_score_30d_answered,
                    "max_answer_score_30d_answered_eligible": row.max_answer_score_30d_answered_eligible,
                    "accepted_cond_any_answer_30d": row.accepted_cond_any_answer_30d,
                    "accepted_cond_any_answer_30d_eligible": row.accepted_cond_any_answer_30d_eligible,
                    "accepted_30d": row.accepted_30d,
                    "accepted_30d_eligible": row.accepted_30d_eligible,
                    "exposure_index": row.exposure_index,
                }
            )
    expanded = pd.DataFrame(expanded_rows)

    rows: list[dict] = []
    for (tag, month_id), g in expanded.groupby(["tag", "month_id"], sort=True):
        def weighted_rate(event_col: str, eligible_col: str) -> tuple[float, float]:
            eligible = g.loc[g[eligible_col] == 1]
            denom = float(eligible["weight"].sum())
            if denom <= 0:
                return float("nan"), 0.0
            return float((eligible[event_col] * eligible["weight"]).sum() / denom), denom

        def weighted_mean(value_col: str, eligible_col: str) -> tuple[float, float]:
            eligible = g.loc[g[eligible_col] == 1, ["weight", value_col]].dropna()
            denom = float(eligible["weight"].sum())
            if denom <= 0:
                return float("nan"), 0.0
            return float((eligible["weight"] * eligible[value_col]).sum() / denom), denom

        first_answer_1d_rate, first_answer_1d_denom = weighted_rate("first_answer_1d", "first_answer_1d_eligible")
        any_answer_7d_rate, any_answer_7d_denom = weighted_rate("any_answer_7d", "any_answer_7d_eligible")
        any_pos_answer_7d_rate, any_pos_answer_7d_denom = weighted_rate("any_pos_answer_7d", "any_pos_answer_7d_eligible")
        first_positive_answer_7d_rate, first_positive_answer_7d_denom = weighted_rate("first_positive_answer_7d", "first_positive_answer_7d_eligible")
        first_positive_latency_mean, first_positive_latency_denom = weighted_mean("first_positive_answer_latency_hours", "first_positive_answer_latency_answered_eligible")
        max_score_zero_mean, max_score_zero_denom = weighted_mean("max_answer_score_30d_zero", "max_answer_score_30d_zero_eligible")
        max_score_answered_mean, max_score_answered_denom = weighted_mean("max_answer_score_30d_answered", "max_answer_score_30d_answered_eligible")
        accepted_cond_rate, accepted_cond_denom = weighted_rate("accepted_cond_any_answer_30d", "accepted_cond_any_answer_30d_eligible")
        accepted_vote_30d_rate, accepted_vote_30d_denom = weighted_rate("accepted_30d", "accepted_30d_eligible")
        rows.append(
            {
                "tag": tag,
                "month_id": month_id,
                "n_questions_weighted": float(g["weight"].sum()),
                "high_tag": int(g["high_tag"].iloc[0]),
                "exposure_index": float(g["exposure_index"].iloc[0]),
                "post_chatgpt": int(g["post_chatgpt"].iloc[0]),
                "first_answer_1d_rate": first_answer_1d_rate,
                "first_answer_1d_denom": first_answer_1d_denom,
                "any_answer_7d_rate": any_answer_7d_rate,
                "any_answer_7d_denom": any_answer_7d_denom,
                "any_pos_answer_7d_rate": any_pos_answer_7d_rate,
                "any_pos_answer_7d_denom": any_pos_answer_7d_denom,
                "first_positive_answer_7d_rate": first_positive_answer_7d_rate,
                "first_positive_answer_7d_denom": first_positive_answer_7d_denom,
                "first_positive_answer_latency_mean": first_positive_latency_mean,
                "first_positive_answer_latency_denom": first_positive_latency_denom,
                "max_answer_score_30d_zero_mean": max_score_zero_mean,
                "max_answer_score_30d_zero_denom": max_score_zero_denom,
                "max_answer_score_30d_answered_mean": max_score_answered_mean,
                "max_answer_score_30d_answered_denom": max_score_answered_denom,
                "accepted_cond_any_answer_30d_rate": accepted_cond_rate,
                "accepted_cond_any_answer_30d_denom": accepted_cond_denom,
                "accepted_vote_30d_rate": accepted_vote_30d_rate,
                "accepted_vote_30d_denom": accepted_vote_30d_denom,
            }
        )
    panel = pd.DataFrame(rows)
    panel["high_post"] = panel["high_tag"] * panel["post_chatgpt"]
    panel["time_index"] = pd.factorize(panel["month_id"], sort=True)[0] + 1
    return panel.sort_values(["tag", "month_id"]).reset_index(drop=True)


def fit_weighted(formula: str, data: pd.DataFrame, weight_col: str):
    return smf.wls(formula, data=data, weights=data[weight_col]).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["tag"], "use_correction": True, "df_correction": True},
    )


def remove_term_from_formula(formula: str, term: str) -> str:
    lhs, rhs = formula.split("~", 1)
    rhs_terms = [piece.strip() for piece in rhs.split("+")]
    kept_terms = [piece for piece in rhs_terms if piece != term]
    return f"{lhs.strip()} ~ {' + '.join(kept_terms)}"


def wild_cluster_bootstrap_pvalue(formula: str, data: pd.DataFrame, weight_col: str, term: str, seed: int) -> float:
    groups = data["tag"].to_numpy()
    unique_groups = pd.Index(sorted(pd.unique(groups)))
    if len(unique_groups) < 6:
        return float("nan")

    restricted_formula = remove_term_from_formula(formula, term)
    y_full, x_full = patsy.dmatrices(formula, data, return_type="dataframe")
    y_restricted, x_restricted = patsy.dmatrices(restricted_formula, data, return_type="dataframe")
    weights = data[weight_col].to_numpy()

    full_fit = sm.WLS(y_full.iloc[:, 0], x_full, weights=weights).fit()
    full_cluster = full_fit.get_robustcov_results(
        cov_type="cluster",
        groups=groups,
        use_correction=True,
        df_correction=True,
    )
    term_index = list(x_full.columns).index(term)
    observed_t = float(full_cluster.params[term_index] / full_cluster.bse[term_index])

    restricted_fit = sm.WLS(y_restricted.iloc[:, 0], x_restricted, weights=weights).fit()
    fitted_restricted = restricted_fit.fittedvalues.to_numpy()
    residuals = y_restricted.iloc[:, 0].to_numpy() - fitted_restricted

    rng = np.random.default_rng(seed)
    group_codes = pd.Categorical(groups, categories=unique_groups).codes
    bootstrap_t: list[float] = []

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
        return float("nan")
    return float(np.mean(np.abs(np.asarray(bootstrap_t)) >= abs(observed_t)))


def fit_spec(name: str, formula: str, data: pd.DataFrame, weight_col: str, term: str, bootstrap: bool = False) -> dict:
    sample = data.loc[data[weight_col] > 0].copy()
    model = fit_weighted(formula, sample, weight_col)
    result = {
        "specification": name,
        "outcome": formula.split("~", 1)[0].strip(),
        "term": term,
        "coef": float(model.params.get(term, np.nan)),
        "se": float(model.bse.get(term, np.nan)),
        "pval": float(model.pvalues.get(term, np.nan)),
        "nobs": int(model.nobs),
        "weight_col": weight_col,
        "formula": formula,
        "wild_cluster_bootstrap_pval": float("nan"),
    }
    if bootstrap:
        result["wild_cluster_bootstrap_pval"] = wild_cluster_bootstrap_pvalue(formula, sample, weight_col, term, seed=abs(hash(name)) % (2**32))
    return result


def fit_window_trajectory(panel: pd.DataFrame) -> pd.DataFrame:
    windows = [
        ("through_2023_02", "2023-02"),
        ("through_2023_06", "2023-06"),
        ("through_2023_12", "2023-12"),
        ("through_2024_12", "2024-12"),
        ("full_2025", None),
    ]
    rows: list[dict] = []
    for window_name, end_month in windows:
        frame = panel.copy()
        if end_month is not None:
            frame = frame.loc[frame["month_id"] <= end_month].copy()
        frame["time_index"] = pd.factorize(frame["month_id"], sort=True)[0] + 1
        for outcome, weight_col in [
            ("first_answer_1d_rate", "first_answer_1d_denom"),
            ("any_answer_7d_rate", "any_answer_7d_denom"),
            ("accepted_vote_30d_rate", "accepted_vote_30d_denom"),
        ]:
            data = frame.loc[frame[weight_col] > 0].copy()
            model = fit_weighted(
                f"{outcome} ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
                data,
                weight_col,
            )
            rows.append(
                {
                    "window": window_name,
                    "outcome": outcome,
                    "coef": float(model.params.get("high_post", np.nan)),
                    "se": float(model.bse.get("high_post", np.nan)),
                    "pval": float(model.pvalues.get("high_post", np.nan)),
                    "nobs": int(model.nobs),
                }
            )
    return pd.DataFrame(rows)


def build_figure(panel: pd.DataFrame) -> None:
    monthly = (
        panel.groupby(["month_id", "high_tag"], as_index=False)
        .agg(
            first_answer_1d_rate=("first_answer_1d_rate", "mean"),
            accepted_vote_30d_rate=("accepted_vote_30d_rate", "mean"),
        )
    )
    monthly["month_start"] = pd.to_datetime(monthly["month_id"] + "-01", utc=True)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for axis, column, title in [
        (axes[0], "first_answer_1d_rate", "Rapid Public Answer Supply"),
        (axes[1], "accepted_vote_30d_rate", "Formalized Accepted Resolution"),
    ]:
        for high_tag, label, color in [(1, "Higher substitutability", "#B3472E"), (0, "Lower substitutability", "#1F5A7A")]:
            subset = monthly.loc[monthly["high_tag"] == high_tag].sort_values("month_start")
            axis.plot(subset["month_start"], subset[column], label=label, color=color, linewidth=2)
        axis.axvline(SHOCK_DATE, color="#444444", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel("Monthly mean rate")
        axis.grid(alpha=0.25)

    axes[1].set_xlabel("Month")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(model_df: pd.DataFrame, window_df: pd.DataFrame) -> None:
    headline = model_df.loc[model_df["specification"] == "primary_first_answer_1d"].iloc[0]
    any_answer = model_df.loc[model_df["specification"] == "primary_any_answer_7d"].iloc[0]
    latency = model_df.loc[model_df["specification"] == "primary_first_positive_latency"].iloc[0]
    accepted_cond = model_df.loc[model_df["specification"] == "primary_accepted_cond_any_answer_30d"].iloc[0]
    accepted_vote = model_df.loc[model_df["specification"] == "primary_accepted_vote_30d"].iloc[0]

    lines = [
        "# Closure Ladder Results",
        "",
        "## Read",
        "",
        "This memo extends the current canonical `2025`-backed manuscript into a clearer staged-resolution ladder.",
        "",
        f"- Headline result remains `first_answer_1d`: coefficient `{headline['coef']:.4f}`, clustered `p = {headline['pval']:.4f}`, wild bootstrap `p = {headline['wild_cluster_bootstrap_pval']:.4f}`.",
        f"- `any_answer_7d` points in the same direction: coefficient `{any_answer['coef']:.4f}`, clustered `p = {any_answer['pval']:.4f}`, wild bootstrap `p = {any_answer['wild_cluster_bootstrap_pval']:.4f}`.",
        f"- `first_positive_answer_latency_mean` worsens sharply in higher-substitutability domains: coefficient `{latency['coef']:.2f}` hours, clustered `p = {latency['pval']:.4f}`, wild bootstrap `p = {latency['wild_cluster_bootstrap_pval']:.4f}`.",
        f"- `accepted_cond_any_answer_30d` is near zero: coefficient `{accepted_cond['coef']:.4f}`, clustered `p = {accepted_cond['pval']:.4f}`.",
        f"- `accepted_vote_30d` remains positive and divergent: coefficient `{accepted_vote['coef']:.4f}`, clustered `p = {accepted_vote['pval']:.4f}`, wild bootstrap `p = {accepted_vote['wild_cluster_bootstrap_pval']:.4f}`.",
        "",
        "## Main Table",
        "",
        model_df[
            [
                "specification",
                "coef",
                "se",
                "pval",
                "wild_cluster_bootstrap_pval",
                "nobs",
            ]
        ].to_markdown(index=False),
        "",
        "## Window Trajectory",
        "",
        window_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The ladder is now clearer. The strongest deterioration is in rapid public answer supply and the time needed to reach the first positively scored answer. That is consistent with weaker queue responsiveness. By contrast, conditional accepted closure among answered questions does not deteriorate in the same direction, and formalized accepted-vote timing remains divergent. This makes the current manuscript more defensible as a staged public resolution paper than as a generic closure-collapse paper.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    question_level, answers, accept_votes, questions, exposure = load_inputs()
    closure_question_level = build_question_level(question_level, questions, answers, accept_votes, exposure)
    primary_panel = build_primary_panel(closure_question_level)
    fractional_panel = build_fractional_panel(closure_question_level)

    formula = "{outcome} ~ high_post + C(tag):time_index + C(tag) + C(month_id)"
    model_rows = [
        fit_spec("primary_first_answer_1d", formula.format(outcome="first_answer_1d_rate"), primary_panel, "first_answer_1d_denom", "high_post", bootstrap=True),
        fit_spec("fractional_first_answer_1d", formula.format(outcome="first_answer_1d_rate"), fractional_panel, "first_answer_1d_denom", "high_post"),
        fit_spec("primary_any_answer_7d", formula.format(outcome="any_answer_7d_rate"), primary_panel, "any_answer_7d_denom", "high_post", bootstrap=True),
        fit_spec("fractional_any_answer_7d", formula.format(outcome="any_answer_7d_rate"), fractional_panel, "any_answer_7d_denom", "high_post"),
        fit_spec("primary_any_pos_answer_7d", formula.format(outcome="any_pos_answer_7d_rate"), primary_panel, "any_pos_answer_7d_denom", "high_post"),
        fit_spec("fractional_any_pos_answer_7d", formula.format(outcome="any_pos_answer_7d_rate"), fractional_panel, "any_pos_answer_7d_denom", "high_post"),
        fit_spec("primary_first_positive_latency", formula.format(outcome="first_positive_answer_latency_mean"), primary_panel, "first_positive_answer_latency_denom", "high_post", bootstrap=True),
        fit_spec("fractional_first_positive_latency", formula.format(outcome="first_positive_answer_latency_mean"), fractional_panel, "first_positive_answer_latency_denom", "high_post"),
        fit_spec("primary_max_score_30d_zero", formula.format(outcome="max_answer_score_30d_zero_mean"), primary_panel, "max_answer_score_30d_zero_denom", "high_post", bootstrap=True),
        fit_spec("fractional_max_score_30d_zero", formula.format(outcome="max_answer_score_30d_zero_mean"), fractional_panel, "max_answer_score_30d_zero_denom", "high_post"),
        fit_spec("primary_accepted_cond_any_answer_30d", formula.format(outcome="accepted_cond_any_answer_30d_rate"), primary_panel, "accepted_cond_any_answer_30d_denom", "high_post", bootstrap=True),
        fit_spec("fractional_accepted_cond_any_answer_30d", formula.format(outcome="accepted_cond_any_answer_30d_rate"), fractional_panel, "accepted_cond_any_answer_30d_denom", "high_post"),
        fit_spec("primary_accepted_vote_30d", formula.format(outcome="accepted_vote_30d_rate"), primary_panel, "accepted_vote_30d_denom", "high_post", bootstrap=True),
        fit_spec("fractional_accepted_vote_30d", formula.format(outcome="accepted_vote_30d_rate"), fractional_panel, "accepted_vote_30d_denom", "high_post"),
    ]
    model_df = pd.DataFrame(model_rows)
    window_df = fit_window_trajectory(primary_panel)

    closure_question_level.to_parquet(QUESTION_LEVEL_EXTRA_PARQUET, index=False)
    primary_panel.to_csv(PRIMARY_PANEL_CSV, index=False)
    fractional_panel.to_csv(FRACTIONAL_PANEL_CSV, index=False)
    model_df.to_csv(RESULTS_CSV, index=False)
    window_df.to_csv(WINDOW_CSV, index=False)
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "model_results": model_rows,
                "window_trajectory": window_df.to_dict(orient="records"),
                "observation_cutoff": str(max(question_level["question_created_at"].max(), answers["answer_created_at"].max(), accept_votes["accept_vote_date"].max())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    build_figure(primary_panel)
    write_summary(model_df, window_df)


if __name__ == "__main__":
    main()
