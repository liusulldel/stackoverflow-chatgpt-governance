from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper"

QUESTIONS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_questions.parquet"
ANSWERS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_answers.parquet"
PRIMARY_PANEL_CSV = PROCESSED_DIR / "stackexchange_20251231_primary_panel.csv"
FRACTIONAL_PANEL_CSV = PROCESSED_DIR / "stackexchange_20251231_fractional_panel.csv"
PANEL_SUMMARY_JSON = PROCESSED_DIR / "stackexchange_20251231_panel_summary.json"
VALIDATION_REPORT_JSON = PROCESSED_DIR / "stackexchange_20251231_validation_report.json"

RESULTS_JSON = PROCESSED_DIR / "extended_sample_results.json"
MODEL_RESULTS_CSV = PROCESSED_DIR / "extended_sample_model_results.csv"
WINDOW_RESULTS_CSV = PROCESSED_DIR / "extended_sample_window_trajectory.csv"
BOOTSTRAP_RESULTS_CSV = PROCESSED_DIR / "extended_sample_wild_cluster_bootstrap.csv"
SUMMARY_MD = PAPER_DIR / "extended_sample_result_readout.md"

SHOCK_DATE = pd.Timestamp("2022-11-30T00:00:00Z")
HIGH_EXPOSURE_TAGS = {
    "bash",
    "excel",
    "javascript",
    "numpy",
    "pandas",
    "python",
    "regex",
    "sql",
}
WILD_BOOTSTRAP_REPS = 399


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = pd.read_csv(PRIMARY_PANEL_CSV)
    fractional = pd.read_csv(FRACTIONAL_PANEL_CSV)
    for frame in (primary, fractional):
        frame["time_index"] = pd.factorize(frame["month_id"], sort=True)[0] + 1
    return primary, fractional


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


def wild_cluster_bootstrap_pvalue(
    formula: str,
    data: pd.DataFrame,
    weight_col: str,
    term: str,
    seed: int,
) -> float:
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
    bootstrap_array = np.asarray(bootstrap_t, dtype=float)
    return float(np.mean(np.abs(bootstrap_array) >= abs(observed_t)))


def build_accepted_answer_created_panel() -> pd.DataFrame:
    questions = pd.read_parquet(QUESTIONS_PARQUET)
    answers = pd.read_parquet(ANSWERS_PARQUET)

    questions["question_created_at"] = pd.to_datetime(questions["question_created_at"], utc=True)
    answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], utc=True)

    accepted_lookup = answers[["answer_id", "answer_created_at"]].rename(
        columns={"answer_id": "accepted_answer_id", "answer_created_at": "accepted_answer_created_at"}
    )
    df = questions.merge(accepted_lookup, on="accepted_answer_id", how="left")
    df["selected_tags_list"] = df["selected_tags"].fillna("").apply(
        lambda value: [tag for tag in str(value).split(";") if tag]
    )
    df["selected_tag_overlap"] = df["selected_tags_list"].str.len()
    df = df.loc[df["selected_tag_overlap"] == 1].copy()
    df["tag"] = df["selected_tags_list"].str[0]
    df["high_tag"] = df["tag"].isin(HIGH_EXPOSURE_TAGS).astype(int)
    df["month_id"] = df["question_created_at"].dt.strftime("%Y-%m")
    df["post_chatgpt"] = (df["question_created_at"] >= SHOCK_DATE).astype(int)
    df["high_post"] = df["high_tag"] * df["post_chatgpt"]

    observation_cutoff = max(questions["question_created_at"].max(), answers["answer_created_at"].max())
    delta_hours = (df["accepted_answer_created_at"] - df["question_created_at"]).dt.total_seconds() / 3600.0
    df["accepted_answer_created_7d"] = ((delta_hours >= 0) & (delta_hours <= 24.0 * 7.0)).fillna(False).astype(int)
    df["accepted_answer_created_7d_denom"] = (
        df["question_created_at"] <= (observation_cutoff - pd.Timedelta(days=7))
    ).astype(int)

    rows: list[dict] = []
    for (tag, month_id), g in df.groupby(["tag", "month_id"], sort=True):
        denom = float(g["accepted_answer_created_7d_denom"].sum())
        rate = float(g.loc[g["accepted_answer_created_7d_denom"] == 1, "accepted_answer_created_7d"].mean()) if denom > 0 else float("nan")
        rows.append(
            {
                "tag": tag,
                "month_id": month_id,
                "n_questions": int(len(g)),
                "high_tag": int(g["high_tag"].iloc[0]),
                "accepted_answer_created_7d_rate": rate,
                "accepted_answer_created_7d_denom": denom,
            }
        )

    panel = pd.DataFrame(rows)
    panel["post_chatgpt"] = (pd.to_datetime(panel["month_id"] + "-01", utc=True) >= SHOCK_DATE).astype(int)
    panel["high_post"] = panel["high_tag"] * panel["post_chatgpt"]
    panel["time_index"] = pd.factorize(panel["month_id"], sort=True)[0] + 1
    return panel.sort_values(["tag", "month_id"]).reset_index(drop=True)


def fit_spec(name: str, formula: str, data: pd.DataFrame, weight_col: str, term: str, seed: int) -> dict:
    model = fit_weighted(formula, data, weight_col)
    return {
        "specification": name,
        "outcome": formula.split("~", 1)[0].strip(),
        "term": term,
        "weight_col": weight_col,
        "coef": float(model.params.get(term, np.nan)),
        "se": float(model.bse.get(term, np.nan)),
        "pval": float(model.pvalues.get(term, np.nan)),
        "nobs": int(model.nobs),
        "wild_cluster_bootstrap_pval": wild_cluster_bootstrap_pvalue(formula, data, weight_col, term, seed),
        "formula": formula,
    }


def fit_window_trajectory(panel: pd.DataFrame) -> pd.DataFrame:
    windows = [
        ("through_2023_02", "2023-02"),
        ("through_2023_06", "2023-06"),
        ("through_2023_12", "2023-12"),
        ("through_2024_12", "2024-12"),
        ("full_2025", None),
    ]
    rows: list[dict] = []
    outcomes = [
        ("accepted_7d_rate", "accepted_7d_denom"),
        ("accepted_30d_rate", "accepted_30d_denom"),
        ("first_answer_1d_rate", "first_answer_1d_denom"),
    ]

    for window_name, end_month in windows:
        frame = panel.copy()
        if end_month is not None:
            frame = frame.loc[frame["month_id"] <= end_month].copy()
        frame["time_index"] = pd.factorize(frame["month_id"], sort=True)[0] + 1
        for outcome, weight_col in outcomes:
            data = frame.loc[frame[weight_col] > 0].copy()
            formula = f"{outcome} ~ high_post + C(tag):time_index + C(tag) + C(month_id)"
            model = fit_weighted(formula, data, weight_col)
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


def write_summary(
    model_rows: list[dict],
    window_df: pd.DataFrame,
    panel_summary: dict,
    validation_summary: dict,
) -> None:
    model_df = pd.DataFrame(model_rows)
    lines = [
        "# Extended Sample Result Readout",
        "",
        "## Data Backbone",
        "",
        f"- Observation cutoff: `{panel_summary['observation_cutoff']}`",
        f"- Question-level rows: `{panel_summary['n_question_level_rows']}`",
        f"- Primary panel rows: `{panel_summary['n_primary_panel_rows']}`",
        f"- Fractional panel rows: `{panel_summary['n_fractional_panel_rows']}`",
        f"- Validation checks passed: `{validation_summary['n_pass']}` / fails: `{validation_summary['n_fail']}`",
        "",
        "## Core Result Split",
        "",
        "The extended sample does not support a single-direction closure story across all outcome definitions.",
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
        "## Read",
        "",
        "- `accepted_vote_7d_primary` and `accepted_vote_30d_primary` are positive in the extended sample.",
        "- `first_answer_1d_primary` remains negative and statistically strong.",
        "- `accepted_answer_created_7d_primary` is also negative, which aligns with slower arrival of the eventually accepted answer even while accept-vote timing moves the other way.",
        "- `accepted_archive_primary` stays negative but statistically weak.",
        "",
        "## Window Trajectory",
        "",
        window_df.to_markdown(index=False),
        "",
        "## Interpretation Boundary",
        "",
        "- The extended sample strengthens the engineering backbone and the censoring logic.",
        "- It also reveals a substantive construct split: answer-generation measures and accept-vote timing do not tell the same story.",
        "- That means the manuscript cannot simply port the old early-sample claim onto the 2020-2025 backbone without redefining the headline construct.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    primary_panel, fractional_panel = load_panels()
    accepted_answer_created_panel = build_accepted_answer_created_panel()

    model_rows = [
        fit_spec(
            "accepted_vote_7d_primary",
            "accepted_7d_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
            primary_panel.loc[primary_panel["accepted_7d_denom"] > 0].copy(),
            "accepted_7d_denom",
            "high_post",
            20260403,
        ),
        fit_spec(
            "accepted_vote_7d_fractional",
            "accepted_7d_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
            fractional_panel.loc[fractional_panel["accepted_7d_denom"] > 0].copy(),
            "accepted_7d_denom",
            "high_post",
            20260404,
        ),
        fit_spec(
            "accepted_vote_30d_primary",
            "accepted_30d_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
            primary_panel.loc[primary_panel["accepted_30d_denom"] > 0].copy(),
            "accepted_30d_denom",
            "high_post",
            20260405,
        ),
        fit_spec(
            "first_answer_1d_primary",
            "first_answer_1d_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
            primary_panel.loc[primary_panel["first_answer_1d_denom"] > 0].copy(),
            "first_answer_1d_denom",
            "high_post",
            20260406,
        ),
        fit_spec(
            "accepted_answer_created_7d_primary",
            "accepted_answer_created_7d_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
            accepted_answer_created_panel.loc[accepted_answer_created_panel["accepted_answer_created_7d_denom"] > 0].copy(),
            "accepted_answer_created_7d_denom",
            "high_post",
            20260407,
        ),
        fit_spec(
            "accepted_archive_primary",
            "accepted_archive_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
            primary_panel.loc[primary_panel["n_questions"] > 0].copy(),
            "n_questions",
            "high_post",
            20260408,
        ),
    ]

    model_df = pd.DataFrame(model_rows)
    bootstrap_df = model_df[
        ["specification", "coef", "pval", "wild_cluster_bootstrap_pval", "nobs"]
    ].copy()
    window_df = fit_window_trajectory(primary_panel)

    panel_summary = json.loads(PANEL_SUMMARY_JSON.read_text(encoding="utf-8"))
    validation_summary = json.loads(VALIDATION_REPORT_JSON.read_text(encoding="utf-8"))

    RESULTS_JSON.write_text(
        json.dumps(
            {
                "panel_summary": panel_summary,
                "validation_summary": {
                    "n_pass": validation_summary["n_pass"],
                    "n_fail": validation_summary["n_fail"],
                },
                "model_results": model_rows,
                "window_trajectory": window_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model_df.to_csv(MODEL_RESULTS_CSV, index=False)
    bootstrap_df.to_csv(BOOTSTRAP_RESULTS_CSV, index=False)
    window_df.to_csv(WINDOW_RESULTS_CSV, index=False)
    write_summary(model_rows, window_df, panel_summary, validation_summary)

    print(RESULTS_JSON)
    print(MODEL_RESULTS_CSV)
    print(BOOTSTRAP_RESULTS_CSV)
    print(WINDOW_RESULTS_CSV)
    print(SUMMARY_MD)


if __name__ == "__main__":
    main()
