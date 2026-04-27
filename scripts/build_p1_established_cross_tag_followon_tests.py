from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

DURABILITY_ENTRANT_PANEL = PROCESSED / "who_still_answers_durability_entrant_panel.parquet"
SUBTYPE_CONSEQUENCE_PANEL = PROCESSED / "p1_jmis_subtype_consequence_panel.csv"
ROLE_QUESTION_PANEL = PROCESSED / "who_still_answers_answer_role_question_panel.parquet"
QUESTION_CLOSURE_PANEL = PROCESSED / "who_still_answers_question_closure_panel.parquet"
FOCAL_ANSWERS = PROCESSED / "stackexchange_20251231_focal_answers.parquet"
ENTRANT_PROFILES = PROCESSED / "who_still_answers_entrant_profiles.csv"
SELECTION_PANEL = PROCESSED / "selection_composition_primary_panel.csv"
PRESHOCK_COHORTS = PROCESSED / "who_still_answers_user_tag_preshock_cohorts.csv"
INCUMBENT_COHORT_PANEL = PROCESSED / "who_still_answers_incumbent_cohort_panel.csv"

ESTABLISHED_DURABILITY_PANEL_CSV = PROCESSED / "p1_established_repeat_split_durability_panel.csv"
ESTABLISHED_DURABILITY_RESULTS_CSV = PROCESSED / "p1_established_repeat_split_durability_results.csv"
ESTABLISHED_SPLIT_PANEL_CSV = PROCESSED / "p1_established_repeat_split_tag_month.csv"
ESTABLISHED_SPLIT_CONSEQUENCE_RESULTS_CSV = PROCESSED / "p1_established_repeat_split_consequence_results.csv"

ESTABLISHED_STAGE_ROLE_PANEL_CSV = PROCESSED / "p1_established_stage_role_panel.csv"
ESTABLISHED_STAGE_ROLE_RESULTS_CSV = PROCESSED / "p1_established_stage_role_results.csv"
ESTABLISHED_STAGE_ROLE_MEANS_CSV = PROCESSED / "p1_established_stage_role_means.csv"

ESTABLISHED_LOCAL_DEPTH_TAG_SUMMARY_CSV = PROCESSED / "p1_established_local_depth_tag_summary.csv"
ESTABLISHED_LOCAL_DEPTH_RESULTS_CSV = PROCESSED / "p1_established_local_depth_results.csv"

READOUT_MD = PAPER / "p1_established_cross_tag_followon_tests_readout_2026-04-05.md"

ROLE_ORDER = [
    "first_answer",
    "first_positive",
    "top_score",
    "accepted_current",
    "later_answer",
    "accepted_30d",
]


def standardized(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = values.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - values.mean()) / std


def fit_wls(frame: pd.DataFrame, formula: str, weight_col: str, cluster_col: str = "primary_tag"):
    model_frame = frame.dropna(subset=[weight_col, cluster_col]).copy()
    y, X = dmatrices(formula, data=model_frame, return_type="dataframe", NA_action="drop")
    weights = model_frame.loc[X.index, weight_col].astype(float)
    groups = model_frame.loc[X.index, cluster_col]
    return sm.WLS(y, X, weights=weights).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True, "df_correction": True},
    )


def build_established_repeat_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entrants = pd.read_parquet(DURABILITY_ENTRANT_PANEL)
    entrants["entry_month"] = entrants["entry_month"].astype(str)
    month_order = {month: idx + 1 for idx, month in enumerate(sorted(entrants["entry_month"].dropna().unique()))}
    entrants["time_index"] = entrants["entry_month"].map(month_order)
    entrants["high_post"] = entrants["high_tag"] * entrants["post_chatgpt"]
    entrants["exposure_post"] = entrants["exposure_index"] * entrants["post_chatgpt"]

    established = entrants.loc[entrants["entrant_type"] == "established_cross_tag"].copy()

    panel_rows: list[pd.DataFrame] = []
    result_rows: list[dict[str, object]] = []
    outcome_specs = [
        ("return_365d", "eligible_365d"),
        ("one_shot_365d", "eligible_365d"),
        ("answers_365d", "eligible_365d"),
        ("active_months_365d", "eligible_365d"),
        ("low_repeat_90d", "eligible_90d"),
    ]
    for outcome, elig_col in outcome_specs:
        sub = established.loc[established[elig_col] == 1].copy()
        panel = (
            sub.groupby(
                ["primary_tag", "entry_month", "time_index", "post_chatgpt", "exposure_index", "high_tag"],
                as_index=False,
            )
            .agg(
                rate=(outcome, "mean"),
                n_eligible=("owner_user_id", "size"),
            )
        )
        panel["high_post"] = panel["high_tag"] * panel["post_chatgpt"]
        panel["exposure_post"] = panel["exposure_index"] * panel["post_chatgpt"]
        panel["outcome"] = outcome
        panel_rows.append(panel)

        for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
            formula = f"rate ~ {term} + C(primary_tag):time_index + C(primary_tag) + C(entry_month)"
            model = fit_wls(panel, formula, "n_eligible")
            result_rows.append(
                {
                    "test_block": "durability_within_established",
                    "family": family,
                    "outcome": outcome,
                    "term": term,
                    "coef": float(model.params.get(term, np.nan)),
                    "se": float(model.bse.get(term, np.nan)),
                    "pval": float(model.pvalues.get(term, np.nan)),
                    "nobs": int(model.nobs),
                    "mean_outcome": float(panel["rate"].mean()),
                }
            )

    durability_panel = pd.concat(panel_rows, ignore_index=True)
    durability_panel.to_csv(ESTABLISHED_DURABILITY_PANEL_CSV, index=False)
    durability_results = pd.DataFrame(result_rows)
    durability_results.to_csv(ESTABLISHED_DURABILITY_RESULTS_CSV, index=False)

    eligible = entrants.loc[entrants["eligible_365d"] == 1].copy()
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
            established_repeat_share_365d=(
                "entrant_type",
                lambda s: np.mean(
                    (s == "established_cross_tag")
                    & (eligible.loc[s.index, "one_shot_365d"].fillna(0).astype(float) == 0.0)
                ),
            ),
            established_total_share_365d=("entrant_type", lambda s: np.mean(s == "established_cross_tag")),
        )
    )
    split_panel["high_post"] = split_panel["high_tag"] * split_panel["post_chatgpt"]
    split_panel["exposure_post"] = split_panel["exposure_index"] * split_panel["post_chatgpt"]
    split_panel.to_csv(ESTABLISHED_SPLIT_PANEL_CSV, index=False)

    consequence_panel = pd.read_csv(SUBTYPE_CONSEQUENCE_PANEL)
    consequence_panel["month_id"] = consequence_panel["month_id"].astype(str)
    joined = consequence_panel.merge(
        split_panel,
        left_on=["primary_tag", "month_id"],
        right_on=["primary_tag", "entry_month"],
        how="inner",
    )
    consequence_rows: list[dict[str, object]] = []
    outcomes = [
        ("any_answer_7d_rate", "any_answer_7d_denom"),
        ("first_positive_answer_latency_mean", "first_positive_answer_latency_denom"),
        ("accepted_cond_any_answer_30d_rate", "accepted_cond_any_answer_30d_denom"),
        ("accepted_vote_30d_rate", "accepted_vote_30d_denom"),
        ("first_answer_1d_rate_closure", "first_answer_1d_denom_closure"),
    ]
    for family, term in [("binary", "high_post_x"), ("continuous", "exposure_post_x")]:
        # merge created duplicated high_post/exposure_post columns; use consequence-surface terms
        actual_term = "high_post_x" if family == "binary" else "exposure_post_x"
        for outcome, weight_col in outcomes:
            formula = (
                f"{outcome} ~ {actual_term} + established_one_shot_share_365d + established_repeat_share_365d + "
                "C(primary_tag):time_index_x + C(primary_tag) + C(month_id)"
            )
            renamed = joined.rename(
                columns={
                    "time_index_x": "time_index_x",
                    "high_post_x": "high_post_x",
                    "exposure_post_x": "exposure_post_x",
                }
            )
            model = fit_wls(renamed, formula, weight_col)
            for model_term in [actual_term, "established_one_shot_share_365d", "established_repeat_share_365d"]:
                consequence_rows.append(
                    {
                        "test_block": "repeat_split_consequence",
                        "family": family,
                        "outcome": outcome,
                        "term": model_term.replace("_x", ""),
                        "coef": float(model.params.get(model_term, np.nan)),
                        "se": float(model.bse.get(model_term, np.nan)),
                        "pval": float(model.pvalues.get(model_term, np.nan)),
                        "nobs": int(model.nobs),
                        "mean_outcome": float(renamed[outcome].mean()),
                    }
                )
    consequence_results = pd.DataFrame(consequence_rows)
    consequence_results.to_csv(ESTABLISHED_SPLIT_CONSEQUENCE_RESULTS_CSV, index=False)
    return durability_results, split_panel, consequence_results


def build_stage_role_decomposition() -> tuple[pd.DataFrame, pd.DataFrame]:
    role_questions = pd.read_parquet(ROLE_QUESTION_PANEL)
    role_questions = role_questions[
        [
            "question_id",
            "answer_id",
            "primary_tag",
            "month_id",
            "time_index",
            "role",
            "post_chatgpt",
            "exposure_index",
            "high_tag",
            "high_post",
            "residual_queue_complexity_index_mean",
            "entrant_type",
        ]
    ].copy()

    questions = pd.read_parquet(
        QUESTION_CLOSURE_PANEL,
        columns=[
            "question_id",
            "primary_tag",
            "month_id",
            "post_chatgpt",
            "exposure_index",
            "high_tag",
            "accepted_answer_id",
            "accepted_30d",
        ],
    ).drop_duplicates("question_id")
    answers = pd.read_parquet(
        FOCAL_ANSWERS,
        columns=[
            "question_id",
            "answer_id",
            "answer_created_at",
            "score",
            "is_current_accepted_answer",
            "owner_user_id",
        ],
    )
    answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], utc=True, format="mixed")
    answers["owner_user_id"] = pd.to_numeric(answers["owner_user_id"], errors="coerce")
    answers = answers.dropna(subset=["owner_user_id"]).copy()
    answers["owner_user_id"] = answers["owner_user_id"].astype("int64")

    entrants = pd.read_csv(ENTRANT_PROFILES, usecols=["primary_tag", "answerer_user_id", "entrant_type"])
    entrants["owner_user_id"] = pd.to_numeric(entrants["answerer_user_id"], errors="coerce")
    entrants = entrants.dropna(subset=["owner_user_id"]).copy()
    entrants["owner_user_id"] = entrants["owner_user_id"].astype("int64")
    entrants = entrants[["primary_tag", "owner_user_id", "entrant_type"]].drop_duplicates()

    selection = pd.read_csv(SELECTION_PANEL)[["tag", "month_id", "residual_queue_complexity_index_mean"]].rename(
        columns={"tag": "primary_tag"}
    )

    merged = answers.merge(questions, on="question_id", how="inner")
    merged = merged.merge(entrants, on=["primary_tag", "owner_user_id"], how="left")
    merged = merged.merge(selection, on=["primary_tag", "month_id"], how="left")
    merged = merged.loc[merged["entrant_type"].notna()].copy()
    merged["month_id"] = merged["month_id"].astype(str)

    first_ids = (
        merged.sort_values(["question_id", "answer_created_at", "answer_id"])
        .drop_duplicates("question_id")[["question_id", "answer_id"]]
        .rename(columns={"answer_id": "first_answer_id"})
    )
    merged = merged.merge(first_ids, on="question_id", how="left")
    later = merged.loc[merged["answer_id"] != merged["first_answer_id"]].copy()
    later["role"] = "later_answer"

    accepted_30d = merged.loc[
        (merged["accepted_30d"] == 1) & (merged["answer_id"] == merged["accepted_answer_id"])
    ].copy()
    accepted_30d["role"] = "accepted_30d"

    month_order = {month: idx + 1 for idx, month in enumerate(sorted(merged["month_id"].dropna().unique()))}
    for frame in [later, accepted_30d]:
        frame["time_index"] = frame["month_id"].map(month_order)
        frame["high_post"] = frame["high_tag"] * frame["post_chatgpt"]

    role_extensions = pd.concat(
        [
            role_questions[
                [
                    "primary_tag",
                    "month_id",
                    "time_index",
                    "role",
                    "post_chatgpt",
                    "exposure_index",
                    "high_tag",
                    "high_post",
                    "residual_queue_complexity_index_mean",
                    "entrant_type",
                ]
            ].copy(),
            later[
                [
                    "primary_tag",
                    "month_id",
                    "time_index",
                    "role",
                    "post_chatgpt",
                    "exposure_index",
                    "high_tag",
                    "high_post",
                    "residual_queue_complexity_index_mean",
                    "entrant_type",
                ]
            ].copy(),
            accepted_30d[
                [
                    "primary_tag",
                    "month_id",
                    "time_index",
                    "role",
                    "post_chatgpt",
                    "exposure_index",
                    "high_tag",
                    "high_post",
                    "residual_queue_complexity_index_mean",
                    "entrant_type",
                ]
            ].copy(),
        ],
        ignore_index=True,
    )
    role_extensions["brand_new_platform_share"] = (role_extensions["entrant_type"] == "brand_new_platform").astype(float)
    role_extensions["low_tenure_existing_share"] = (role_extensions["entrant_type"] == "low_tenure_existing").astype(float)
    role_extensions["established_cross_tag_share"] = (role_extensions["entrant_type"] == "established_cross_tag").astype(float)

    panel = (
        role_extensions.groupby(
            [
                "primary_tag",
                "month_id",
                "time_index",
                "role",
                "post_chatgpt",
                "exposure_index",
                "high_tag",
                "high_post",
                "residual_queue_complexity_index_mean",
            ],
            as_index=False,
        )
        .agg(
            n_role_rows=("role", "size"),
            brand_new_platform_share=("brand_new_platform_share", "mean"),
            low_tenure_existing_share=("low_tenure_existing_share", "mean"),
            established_cross_tag_share=("established_cross_tag_share", "mean"),
        )
    )
    panel["exposure_post"] = panel["exposure_index"] * panel["post_chatgpt"]
    panel.to_csv(ESTABLISHED_STAGE_ROLE_PANEL_CSV, index=False)

    mean_rows = (
        panel.groupby("role", as_index=False)[
            ["brand_new_platform_share", "low_tenure_existing_share", "established_cross_tag_share", "n_role_rows"]
        ]
        .mean()
    )
    mean_rows.to_csv(ESTABLISHED_STAGE_ROLE_MEANS_CSV, index=False)

    result_rows: list[dict[str, object]] = []
    for subtype in ["brand_new_platform_share", "established_cross_tag_share"]:
        for role in ROLE_ORDER:
            role_panel = panel.loc[panel["role"] == role].copy()
            for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
                formula = (
                    f"{subtype} ~ {term} + residual_queue_complexity_index_mean + "
                    "C(primary_tag):time_index + C(primary_tag) + C(month_id)"
                )
                model = fit_wls(role_panel, formula, "n_role_rows")
                result_rows.append(
                    {
                        "test_block": "stage_role_decomposition",
                        "subtype": subtype,
                        "role": role,
                        "family": family,
                        "term": term,
                        "coef": float(model.params.get(term, np.nan)),
                        "se": float(model.bse.get(term, np.nan)),
                        "pval": float(model.pvalues.get(term, np.nan)),
                        "nobs": int(model.nobs),
                        "mean_share": float(role_panel[subtype].mean()),
                    }
                )
    results = pd.DataFrame(result_rows)
    results.to_csv(ESTABLISHED_STAGE_ROLE_RESULTS_CSV, index=False)
    return mean_rows, results


def build_local_depth_attenuation() -> tuple[pd.DataFrame, pd.DataFrame]:
    preshock = pd.read_csv(PRESHOCK_COHORTS)
    depth_tag = (
        preshock.groupby("primary_tag", as_index=False)
        .agg(
            preshock_incumbent_stock=("is_incumbent", "sum"),
            preshock_expert_stock=("is_expert", "sum"),
            preshock_local_nonexpert_stock=("is_incumbent_nonexpert", "sum"),
        )
    )

    incumbent_panel = pd.read_csv(INCUMBENT_COHORT_PANEL)
    incumbent_panel["month_id"] = incumbent_panel["month_id"].astype(str)
    preshock_months = incumbent_panel.loc[incumbent_panel["post_chatgpt"] == 0].copy()
    active_summary = (
        preshock_months.groupby("primary_tag", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "preshock_active_bench": np.average(g["share_active"], weights=g["n_user_tag_pairs"]),
                    "preshock_answerer_depth": g["n_user_tag_pairs"].mean(),
                }
            )
        )
        .reset_index(drop=True)
    )

    depth_tag = depth_tag.merge(active_summary, on="primary_tag", how="left")
    depth_tag["thin_local_depth"] = (
        -standardized(np.log1p(depth_tag["preshock_incumbent_stock"]))
        - standardized(np.log1p(depth_tag["preshock_expert_stock"]))
        - standardized(np.log1p(depth_tag["preshock_local_nonexpert_stock"]))
        - standardized(depth_tag["preshock_active_bench"])
        - standardized(np.log1p(depth_tag["preshock_answerer_depth"]))
    ) / 5.0
    depth_tag = depth_tag.sort_values("thin_local_depth", ascending=False)
    depth_tag.to_csv(ESTABLISHED_LOCAL_DEPTH_TAG_SUMMARY_CSV, index=False)

    panel = pd.read_csv(SUBTYPE_CONSEQUENCE_PANEL)
    panel = panel.merge(depth_tag[["primary_tag", "thin_local_depth"]], on="primary_tag", how="left")
    panel["established_x_thin_local_depth"] = panel["established_cross_tag_share"] * panel["thin_local_depth"]

    result_rows: list[dict[str, object]] = []
    outcomes = [
        ("any_answer_7d_rate", "any_answer_7d_denom"),
        ("first_positive_answer_latency_mean", "first_positive_answer_latency_denom"),
        ("first_answer_1d_rate_closure", "first_answer_1d_denom_closure"),
    ]
    for outcome, weight_col in outcomes:
        for label, formula in [
            (
                "baseline",
                f"{outcome} ~ high_post + established_cross_tag_share + "
                "C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            ),
            (
                "with_thin_local_interaction",
                f"{outcome} ~ high_post + established_cross_tag_share + established_x_thin_local_depth + "
                "C(primary_tag):time_index + C(primary_tag) + C(month_id)",
            ),
        ]:
            model = fit_wls(panel, formula, weight_col)
            for term in ["high_post", "established_cross_tag_share", "established_x_thin_local_depth"]:
                if term not in model.params.index:
                    continue
                result_rows.append(
                    {
                        "test_block": "local_depth_attenuation",
                        "model": label,
                        "outcome": outcome,
                        "term": term,
                        "coef": float(model.params.get(term, np.nan)),
                        "se": float(model.bse.get(term, np.nan)),
                        "pval": float(model.pvalues.get(term, np.nan)),
                        "nobs": int(model.nobs),
                    }
                )
    results = pd.DataFrame(result_rows)
    results.to_csv(ESTABLISHED_LOCAL_DEPTH_RESULTS_CSV, index=False)
    return depth_tag, results


def write_readout(
    durability_results: pd.DataFrame,
    split_consequence_results: pd.DataFrame,
    stage_means: pd.DataFrame,
    stage_results: pd.DataFrame,
    local_depth_tag: pd.DataFrame,
    local_depth_results: pd.DataFrame,
) -> None:
    def pick(df: pd.DataFrame, **conds):
        mask = pd.Series(True, index=df.index)
        for key, value in conds.items():
            mask &= df[key] == value
        sub = df.loc[mask]
        if sub.empty:
            return None
        return sub.iloc[0]

    ret365 = pick(durability_results, family="continuous", outcome="return_365d", term="exposure_post")
    one365 = pick(durability_results, family="continuous", outcome="one_shot_365d", term="exposure_post")
    ans365 = pick(durability_results, family="continuous", outcome="answers_365d", term="exposure_post")
    act365 = pick(durability_results, family="continuous", outcome="active_months_365d", term="exposure_post")

    split_any_one = pick(
        split_consequence_results,
        family="continuous",
        outcome="any_answer_7d_rate",
        term="established_one_shot_share_365d",
    )
    split_any_repeat = pick(
        split_consequence_results,
        family="continuous",
        outcome="any_answer_7d_rate",
        term="established_repeat_share_365d",
    )
    split_lat_one = pick(
        split_consequence_results,
        family="continuous",
        outcome="first_positive_answer_latency_mean",
        term="established_one_shot_share_365d",
    )
    split_lat_repeat = pick(
        split_consequence_results,
        family="continuous",
        outcome="first_positive_answer_latency_mean",
        term="established_repeat_share_365d",
    )

    stage_est_first = pick(stage_results, subtype="established_cross_tag_share", role="first_answer", family="continuous")
    stage_est_later = pick(stage_results, subtype="established_cross_tag_share", role="later_answer", family="continuous")
    stage_est_acc30 = pick(stage_results, subtype="established_cross_tag_share", role="accepted_30d", family="continuous")
    stage_brand_first = pick(stage_results, subtype="brand_new_platform_share", role="first_answer", family="continuous")
    stage_brand_later = pick(stage_results, subtype="brand_new_platform_share", role="later_answer", family="continuous")

    local_any_base = pick(local_depth_results, model="baseline", outcome="any_answer_7d_rate", term="established_cross_tag_share")
    local_any_int = pick(local_depth_results, model="with_thin_local_interaction", outcome="any_answer_7d_rate", term="established_x_thin_local_depth")
    local_lat_base = pick(local_depth_results, model="baseline", outcome="first_positive_answer_latency_mean", term="established_cross_tag_share")
    local_lat_int = pick(local_depth_results, model="with_thin_local_interaction", outcome="first_positive_answer_latency_mean", term="established_x_thin_local_depth")

    thinnest = local_depth_tag.head(3)
    deepest = local_depth_tag.tail(3)

    lines = [
        "# Established Cross-Tag Follow-On Tests",
        "",
        "## 1. Repeat Versus One-Shot Established Entrants",
        "",
        "This is the strongest of the three new tests, and it lands in the direction the current fallback mechanism needs.",
        "",
        f"- `return_365d`: coef `{ret365['coef']:.4f}`, p `{ret365['pval']:.4g}`.",
        f"- `one_shot_365d`: coef `{one365['coef']:.4f}`, p `{one365['pval']:.4g}`.",
        f"- `answers_365d`: coef `{ans365['coef']:.4f}`, p `{ans365['pval']:.4g}`.",
        f"- `active_months_365d`: coef `{act365['coef']:.4f}`, p `{act365['pval']:.4g}`.",
        "",
        "The interpretation is more nuanced than a simple thinning story. In more exposed domains after the break, established cross-tag entrants are less likely to return within a year and more likely to remain one-shot. At the same time, average answers and active months rise among the survivors, which suggests a smaller but more intensive repeating core rather than uniformly weaker embedding.",
        "",
        "The split-share consequence layer sharpens that read:",
        "",
        f"- `any_answer_7d_rate`: one-shot coef `{split_any_one['coef']:.4f}`, p `{split_any_one['pval']:.4g}`; repeat coef `{split_any_repeat['coef']:.4f}`, p `{split_any_repeat['pval']:.4g}`.",
        f"- `first_positive_answer_latency_mean`: one-shot coef `{split_lat_one['coef']:.1f}`, p `{split_lat_one['pval']:.4g}`; repeat coef `{split_lat_repeat['coef']:.1f}`, p `{split_lat_repeat['pval']:.4g}`.",
        "",
        "The clearest consequence result is that one-shot established shares carry the strongest coverage loss, while slower deeper progression loads on both one-shot and repeat shares and is actually larger on the repeat share. The safest read is therefore a polarized fallback pool: more one-shot imported responders on the front end, plus a smaller repeating core that still does not restore broad fast coverage.",
        "",
        "## 2. Stage-Role Decomposition",
        "",
        "The stage-role extension adds useful structure, but not a clean exposure-graded role shift.",
        "",
        f"- `established_cross_tag_share`, `first_answer`: coef `{stage_est_first['coef']:.4f}`, p `{stage_est_first['pval']:.4g}`.",
        f"- `established_cross_tag_share`, `later_answer`: coef `{stage_est_later['coef']:.4f}`, p `{stage_est_later['pval']:.4g}`.",
        f"- `established_cross_tag_share`, `accepted_30d`: coef `{stage_est_acc30['coef']:.4f}`, p `{stage_est_acc30['pval']:.4g}`.",
        f"- `brand_new_platform_share`, `first_answer`: coef `{stage_brand_first['coef']:.4f}`, p `{stage_brand_first['pval']:.4g}`.",
        f"- `brand_new_platform_share`, `later_answer`: coef `{stage_brand_later['coef']:.4f}`, p `{stage_brand_later['pval']:.4g}`.",
        "",
        "The safest positive read is descriptive rather than causal. Established cross-tag shares are higher in accepted-current and accepted-30d pools than in first-answer or later-answer pools, while brand-new shares are relatively higher in first-answer and especially later-answer pools. But the exposure-graded role coefficients are weak for established cross-tag entrants, so this belongs in the supporting mechanism layer rather than the promoted main result.",
        "",
        "## 3. Local-Depth Attenuation Test",
        "",
        "This is the most sensitive test, so it stays bounded.",
        "",
        f"- baseline `established_cross_tag_share` on `any_answer_7d_rate`: coef `{local_any_base['coef']:.4f}`, p `{local_any_base['pval']:.4g}`.",
        f"- interaction with `thin_local_depth` on `any_answer_7d_rate`: coef `{local_any_int['coef']:.4f}`, p `{local_any_int['pval']:.4g}`.",
        f"- baseline `established_cross_tag_share` on `first_positive_answer_latency_mean`: coef `{local_lat_base['coef']:.1f}`, p `{local_lat_base['pval']:.4g}`.",
        f"- interaction with `thin_local_depth` on `first_positive_answer_latency_mean`: coef `{local_lat_int['coef']:.1f}`, p `{local_lat_int['pval']:.4g}`.",
        "",
        "This test does not strengthen the thin-local amplification version of the mechanism. Under the current frozen pre-shock thin-local index, the adverse established-cross-tag association softens rather than intensifies in thinner-local tags. That means local depth remains useful as a discipline check, but not as promoted positive support for the fallback story.",
        "",
        "The current thinnest local-depth tags are:",
        "",
    ]
    for _, row in thinnest.iterrows():
        lines.append(f"- `{row['primary_tag']}`: `thin_local_depth = {row['thin_local_depth']:.3f}`")
    lines += [
        "",
        "The current deepest local-depth tags are:",
        "",
    ]
    for _, row in deepest.iterrows():
        lines.append(f"- `{row['primary_tag']}`: `thin_local_depth = {row['thin_local_depth']:.3f}`")
    lines += [
        "",
        "## Program-Chair Read",
        "",
        "The first test is the clear winner. It sharpens the mechanism from broad fallback dependence toward a more specific one-shot imported fallback story, with a smaller but more intensive repeating core. The second test adds supporting role structure but is not strong enough to headline. The third test currently serves as a useful negative discipline result rather than positive reinforcement.",
    ]
    READOUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    durability_results, _, split_consequence_results = build_established_repeat_panels()
    stage_means, stage_results = build_stage_role_decomposition()
    local_depth_tag, local_depth_results = build_local_depth_attenuation()
    write_readout(
        durability_results=durability_results,
        split_consequence_results=split_consequence_results,
        stage_means=stage_means,
        stage_results=stage_results,
        local_depth_tag=local_depth_tag,
        local_depth_results=local_depth_results,
    )


if __name__ == "__main__":
    main()
