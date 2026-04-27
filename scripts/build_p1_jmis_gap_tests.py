from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

HARMONIZED_PANEL_CSV = PROCESSED / "p1_p2_harmonized_tag_month_panel.csv"
SUBTYPE_PANEL_CSV = PROCESSED / "p1_p2_harmonized_subtype_panel.csv"
SELECTION_PANEL_CSV = PROCESSED / "selection_composition_primary_panel.csv"
QUESTION_LEVEL_PARQUET = PROCESSED / "closure_ladder_question_level.parquet"
QUESTION_CLOSURE_PARQUET = PROCESSED / "who_still_answers_question_closure_panel.parquet"
ENTRANT_PROFILE_CSV = PROCESSED / "who_still_answers_entrant_profiles.csv"
FOCAL_ANSWERS_PARQUET = PROCESSED / "stackexchange_20251231_focal_answers.parquet"
QUESTION_COMPLEXITY_PARQUET = PROCESSED / "stackexchange_20251231_question_complexity_features.parquet"

RESIDUAL_QUEUE_RESULTS_CSV = PROCESSED / "p1_jmis_residual_queue_results.csv"
RESIDUAL_QUEUE_PANEL_CSV = PROCESSED / "p1_jmis_residual_queue_panel.csv"
SUBTYPE_CONSEQUENCE_RESULTS_CSV = PROCESSED / "p1_jmis_subtype_consequence_results.csv"
SUBTYPE_CONSEQUENCE_PANEL_CSV = PROCESSED / "p1_jmis_subtype_consequence_panel.csv"
DIRECT_VALIDATION_SAMPLE_PARQUET = PROCESSED / "entrant_first_question_validation_sample.parquet"
DIRECT_VALIDATION_RESULTS_CSV = PROCESSED / "entrant_first_question_validation_results.csv"
DIRECT_VALIDATION_SUMMARY_CSV = PROCESSED / "entrant_first_question_validation_summary.csv"
READOUT_MD = PAPER / "p1_jmis_gap_tests_readout.md"

POST_CUTOFF = pd.Timestamp("2022-11-30", tz="UTC")


def standardized(series: pd.Series) -> pd.Series:
    valid = series.astype(float)
    mean = valid.mean()
    std = valid.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(valid)), index=series.index, dtype=float)
    return (valid - mean) / std


def fit_model(
    frame: pd.DataFrame,
    formula: str,
    weight_col: str | None,
    cluster_col: str = "primary_tag",
):
    outcome = formula.split("~", 1)[0].strip()
    rhs_vars = []
    for token in formula.replace("~", "+").split("+"):
        token = token.strip()
        if token.startswith("C(") or token == "1" or ":" in token or token == "":
            continue
        rhs_vars.append(token)
    needed = [outcome, cluster_col] + rhs_vars
    if weight_col:
        needed.append(weight_col)
    model_frame = frame.dropna(subset=[col for col in needed if col in frame.columns]).copy()
    if weight_col:
        weights = model_frame[weight_col].astype(float)
        model = smf.wls(formula, data=model_frame, weights=weights).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_frame[cluster_col]},
        )
    else:
        model = smf.ols(formula, data=model_frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_frame[cluster_col]},
        )
    return model, model_frame


def extract_terms(
    model,
    term_names: list[str],
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for term in term_names:
        if term not in model.params.index:
            continue
        rows.append(
            {
                **metadata,
                "term": term,
                "coef": float(model.params[term]),
                "se": float(model.bse[term]),
                "pval": float(model.pvalues[term]),
                "nobs": int(model.nobs),
            }
        )
    return rows


def build_residual_queue_panel() -> pd.DataFrame:
    subtype_panel = pd.read_csv(SUBTYPE_PANEL_CSV)
    selection_panel = pd.read_csv(SELECTION_PANEL_CSV).rename(columns={"tag": "primary_tag"})
    keep_cols = [
        "primary_tag",
        "month_id",
        "residual_queue_complexity_index_mean",
        "body_word_count_mean",
        "tag_count_full_mean",
        "has_edit_mean",
        "complexity_index_broad_mean",
    ]
    merged = subtype_panel.merge(selection_panel[keep_cols], on=["primary_tag", "month_id"], how="inner")
    merged.to_csv(RESIDUAL_QUEUE_PANEL_CSV, index=False)
    return merged


def run_residual_queue_tests(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    complexity_outcomes = [
        "residual_queue_complexity_index_mean",
        "body_word_count_mean",
        "tag_count_full_mean",
        "has_edit_mean",
    ]
    for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
        for outcome in complexity_outcomes:
            formula = f"{outcome} ~ {term} + C(primary_tag):time_index + C(primary_tag) + C(month_id)"
            model, _ = fit_model(panel, formula, "n_questions")
            rows.extend(
                extract_terms(
                    model,
                    [term],
                    {
                        "test_family": "residual_queue_shift",
                        "family": family,
                        "model": "baseline",
                        "outcome": outcome,
                        "weight_col": "n_questions",
                        "formula": formula,
                    },
                )
            )

        for outcome, weight_col in [
            ("brand_new_platform_share", "n_new_answerers_profiles"),
            ("low_tenure_existing_share", "n_new_answerers_profiles"),
            ("established_cross_tag_share", "n_new_answerers_profiles"),
            ("first_answer_1d_rate_closure", "first_answer_1d_denom_closure"),
            ("accepted_vote_30d_rate", "accepted_vote_30d_denom"),
        ]:
            baseline_formula = f"{outcome} ~ {term} + C(primary_tag):time_index + C(primary_tag) + C(month_id)"
            bridge_formula = (
                f"{outcome} ~ {term} + residual_queue_complexity_index_mean + "
                "C(primary_tag):time_index + C(primary_tag) + C(month_id)"
            )
            for label, formula in [("baseline", baseline_formula), ("with_residual_queue", bridge_formula)]:
                model, _ = fit_model(panel, formula, weight_col)
                rows.extend(
                    extract_terms(
                        model,
                        [term, "residual_queue_complexity_index_mean"],
                        {
                            "test_family": "residual_queue_bridge",
                            "family": family,
                            "model": label,
                            "outcome": outcome,
                            "weight_col": weight_col,
                            "formula": formula,
                        },
                    )
                )
    out = pd.DataFrame(rows)
    out.to_csv(RESIDUAL_QUEUE_RESULTS_CSV, index=False)
    return out


def build_subtype_consequence_panel(common_keys: pd.DataFrame, subtype_panel: pd.DataFrame) -> pd.DataFrame:
    question_level = pd.read_parquet(
        QUESTION_LEVEL_PARQUET,
        columns=[
            "primary_tag",
            "month_id",
            "keep_single_focal",
            "any_answer_7d",
            "any_answer_7d_eligible",
            "first_positive_answer_latency_hours",
            "first_positive_answer_latency_answered_eligible",
            "accepted_cond_any_answer_30d",
            "accepted_cond_any_answer_30d_eligible",
        ],
    )
    question_level = question_level[question_level["keep_single_focal"] == 1].copy()
    question_level = question_level.merge(common_keys, on=["primary_tag", "month_id"], how="inner")
    agg = (
        question_level.groupby(["primary_tag", "month_id"], as_index=False)
        .agg(
            any_answer_7d_num=("any_answer_7d", "sum"),
            any_answer_7d_denom=("any_answer_7d_eligible", "sum"),
            first_positive_answer_latency_sum=("first_positive_answer_latency_hours", "sum"),
            first_positive_answer_latency_denom=("first_positive_answer_latency_answered_eligible", "sum"),
            accepted_cond_any_answer_30d_num=("accepted_cond_any_answer_30d", "sum"),
            accepted_cond_any_answer_30d_denom=("accepted_cond_any_answer_30d_eligible", "sum"),
        )
    )
    agg["any_answer_7d_rate"] = agg["any_answer_7d_num"] / agg["any_answer_7d_denom"].replace({0: np.nan})
    agg["first_positive_answer_latency_mean"] = (
        agg["first_positive_answer_latency_sum"] / agg["first_positive_answer_latency_denom"].replace({0: np.nan})
    )
    agg["accepted_cond_any_answer_30d_rate"] = (
        agg["accepted_cond_any_answer_30d_num"] / agg["accepted_cond_any_answer_30d_denom"].replace({0: np.nan})
    )
    keep_cols = [
        "primary_tag",
        "month_id",
        "time_index",
        "high_post",
        "exposure_post",
        "first_answer_1d_rate_closure",
        "first_answer_1d_denom_closure",
        "accepted_vote_30d_rate",
        "accepted_vote_30d_denom",
        "brand_new_platform_share",
        "low_tenure_existing_share",
        "established_cross_tag_share",
    ]
    panel = agg.merge(subtype_panel[keep_cols], on=["primary_tag", "month_id"], how="inner")
    panel.to_csv(SUBTYPE_CONSEQUENCE_PANEL_CSV, index=False)
    return panel


def run_subtype_consequence_tests(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    outcomes = [
        ("any_answer_7d_rate", "any_answer_7d_denom"),
        ("first_positive_answer_latency_mean", "first_positive_answer_latency_denom"),
        ("accepted_cond_any_answer_30d_rate", "accepted_cond_any_answer_30d_denom"),
        ("accepted_vote_30d_rate", "accepted_vote_30d_denom"),
        ("first_answer_1d_rate_closure", "first_answer_1d_denom_closure"),
    ]
    subtype_terms = ["brand_new_platform_share", "low_tenure_existing_share"]
    for family, term in [("binary", "high_post"), ("continuous", "exposure_post")]:
        for outcome, weight_col in outcomes:
            baseline_formula = f"{outcome} ~ {term} + C(primary_tag):time_index + C(primary_tag) + C(month_id)"
            subtype_formula = (
                f"{outcome} ~ {term} + brand_new_platform_share + low_tenure_existing_share + "
                "C(primary_tag):time_index + C(primary_tag) + C(month_id)"
            )
            established_formula = (
                f"{outcome} ~ {term} + established_cross_tag_share + "
                "C(primary_tag):time_index + C(primary_tag) + C(month_id)"
            )
            for label, formula in [
                ("baseline", baseline_formula),
                ("with_subtypes", subtype_formula),
                ("established_only", established_formula),
            ]:
                model, _ = fit_model(panel, formula, weight_col)
                rows.extend(
                    extract_terms(
                        model,
                        [term] + subtype_terms + ["established_cross_tag_share"],
                        {
                            "family": family,
                            "model": label,
                            "outcome": outcome,
                            "weight_col": weight_col,
                            "formula": formula,
                        },
                    )
                )
    out = pd.DataFrame(rows)
    out.to_csv(SUBTYPE_CONSEQUENCE_RESULTS_CSV, index=False)
    return out


def build_direct_validation_sample(common_keys: pd.DataFrame) -> pd.DataFrame:
    profiles = pd.read_csv(ENTRANT_PROFILE_CSV)
    profiles["first_tag_answer_at"] = pd.to_datetime(profiles["first_tag_answer_at"], format="mixed", utc=True)
    profiles["answerer_user_id"] = pd.to_numeric(profiles["answerer_user_id"], errors="coerce")
    profiles = profiles[profiles["first_tag_answer_at"] >= POST_CUTOFF].copy()
    profiles["entry_month"] = profiles["first_tag_answer_at"].dt.strftime("%Y-%m")

    answers = pd.read_parquet(
        FOCAL_ANSWERS_PARQUET,
        columns=["answer_id", "question_id", "answer_created_at", "score", "owner_user_id", "is_current_accepted_answer"],
    ).rename(columns={"owner_user_id": "answerer_user_id"})
    answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], format="mixed", utc=True)
    answers["answerer_user_id"] = pd.to_numeric(answers["answerer_user_id"], errors="coerce")

    question_panel = pd.read_parquet(
        QUESTION_CLOSURE_PARQUET,
        columns=[
            "question_id",
            "primary_tag",
            "month_id",
            "keep_single_focal",
            "exposure_index",
            "high_tag",
            "first_answer_1d",
            "accepted_30d",
        ],
    )
    question_panel = question_panel[question_panel["keep_single_focal"] == 1].copy()
    question_panel = question_panel.merge(common_keys, on=["primary_tag", "month_id"], how="inner")

    first_question_crosswalk = (
        answers.merge(question_panel, on="question_id", how="inner")
        .sort_values(["primary_tag", "answerer_user_id", "answer_created_at", "answer_id"])
        .drop_duplicates(["primary_tag", "answerer_user_id"], keep="first")
        .rename(columns={"question_id": "first_question_id", "answer_created_at": "first_tag_answer_at"})
    )
    validation = profiles.merge(
        first_question_crosswalk[
            [
                "primary_tag",
                "answerer_user_id",
                "first_tag_answer_at",
                "first_question_id",
                "score",
                "is_current_accepted_answer",
                "month_id",
                "exposure_index",
                "high_tag",
                "first_answer_1d",
                "accepted_30d",
            ]
        ],
        on=["primary_tag", "answerer_user_id", "first_tag_answer_at"],
        how="inner",
    )
    complexity = pd.read_parquet(
        QUESTION_COMPLEXITY_PARQUET,
        columns=[
            "question_id",
            "title_length_chars",
            "body_word_count",
            "code_block_count",
            "code_char_count",
            "tag_count_full",
            "comment_count",
            "has_edit",
        ],
    ).rename(columns={"question_id": "first_question_id"})
    validation = validation.merge(complexity, on="first_question_id", how="inner")

    validation["log_body_word_count"] = np.log1p(validation["body_word_count"].astype(float))
    validation["log_code_block_count"] = np.log1p(validation["code_block_count"].astype(float))
    validation["log_code_char_count"] = np.log1p(validation["code_char_count"].astype(float))
    validation["log_comment_count"] = np.log1p(validation["comment_count"].astype(float))
    components = [
        standardized(validation["title_length_chars"]),
        standardized(validation["log_body_word_count"]),
        standardized(validation["log_code_block_count"]),
        standardized(validation["log_code_char_count"]),
        standardized(validation["tag_count_full"]),
        standardized(validation["log_comment_count"]),
        standardized(validation["has_edit"]),
    ]
    validation["direct_complexity_index"] = pd.concat(components, axis=1).mean(axis=1)
    validation.to_parquet(DIRECT_VALIDATION_SAMPLE_PARQUET, index=False)
    return validation


def run_direct_validation_tests(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summary_cols = [
        "direct_complexity_index",
        "exposure_index",
        "score",
        "is_current_accepted_answer",
        "first_answer_1d",
        "accepted_30d",
        "title_length_chars",
        "body_word_count",
        "code_char_count",
        "tag_count_full",
        "has_edit",
    ]
    summary = (
        sample.groupby("entrant_type", as_index=False)
        .agg(
            n_first_questions=("first_question_id", "size"),
            **{col: (col, "mean") for col in summary_cols},
        )
    )
    summary.to_csv(DIRECT_VALIDATION_SUMMARY_CSV, index=False)

    entrant_formula = "C(entrant_type, Treatment(reference='established_cross_tag'))"
    term_names = [
        "C(entrant_type, Treatment(reference='established_cross_tag'))[T.brand_new_platform]",
        "C(entrant_type, Treatment(reference='established_cross_tag'))[T.low_tenure_existing]",
    ]
    for outcome in [
        "direct_complexity_index",
        "exposure_index",
        "score",
        "is_current_accepted_answer",
        "first_answer_1d",
        "accepted_30d",
        "title_length_chars",
        "body_word_count",
        "code_char_count",
        "tag_count_full",
        "has_edit",
    ]:
        formula = f"{outcome} ~ {entrant_formula} + C(primary_tag) + C(entry_month)"
        model, _ = fit_model(sample, formula, None)
        rows.extend(
            extract_terms(
                model,
                term_names,
                {
                    "outcome": outcome,
                    "formula": formula,
                },
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(DIRECT_VALIDATION_RESULTS_CSV, index=False)
    return out, summary


def write_readout(
    residual_results: pd.DataFrame,
    consequence_results: pd.DataFrame,
    direct_results: pd.DataFrame,
    direct_summary: pd.DataFrame,
    direct_sample: pd.DataFrame,
) -> None:
    def pick(df: pd.DataFrame, **filters):
        temp = df.copy()
        for key, value in filters.items():
            temp = temp[temp[key] == value]
        if temp.empty:
            return None
        return temp.iloc[0]

    residual_binary = pick(
        residual_results,
        test_family="residual_queue_shift",
        family="binary",
        model="baseline",
        outcome="residual_queue_complexity_index_mean",
        term="high_post",
    )
    residual_cont = pick(
        residual_results,
        test_family="residual_queue_shift",
        family="continuous",
        model="baseline",
        outcome="residual_queue_complexity_index_mean",
        term="exposure_post",
    )
    brand_queue_binary = pick(
        residual_results,
        test_family="residual_queue_bridge",
        family="binary",
        model="with_residual_queue",
        outcome="brand_new_platform_share",
        term="residual_queue_complexity_index_mean",
    )
    low_queue_binary = pick(
        residual_results,
        test_family="residual_queue_bridge",
        family="binary",
        model="with_residual_queue",
        outcome="low_tenure_existing_share",
        term="residual_queue_complexity_index_mean",
    )
    first_answer_binary = pick(
        residual_results,
        test_family="residual_queue_bridge",
        family="binary",
        model="with_residual_queue",
        outcome="first_answer_1d_rate_closure",
        term="residual_queue_complexity_index_mean",
    )
    pos_answer_binary = pick(
        consequence_results,
        family="binary",
        model="with_subtypes",
        outcome="any_answer_7d_rate",
        term="brand_new_platform_share",
    )
    latency_established_binary = pick(
        consequence_results,
        family="binary",
        model="established_only",
        outcome="first_positive_answer_latency_mean",
        term="established_cross_tag_share",
    )
    accepted_cond_binary = pick(
        consequence_results,
        family="binary",
        model="with_subtypes",
        outcome="accepted_cond_any_answer_30d_rate",
        term="brand_new_platform_share",
    )
    direct_brand_complexity = pick(
        direct_results,
        outcome="direct_complexity_index",
        term="C(entrant_type, Treatment(reference='established_cross_tag'))[T.brand_new_platform]",
    )
    direct_low_complexity = pick(
        direct_results,
        outcome="direct_complexity_index",
        term="C(entrant_type, Treatment(reference='established_cross_tag'))[T.low_tenure_existing]",
    )
    direct_brand_code = pick(
        direct_results,
        outcome="code_char_count",
        term="C(entrant_type, Treatment(reference='established_cross_tag'))[T.brand_new_platform]",
    )
    direct_brand_tags = pick(
        direct_results,
        outcome="tag_count_full",
        term="C(entrant_type, Treatment(reference='established_cross_tag'))[T.brand_new_platform]",
    )
    direct_brand_edit = pick(
        direct_results,
        outcome="has_edit",
        term="C(entrant_type, Treatment(reference='established_cross_tag'))[T.brand_new_platform]",
    )

    lines = [
        "# P1 JMIS Gap Tests Readout",
        "",
        "## Scope",
        "",
        "- Residual-queue test: merge the existing selection-composition panel into the harmonized `P1/P2` tag-month sample.",
        "- Subtype-to-consequence test: aggregate additional answer-side outcomes on the same `1147` tag-month keys and test subtype shares under the unified design.",
        "- Direct validation test: link post-shock entrant profiles to their exact first focal-tag answer question and compare what each entrant subtype actually answers.",
        "",
        "## Residual-Queue Test",
        "",
        f"- Binary residual-queue shift (`high_post -> residual_queue_complexity_index_mean`): coef `{residual_binary['coef']:.4f}`, p `{residual_binary['pval']:.4g}`." if residual_binary is not None else "- Binary residual-queue shift: unavailable.",
        f"- Continuous residual-queue shift (`exposure_post -> residual_queue_complexity_index_mean`): coef `{residual_cont['coef']:.4f}`, p `{residual_cont['pval']:.4g}`." if residual_cont is not None else "- Continuous residual-queue shift: unavailable.",
        f"- Conditional on unified controls, residual complexity predicts `brand_new_platform_share` in the binary bridge model: coef `{brand_queue_binary['coef']:.4f}`, p `{brand_queue_binary['pval']:.4g}`." if brand_queue_binary is not None else "- Residual complexity -> brand-new share bridge: unavailable.",
        f"- Conditional on unified controls, residual complexity predicts `low_tenure_existing_share` in the binary bridge model: coef `{low_queue_binary['coef']:.4f}`, p `{low_queue_binary['pval']:.4g}`." if low_queue_binary is not None else "- Residual complexity -> low-tenure share bridge: unavailable.",
        f"- Residual complexity also predicts `first_answer_1d_rate_closure`: coef `{first_answer_binary['coef']:.4f}`, p `{first_answer_binary['pval']:.4g}`." if first_answer_binary is not None else "- Residual complexity -> first-answer speed bridge: unavailable.",
        "",
        "## Subtype-to-Consequence Test",
        "",
        f"- In the binary subtype consequence model, `brand_new_platform_share` predicts `any_answer_7d_rate`: coef `{pos_answer_binary['coef']:.4f}`, p `{pos_answer_binary['pval']:.4g}`." if pos_answer_binary is not None else "- Brand-new share -> any-answer consequence: unavailable.",
        f"- In the binary established-only screen, `established_cross_tag_share` predicts `first_positive_answer_latency_mean`: coef `{latency_established_binary['coef']:.4f}`, p `{latency_established_binary['pval']:.4g}`." if latency_established_binary is not None else "- Established cross-tag share -> latency consequence: unavailable.",
        f"- In the binary subtype consequence model, `brand_new_platform_share` predicts `accepted_cond_any_answer_30d_rate`: coef `{accepted_cond_binary['coef']:.4f}`, p `{accepted_cond_binary['pval']:.4g}`." if accepted_cond_binary is not None else "- Brand-new share -> accepted-conditional consequence: unavailable.",
        "- Two-share models omit `established_cross_tag_share` as the reference composition component, while the separate established-only screen tests whether that subtype is the more credible adverse consequence candidate.",
        "",
        "## Direct Validation Test",
        "",
        f"- Post-shock entrant first-question sample size: `{len(direct_sample):,}` rows.",
        f"- `brand_new_platform` vs `established_cross_tag` on direct complexity index: coef `{direct_brand_complexity['coef']:.4f}`, p `{direct_brand_complexity['pval']:.4g}`." if direct_brand_complexity is not None else "- Brand-new vs established on complexity index: unavailable.",
        f"- `low_tenure_existing` vs `established_cross_tag` on direct complexity index: coef `{direct_low_complexity['coef']:.4f}`, p `{direct_low_complexity['pval']:.4g}`." if direct_low_complexity is not None else "- Low-tenure vs established on complexity index: unavailable.",
        f"- `brand_new_platform` vs `established_cross_tag` on `code_char_count`: coef `{direct_brand_code['coef']:.4f}`, p `{direct_brand_code['pval']:.4g}`." if direct_brand_code is not None else "- Brand-new vs established on code_char_count: unavailable.",
        f"- `brand_new_platform` vs `established_cross_tag` on `tag_count_full`: coef `{direct_brand_tags['coef']:.4f}`, p `{direct_brand_tags['pval']:.4g}`." if direct_brand_tags is not None else "- Brand-new vs established on tag_count_full: unavailable.",
        f"- `brand_new_platform` vs `established_cross_tag` on `has_edit`: coef `{direct_brand_edit['coef']:.4f}`, p `{direct_brand_edit['pval']:.4g}`." if direct_brand_edit is not None else "- Brand-new vs established on has_edit: unavailable.",
        "",
        "## Read",
        "",
        "- If residual complexity rises and helps explain subtype re-sorting, P1 gains a stronger mechanism layer than it had after harmonization alone.",
        "- If subtype shares predict answer-side consequences, P1 gains an editor-visible consequence layer rather than stopping at composition change.",
        "- If direct validation shows entrant subtypes systematically sort into different first-question profiles, P1 can defend the subtype story as behavioral rather than purely reduced-form.",
        "",
        "## Files",
        "",
        f"- [{RESIDUAL_QUEUE_RESULTS_CSV.name}]({RESIDUAL_QUEUE_RESULTS_CSV.as_posix()})",
        f"- [{SUBTYPE_CONSEQUENCE_RESULTS_CSV.name}]({SUBTYPE_CONSEQUENCE_RESULTS_CSV.as_posix()})",
        f"- [{DIRECT_VALIDATION_RESULTS_CSV.name}]({DIRECT_VALIDATION_RESULTS_CSV.as_posix()})",
        f"- [{DIRECT_VALIDATION_SUMMARY_CSV.name}]({DIRECT_VALIDATION_SUMMARY_CSV.as_posix()})",
        f"- [{DIRECT_VALIDATION_SAMPLE_PARQUET.name}]({DIRECT_VALIDATION_SAMPLE_PARQUET.as_posix()})",
    ]
    READOUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    harmonized = pd.read_csv(HARMONIZED_PANEL_CSV)
    common_keys = harmonized[["primary_tag", "month_id"]].drop_duplicates().copy()

    residual_panel = build_residual_queue_panel()
    residual_results = run_residual_queue_tests(residual_panel)

    subtype_panel = pd.read_csv(SUBTYPE_PANEL_CSV)
    consequence_panel = build_subtype_consequence_panel(common_keys, subtype_panel)
    consequence_results = run_subtype_consequence_tests(consequence_panel)

    direct_sample = build_direct_validation_sample(common_keys)
    direct_results, direct_summary = run_direct_validation_tests(direct_sample)

    write_readout(
        residual_results=residual_results,
        consequence_results=consequence_results,
        direct_results=direct_results,
        direct_summary=direct_summary,
        direct_sample=direct_sample,
    )


if __name__ == "__main__":
    main()
