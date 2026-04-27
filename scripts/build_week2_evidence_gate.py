from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper"

QUESTION_LEVEL_PARQUET = PROCESSED_DIR / "stackexchange_20251231_question_level_enriched.parquet"
QUESTION_COMPLEXITY_PARQUET = PROCESSED_DIR / "stackexchange_20251231_question_complexity_features.parquet"
FOCAL_ANSWERS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_answers.parquet"

TAG_MONTH_PANEL_CSV = PROCESSED_DIR / "who_still_answers_tag_month_entry_panel.csv"
IDENTIFICATION_PROFILE_CSV = PROCESSED_DIR / "who_still_answers_identification_profile.csv"
SMALL_SAMPLE_INFERENCE_CSV = PROCESSED_DIR / "who_still_answers_small_sample_inference.csv"
LEAVE_TWO_OUT_CSV = PROCESSED_DIR / "who_still_answers_leave_two_out.csv"
CONSTRUCT_LADDERS_CSV = PROCESSED_DIR / "who_still_answers_construct_ladders.csv"
ENTRANT_PROFILES_CSV = PROCESSED_DIR / "who_still_answers_entrant_profiles.csv"

QUESTION_MIX_BALANCE_CSV = PROCESSED_DIR / "who_still_answers_question_mix_balance.csv"
QUESTION_MIX_CONTROL_LADDER_CSV = PROCESSED_DIR / "who_still_answers_question_mix_control_ladder.csv"
QUESTION_SHAPE_BIN_RESULTS_CSV = PROCESSED_DIR / "who_still_answers_question_shape_bin_results.csv"
ENTRANT_SUBTYPE_DECOMP_CSV = PROCESSED_DIR / "who_still_answers_entrant_subtype_decomposition.csv"
WEEK2_GATE_SUMMARY_JSON = PROCESSED_DIR / "who_still_answers_week2_gate_summary.json"

WEEK2_GATE_MEMO = PAPER_DIR / "who_still_answers_week2_evidence_gate.md"

SHOCK_TS = pd.Timestamp("2022-11-30T00:00:00Z")
QUESTION_FEATURE_COLS = [
    "title_length_chars",
    "body_word_count",
    "code_block_count",
    "code_char_count",
    "error_keyword_density",
    "tag_count_full",
]


def fit_weighted(formula: str, data: pd.DataFrame, weight_col: str | None, cluster_col: str = "primary_tag"):
    if weight_col is not None:
        data = data.loc[data[weight_col].fillna(0) > 0].copy()
    y, x = patsy.dmatrices(formula, data=data, return_type="dataframe", NA_action="drop")
    frame = data.loc[y.index].copy()
    groups = frame[cluster_col]
    if weight_col is None:
        return sm.OLS(y, x).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True, "df_correction": True},
        )
    weights = frame[weight_col].astype(float)
    return sm.WLS(y, x, weights=weights).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True, "df_correction": True},
    )


def extract_term(model, term: str = "exposure_post") -> dict[str, float]:
    return {
        "coef": float(model.params.get(term, np.nan)),
        "se": float(model.bse.get(term, np.nan)),
        "pval": float(model.pvalues.get(term, np.nan)),
        "nobs": float(getattr(model, "nobs", np.nan)),
        "r2": float(getattr(model, "rsquared", np.nan)),
    }


def load_question_mix() -> tuple[pd.DataFrame, pd.DataFrame]:
    question_cols = [
        "question_id",
        "primary_tag",
        "month_id",
        "post_chatgpt",
        "exposure_index",
        "keep_single_focal",
    ]
    questions = pd.read_parquet(QUESTION_LEVEL_PARQUET, columns=question_cols)
    questions = questions.loc[(questions["keep_single_focal"] == 1) & questions["primary_tag"].notna()].copy()

    complexity_cols = ["question_id"] + QUESTION_FEATURE_COLS
    complexity = pd.read_parquet(QUESTION_COMPLEXITY_PARQUET, columns=complexity_cols)
    questions = questions.merge(complexity, on="question_id", how="left")

    for col in QUESTION_FEATURE_COLS:
        questions[col] = pd.to_numeric(questions[col], errors="coerce")
        col_std = questions[col].std(ddof=0)
        if pd.isna(col_std) or col_std == 0:
            questions[f"z_{col}"] = 0.0
        else:
            questions[f"z_{col}"] = (questions[col] - questions[col].mean()) / col_std

    z_cols = [f"z_{col}" for col in QUESTION_FEATURE_COLS]
    questions["complexity_index"] = questions[z_cols].mean(axis=1)
    questions["question_shape_bin"] = pd.qcut(
        questions["complexity_index"].rank(method="first"),
        4,
        labels=["q1", "q2", "q3", "q4"],
    )

    monthly = (
        questions.groupby(["primary_tag", "month_id"], as_index=False)
        .agg(
            n_questions=("question_id", "size"),
            exposure_index=("exposure_index", "first"),
            post_chatgpt=("post_chatgpt", "first"),
            complexity_index_mean=("complexity_index", "mean"),
            title_length_chars_mean=("title_length_chars", "mean"),
            body_word_count_mean=("body_word_count", "mean"),
            code_block_count_mean=("code_block_count", "mean"),
            code_char_count_mean=("code_char_count", "mean"),
            error_keyword_density_mean=("error_keyword_density", "mean"),
            tag_count_full_mean=("tag_count_full", "mean"),
            shape_q1_share=("question_shape_bin", lambda s: float((s == "q1").mean())),
            shape_q2_share=("question_shape_bin", lambda s: float((s == "q2").mean())),
            shape_q3_share=("question_shape_bin", lambda s: float((s == "q3").mean())),
            shape_q4_share=("question_shape_bin", lambda s: float((s == "q4").mean())),
        )
    )
    monthly["exposure_post"] = monthly["exposure_index"] * monthly["post_chatgpt"]
    return questions, monthly


def build_question_mix_balance(monthly: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "complexity_index_mean",
        "title_length_chars_mean",
        "body_word_count_mean",
        "code_block_count_mean",
        "code_char_count_mean",
        "error_keyword_density_mean",
        "tag_count_full_mean",
        "shape_q2_share",
        "shape_q3_share",
        "shape_q4_share",
    ]
    rows = []
    for outcome in outcomes:
        model = fit_weighted(
            f"{outcome} ~ exposure_post + C(primary_tag) + C(month_id)",
            monthly,
            "n_questions",
        )
        row = {"outcome": outcome}
        row.update(extract_term(model))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(QUESTION_MIX_BALANCE_CSV, index=False)
    return out


def build_question_mix_control_ladder(tag_month_panel: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    panel = tag_month_panel.merge(
        monthly[
            [
                "primary_tag",
                "month_id",
                "complexity_index_mean",
                "title_length_chars_mean",
                "body_word_count_mean",
                "code_block_count_mean",
                "code_char_count_mean",
                "error_keyword_density_mean",
                "tag_count_full_mean",
                "shape_q2_share",
                "shape_q3_share",
                "shape_q4_share",
            ]
        ],
        on=["primary_tag", "month_id"],
        how="left",
    )
    panel = panel.loc[panel["n_new_answerers"].fillna(0) > 0].copy()

    specs = [
        ("baseline", "novice_entry_share ~ exposure_post + C(primary_tag) + C(month_id)"),
        ("complexity_index", "novice_entry_share ~ exposure_post + complexity_index_mean + C(primary_tag) + C(month_id)"),
        (
            "text_shape_bundle",
            "novice_entry_share ~ exposure_post + title_length_chars_mean + body_word_count_mean + "
            "code_block_count_mean + code_char_count_mean + error_keyword_density_mean + tag_count_full_mean + "
            "C(primary_tag) + C(month_id)",
        ),
        (
            "shape_share_bundle",
            "novice_entry_share ~ exposure_post + shape_q2_share + shape_q3_share + shape_q4_share + "
            "C(primary_tag) + C(month_id)",
        ),
        (
            "full_question_mix_bundle",
            "novice_entry_share ~ exposure_post + title_length_chars_mean + body_word_count_mean + "
            "code_block_count_mean + code_char_count_mean + error_keyword_density_mean + tag_count_full_mean + "
            "shape_q2_share + shape_q3_share + shape_q4_share + C(primary_tag) + C(month_id)",
        ),
    ]

    rows = []
    baseline_coef = np.nan
    for variant, formula in specs:
        model = fit_weighted(formula, panel, "n_new_answerers")
        row = {"variant": variant, "formula": formula}
        row.update(extract_term(model))
        if variant == "baseline":
            baseline_coef = row["coef"]
        row["sign_matches_baseline"] = np.nan if pd.isna(baseline_coef) else int(np.sign(row["coef"]) == np.sign(baseline_coef))
        row["retained_share_of_baseline"] = np.nan if pd.isna(baseline_coef) or baseline_coef == 0 else float(row["coef"] / baseline_coef)
        row["absolute_retained_share"] = np.nan if pd.isna(baseline_coef) or baseline_coef == 0 else float(abs(row["coef"]) / abs(baseline_coef))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(QUESTION_MIX_CONTROL_LADDER_CSV, index=False)
    return out


def build_question_shape_bin_results(
    questions: pd.DataFrame,
    tag_month_panel: pd.DataFrame,
    entrant_profiles: pd.DataFrame,
) -> pd.DataFrame:
    tag_map = questions[["question_id", "primary_tag", "question_shape_bin"]].drop_duplicates()
    answers = pd.read_parquet(
        FOCAL_ANSWERS_PARQUET,
        columns=["answer_id", "question_id", "answer_created_at", "owner_user_id"],
    )
    answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], utc=True)
    answers = answers.rename(columns={"owner_user_id": "answerer_user_id"})
    answers = answers.merge(tag_map[["question_id", "primary_tag", "question_shape_bin"]], on="question_id", how="inner")
    first_answers = (
        answers.sort_values(["primary_tag", "answerer_user_id", "answer_created_at", "answer_id"])
        .drop_duplicates(["primary_tag", "answerer_user_id"], keep="first")
        .rename(columns={"question_id": "first_question_id", "answer_created_at": "first_tag_answer_at"})
    )
    first_answers["entry_month"] = first_answers["first_tag_answer_at"].dt.strftime("%Y-%m")

    profiles = entrant_profiles.copy()
    profiles["first_tag_answer_at"] = pd.to_datetime(profiles["first_tag_answer_at"], utc=True, format="mixed")
    first_answers = first_answers.merge(
        profiles[
            [
                "primary_tag",
                "answerer_user_id",
                "first_tag_answer_at",
                "is_novice_entrant",
                "entrant_type",
                "observed_tenure_days",
            ]
        ],
        on=["primary_tag", "answerer_user_id", "first_tag_answer_at"],
        how="left",
    )
    first_answers["is_novice_entrant"] = first_answers["is_novice_entrant"].fillna(0).astype(int)
    first_answers["entrant_type"] = first_answers["entrant_type"].fillna("pre_shock_new_answerer")

    rows = []
    for shape_bin in ["q1", "q2", "q3", "q4"]:
        temp = first_answers.loc[first_answers["question_shape_bin"] == shape_bin].copy()
        monthly = (
            temp.groupby(["primary_tag", "entry_month"], as_index=False)
            .agg(
                n_new_answerers=("answerer_user_id", "size"),
                novice_entrants=("is_novice_entrant", "sum"),
            )
            .rename(columns={"entry_month": "month_id"})
        )
        monthly["novice_entry_share"] = np.where(
            monthly["n_new_answerers"] > 0,
            monthly["novice_entrants"] / monthly["n_new_answerers"],
            np.nan,
        )
        monthly = monthly.merge(
            tag_month_panel[["primary_tag", "month_id", "exposure_post"]],
            on=["primary_tag", "month_id"],
            how="left",
        )
        model = fit_weighted(
            "novice_entry_share ~ exposure_post + C(primary_tag) + C(month_id)",
            monthly,
            "n_new_answerers",
        )
        row = {"question_shape_bin": shape_bin}
        row.update(extract_term(model))
        row["n_bin_rows"] = int(len(monthly))
        row["n_bin_new_answerers"] = float(monthly["n_new_answerers"].sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(QUESTION_SHAPE_BIN_RESULTS_CSV, index=False)
    return out


def build_entrant_subtype_decomposition(
    entrant_profiles: pd.DataFrame,
    construct_ladders: pd.DataFrame,
) -> pd.DataFrame:
    post = entrant_profiles.copy()
    grouped = (
        post.groupby("entrant_type", as_index=False)
        .agg(
            n_entrants=("answerer_user_id", "size"),
            mean_tenure_days=("observed_tenure_days", "mean"),
            mean_prior_posts=("prior_posts", "mean"),
            mean_prior_answers=("prior_answers", "mean"),
            mean_prior_questions=("prior_questions", "mean"),
        )
    )
    grouped["share_of_postshock_entrants"] = grouped["n_entrants"] / grouped["n_entrants"].sum()

    subtype_effects = construct_ladders.loc[
        construct_ladders["family"] == "entrant_subtype_effect",
        ["variant", "value", "extra_1", "extra_2"],
    ].rename(
        columns={
            "variant": "entrant_type",
            "value": "exposure_post_coef",
            "extra_1": "exposure_post_pval",
            "extra_2": "subtype_count_total",
        }
    )
    out = grouped.merge(subtype_effects, on="entrant_type", how="left")
    out.to_csv(ENTRANT_SUBTYPE_DECOMP_CSV, index=False)
    return out


def build_gate_summary(
    identification_profile: pd.DataFrame,
    small_sample: pd.DataFrame,
    leave_two_out: pd.DataFrame,
    control_ladder: pd.DataFrame,
    shape_bins: pd.DataFrame,
    subtype_decomp: pd.DataFrame,
) -> dict:
    novice_id = identification_profile.loc[identification_profile["specification"] == "novice_entry_share"].iloc[0]
    novice_inf = small_sample.loc[small_sample["specification"] == "novice_entry_share"].iloc[0]
    leave_two = leave_two_out.loc[leave_two_out["specification"] == "novice_entry_share"].copy()
    baseline = control_ladder.loc[control_ladder["variant"] == "baseline"].iloc[0]
    full_bundle = control_ladder.loc[control_ladder["variant"] == "full_question_mix_bundle"].iloc[0]
    brand_new = subtype_decomp.loc[subtype_decomp["entrant_type"] == "brand_new_platform"].iloc[0]

    positive_shape_bins = int((shape_bins["coef"] > 0).sum())
    leave_two_positive_share = float((leave_two["coef"] > 0).mean()) if not leave_two.empty else np.nan

    timing_pass = bool(
        novice_id["actual_rank_vs_pre_breaks"] <= 5 and
        float(novice_id["share_significant_pre_breaks"]) < 0.25
    )
    few_cluster_pass = bool(
        float(novice_inf["cr2_pval"]) < 0.05 and
        leave_two_positive_share == 1.0 and
        (
            (float(novice_inf["wild_cluster_bootstrap_pval"]) < 0.05 and float(novice_inf["randomization_pval"]) < 0.10) or
            (float(novice_inf["randomization_pval"]) < 0.05 and float(novice_inf["wild_cluster_bootstrap_pval"]) < 0.10)
        )
    )
    composition_pass = bool(
        np.sign(full_bundle["coef"]) == np.sign(baseline["coef"]) and
        float(full_bundle["absolute_retained_share"]) >= 0.65 and
        positive_shape_bins >= 3 and
        float(brand_new["exposure_post_coef"]) >= 0.5 * float(baseline["coef"])
    )

    return {
        "timing_gate": {
            "pass": timing_pass,
            "actual_rank_vs_pre_breaks": int(novice_id["actual_rank_vs_pre_breaks"]),
            "n_pre_breaks": int(novice_id["n_pre_breaks"]),
            "share_significant_pre_breaks": float(novice_id["share_significant_pre_breaks"]),
            "safe_language": "ChatGPT-period differential exposure" if not timing_pass else "post-ChatGPT break",
        },
        "few_cluster_gate": {
            "pass": few_cluster_pass,
            "cr2_pval": float(novice_inf["cr2_pval"]),
            "wild_cluster_bootstrap_pval": float(novice_inf["wild_cluster_bootstrap_pval"]),
            "randomization_pval": float(novice_inf["randomization_pval"]),
            "leave_two_out_positive_share": leave_two_positive_share,
        },
        "composition_gate": {
            "pass": composition_pass,
            "baseline_coef": float(baseline["coef"]),
            "full_bundle_coef": float(full_bundle["coef"]),
            "absolute_retained_share": float(full_bundle["absolute_retained_share"]),
            "positive_shape_bins": positive_shape_bins,
            "brand_new_platform_coef": float(brand_new["exposure_post_coef"]),
        },
        "week2_decision": {
            "submission_safe_now": bool(timing_pass and few_cluster_pass and composition_pass),
            "headline_survives_as_bounded_claim": bool(composition_pass),
        },
    }


def write_week2_memo(
    gate_summary: dict,
    control_ladder: pd.DataFrame,
    balance: pd.DataFrame,
    shape_bins: pd.DataFrame,
    subtype_decomp: pd.DataFrame,
) -> None:
    baseline = control_ladder.loc[control_ladder["variant"] == "baseline"].iloc[0]
    full_bundle = control_ladder.loc[control_ladder["variant"] == "full_question_mix_bundle"].iloc[0]
    brand_new = subtype_decomp.loc[subtype_decomp["entrant_type"] == "brand_new_platform"].iloc[0]
    established = subtype_decomp.loc[subtype_decomp["entrant_type"] == "established_cross_tag"].iloc[0]
    complexity_balance = balance.loc[balance["outcome"] == "complexity_index_mean"].iloc[0]

    lines = [
        "# Week 2 Evidence Gate Memo",
        "",
        "## What Week 2 Was Supposed to Do",
        "",
        "Week 2 was designed to do four things: repair timing language, surface the few-cluster inference pack, test whether the entrant result survives question-mix controls, and clarify which entrant subtype drives the effect.",
        "",
        "## Gate 1: Timing",
        "",
        f"- Status: `{'PASS' if gate_summary['timing_gate']['pass'] else 'FAIL'}`",
        f"- The preferred novice break still ranks `{gate_summary['timing_gate']['actual_rank_vs_pre_breaks']}` against `{gate_summary['timing_gate']['n_pre_breaks']}` pre-break candidates.",
        f"- Share of significant pre-break slopes: `{gate_summary['timing_gate']['share_significant_pre_breaks']:.3f}`.",
        f"- Required language: `{gate_summary['timing_gate']['safe_language']}`.",
        "",
        "Read: the paper still cannot be packaged as a clean ChatGPT shock paper. Timing repair in Week 2 means enforcing bounded language, not pretending the timing test passed.",
        "",
        "## Gate 2: Few-Cluster Inference",
        "",
        f"- Status: `{'PASS' if gate_summary['few_cluster_gate']['pass'] else 'FAIL'}`",
        f"- CR2 p-value: `{gate_summary['few_cluster_gate']['cr2_pval']:.4f}`.",
        f"- Wild-cluster bootstrap p-value: `{gate_summary['few_cluster_gate']['wild_cluster_bootstrap_pval']:.4f}`.",
        f"- Randomization p-value: `{gate_summary['few_cluster_gate']['randomization_pval']:.4f}`.",
        f"- Leave-two-out positive-sign share: `{gate_summary['few_cluster_gate']['leave_two_out_positive_share']:.3f}`.",
        "",
        "Read: the entrant-side result remains directionally resilient, but the strict Week 2 few-cluster headline rule is still not fully met because the bootstrap and randomization pair do not yet deliver the required sub-`.05` and sub-`.10` combination.",
        "",
        "## Gate 3: Composition-Confounding Discrimination",
        "",
        f"- Status: `{'PASS' if gate_summary['composition_gate']['pass'] else 'FAIL'}`",
        f"- Baseline novice-entry coefficient: `{gate_summary['composition_gate']['baseline_coef']:.4f}`.",
        f"- Full question-mix-control coefficient: `{gate_summary['composition_gate']['full_bundle_coef']:.4f}`.",
        f"- Absolute coefficient retention after question-mix controls: `{gate_summary['composition_gate']['absolute_retained_share']:.3f}`.",
        f"- Positive within-bin estimates across four question-shape bins: `{gate_summary['composition_gate']['positive_shape_bins']}/4`.",
        f"- `brand_new_platform` subtype coefficient: `{gate_summary['composition_gate']['brand_new_platform_coef']:.4f}`.",
        "",
        f"- Queue-complexity balance read: `complexity_index_mean` coefficient `{complexity_balance['coef']:.4f}` with p-value `{complexity_balance['pval']:.4f}`.",
        "",
        "Read: this gate asks whether the result disappears once we account for what kinds of questions are arriving. It passes only if the entrant effect remains same-sign and materially intact after queue-shape controls and within-bin re-estimation.",
        "",
        "## Entrant Subtype Read",
        "",
        f"- `brand_new_platform` share of post-shock entrants: `{float(brand_new['share_of_postshock_entrants']):.3f}`.",
        f"- `brand_new_platform` exposure-post coefficient: `{float(brand_new['exposure_post_coef']):.4f}` with p-value `{float(brand_new['exposure_post_pval']):.4f}`.",
        f"- `established_cross_tag` exposure-post coefficient: `{float(established['exposure_post_coef']):.4f}` with p-value `{float(established['exposure_post_pval']):.4f}`.",
        "",
        "Read: the entrant-side reallocation is not a generic rise in all new contributors. It is concentrated in brand-new-platform entrants, while established cross-tag entrants move the other way.",
        "",
        "## Week 2 Decision",
        "",
        f"- Submission-safe now: `{'YES' if gate_summary['week2_decision']['submission_safe_now'] else 'NO'}`",
        f"- Headline survives as a bounded entrant-side claim: `{'YES' if gate_summary['week2_decision']['headline_survives_as_bounded_claim'] else 'NO'}`",
        "",
        "## Operational Conclusion",
        "",
        "Week 2 should be treated as a mixed pass. It succeeds at clarifying the bounded language, surfacing the few-cluster weakness cleanly, and directly testing the question-mix alternative. It does not yet convert the paper into a submission-safe clean-shock ISR manuscript.",
        "",
        "The correct next move is:",
        "",
        "1. keep the narrow entrant-side branch",
        "2. use Week 2 controls and subtype evidence in the main paper and appendix",
        "3. do not claim a clean ChatGPT shock",
        "4. do not claim the few-cluster gate is fully passed if it is not",
        "5. carry the paper into Week 3 as a bounded entrant-side ISR candidate rather than as a solved submission package",
    ]
    WEEK2_GATE_MEMO.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    tag_month_panel = pd.read_csv(TAG_MONTH_PANEL_CSV)
    identification_profile = pd.read_csv(IDENTIFICATION_PROFILE_CSV)
    small_sample = pd.read_csv(SMALL_SAMPLE_INFERENCE_CSV)
    leave_two_out = pd.read_csv(LEAVE_TWO_OUT_CSV)
    construct_ladders = pd.read_csv(CONSTRUCT_LADDERS_CSV)
    entrant_profiles = pd.read_csv(ENTRANT_PROFILES_CSV)

    questions, monthly_mix = load_question_mix()
    balance = build_question_mix_balance(monthly_mix)
    control_ladder = build_question_mix_control_ladder(tag_month_panel, monthly_mix)
    shape_bins = build_question_shape_bin_results(questions, tag_month_panel, entrant_profiles)
    subtype_decomp = build_entrant_subtype_decomposition(entrant_profiles, construct_ladders)

    gate_summary = build_gate_summary(
        identification_profile,
        small_sample,
        leave_two_out,
        control_ladder,
        shape_bins,
        subtype_decomp,
    )
    WEEK2_GATE_SUMMARY_JSON.write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")
    write_week2_memo(gate_summary, control_ladder, balance, shape_bins, subtype_decomp)


if __name__ == "__main__":
    main()
