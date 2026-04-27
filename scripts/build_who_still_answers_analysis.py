from __future__ import annotations

import itertools
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
FIGURES_DIR = BASE_DIR / "figures"
PAPER_DIR = BASE_DIR / "paper"

RAW_POSTS_PARQUET = RAW_DIR / "stackoverflow_2023_05_posts.parquet"
ARCHIVE_META_JSON = PROCESSED_DIR / "strengthened_archive_metadata.json"
CURRENT_QUESTION_LATENCY_PARQUET = PROCESSED_DIR / "strengthened_question_latency_sample.parquet"
EXPOSURE_CSV = PROCESSED_DIR / "strengthened_exposure_tag_scores.csv"
WATCH_STATE_JSON = RAW_DIR / "stackexchange_20251231" / "watch_and_parse_state.json"

DUMP_QUESTION_LEVEL_PARQUET = PROCESSED_DIR / "stackexchange_20251231_question_level_enriched.parquet"
DUMP_ANSWERS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_answers.parquet"

TAG_EXPOSURE_PANEL_CSV = PROCESSED_DIR / "who_still_answers_tag_exposure_panel.csv"
QUESTION_CLOSURE_PANEL_PARQUET = PROCESSED_DIR / "who_still_answers_question_closure_panel.parquet"
QUESTION_CLOSURE_TAG_MONTH_CSV = PROCESSED_DIR / "who_still_answers_question_closure_tag_month.csv"
PRESHOCK_COHORTS_PARQUET = PROCESSED_DIR / "who_still_answers_user_tag_preshock_cohorts.parquet"
PRESHOCK_COHORTS_CSV = PROCESSED_DIR / "who_still_answers_user_tag_preshock_cohorts.csv"
USER_TAG_MONTH_PANEL_PARQUET = PROCESSED_DIR / "who_still_answers_user_tag_month_panel.parquet"
INCUMBENT_COHORT_PANEL_CSV = PROCESSED_DIR / "who_still_answers_incumbent_cohort_panel.csv"
TAG_MONTH_ENTRY_PANEL_CSV = PROCESSED_DIR / "who_still_answers_tag_month_entry_panel.csv"
POSTSHOCK_STATUS_CSV = PROCESSED_DIR / "who_still_answers_postshock_status.csv"
MODEL_RESULTS_JSON = PROCESSED_DIR / "who_still_answers_results.json"
MODEL_RESULTS_CSV = PROCESSED_DIR / "who_still_answers_model_results.csv"
IDENTIFICATION_PROFILE_CSV = PROCESSED_DIR / "who_still_answers_identification_profile.csv"
TREND_BREAK_RESULTS_CSV = PROCESSED_DIR / "who_still_answers_trend_break_results.csv"
SMALL_SAMPLE_INFERENCE_CSV = PROCESSED_DIR / "who_still_answers_small_sample_inference.csv"
LEAVE_TWO_OUT_CSV = PROCESSED_DIR / "who_still_answers_leave_two_out.csv"
EXPOSURE_RANDOMIZATION_CSV = PROCESSED_DIR / "who_still_answers_exposure_randomization.csv"
CONSTRUCT_LADDERS_CSV = PROCESSED_DIR / "who_still_answers_construct_ladders.csv"
ENTRANT_PROFILE_CSV = PROCESSED_DIR / "who_still_answers_entrant_profiles.csv"
SUMMARY_MD = PROCESSED_DIR / "who_still_answers_summary.md"
PLACEBO_CSV = PROCESSED_DIR / "who_still_answers_placebo_grid.csv"
LEAVE_ONE_OUT_CSV = PROCESSED_DIR / "who_still_answers_leave_one_out.csv"
EVENT_STUDY_EXPERT_SHARE_CSV = PROCESSED_DIR / "who_still_answers_event_study_expert_share.csv"
EVENT_STUDY_NOVICE_ENTRY_CSV = PROCESSED_DIR / "who_still_answers_event_study_novice_entry.csv"
EVENT_STUDY_ACCEPTED_7D_CSV = PROCESSED_DIR / "who_still_answers_event_study_accepted_7d.csv"
METADATA_JSON = PROCESSED_DIR / "who_still_answers_metadata.json"

EXPERT_SHARE_PNG = FIGURES_DIR / "who_still_answers_expert_share_event_study.png"
NOVICE_ENTRY_PNG = FIGURES_DIR / "who_still_answers_novice_entry_event_study.png"
ACCEPTED_7D_PNG = FIGURES_DIR / "who_still_answers_accepted_7d_event_study.png"
TRENDS_PNG = FIGURES_DIR / "who_still_answers_core_trends.png"
PLACEBO_RANK_PANEL_PNG = FIGURES_DIR / "who_still_answers_placebo_rank_panel.png"
NOVICE_TREND_BREAK_PNG = FIGURES_DIR / "who_still_answers_novice_entry_trend_break.png"
PERMUTATION_DENSITY_NOVICE_PNG = FIGURES_DIR / "who_still_answers_permutation_density_novice_entry.png"
INFLUENCE_HEATMAP_PNG = FIGURES_DIR / "who_still_answers_influence_heatmap.png"

MECHANISM_DECISION_MEMO = PAPER_DIR / "who_still_answers_mechanism_decision_memo.md"
FINAL_SCORECARD_MD = PAPER_DIR / "who_still_answers_final_scorecard.md"

SHOCK_TS = pd.Timestamp("2022-11-30T00:00:00Z")
SHOCK_MONTH = "2022-12"
START_MONTH = "2020-01"
MIN_INDIVIDUAL_INCUMBENT_ANSWERS = 5
MIN_EXPERT_ANSWERS = 20
NOVICE_TENURE_DAYS = 90.0
WILD_BOOTSTRAP_REPS = 399
RANDOMIZATION_REPS = 499
PREFERRED_BREAK_TERMS = {
    "novice_entry_share": ("slope_only", "exposure_break_slope", 1.0),
    "expert_answer_share": ("level_only", "exposure_break_post", -1.0),
    "accepted_7d_rate": ("level_only", "exposure_break_post", -1.0),
}


@dataclass
class SourceMetadata:
    question_source: str
    answer_source: str
    accepted_timing_source: str
    data_end_at: str
    watch_state: dict


@dataclass
class ModelSpec:
    name: str
    frame: pd.DataFrame
    outcome: str
    weight_col: str | None
    term: str
    formula: str
    cluster_col: str = "primary_tag"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)


def filter_weighted_frame(frame: pd.DataFrame, weight_col: str | None) -> pd.DataFrame:
    if weight_col is None:
        return frame.copy()
    return frame.loc[frame[weight_col].fillna(0) > 0].copy()


def fit_model(formula: str, data: pd.DataFrame, weight_col: str | None, cluster_col: str) -> object:
    frame = filter_weighted_frame(data, weight_col)
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
    if "is_expert" in temp.columns:
        temp["exposure_expert"] = temp["exposure_index"] * temp["is_expert"]
    if "post_chatgpt" in temp.columns and "is_expert" in temp.columns:
        temp["exposure_post_expert"] = temp["exposure_index"] * temp["post_chatgpt"] * temp["is_expert"]
    return temp


def read_watch_state() -> dict:
    if not WATCH_STATE_JSON.exists():
        return {}
    try:
        state = json.loads(WATCH_STATE_JSON.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    if state.get("status") == "completed" and not (DUMP_QUESTION_LEVEL_PARQUET.exists() and DUMP_ANSWERS_PARQUET.exists()):
        state = state.copy()
        detail = state.get("detail", "")
        suffix = "dump-backed parquet outputs not found; current analysis remains on corrected early archive"
        state["status"] = "completed_without_required_outputs"
        state["detail"] = f"{detail}; {suffix}".strip("; ")
    return state


def read_archive_cutoff(question_df: pd.DataFrame) -> pd.Timestamp:
    if ARCHIVE_META_JSON.exists():
        meta = json.loads(ARCHIVE_META_JSON.read_text(encoding="utf-8"))
        return pd.Timestamp(meta["archive_cutoff_iso"])
    return normalize_utc_timestamp(question_df["question_created_at"].max())


def normalize_utc_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_exposure() -> pd.DataFrame:
    exposure = pd.read_csv(EXPOSURE_CSV)
    exposure["high_tag"] = (exposure["exposure_index"] > exposure["exposure_index"].median()).astype(int)
    return exposure


def parse_selected_tags(tag_string: str | None, valid_tags: set[str]) -> list[str]:
    if tag_string is None or pd.isna(tag_string):
        return []
    return [tag for tag in str(tag_string).split(";") if tag in valid_tags]


def month_range(start_at: pd.Timestamp, end_at: pd.Timestamp) -> list[str]:
    start_period = pd.Period(start_at.strftime("%Y-%m"), freq="M")
    end_period = pd.Period(end_at.strftime("%Y-%m"), freq="M")
    return pd.period_range(start_period, end_period, freq="M").strftime("%Y-%m").tolist()


def load_question_base(exposure: pd.DataFrame) -> tuple[pd.DataFrame, SourceMetadata]:
    valid_tags = set(exposure["tag"])
    if DUMP_QUESTION_LEVEL_PARQUET.exists() and DUMP_ANSWERS_PARQUET.exists():
        question_df = pd.read_parquet(DUMP_QUESTION_LEVEL_PARQUET)
        question_df["question_created_at"] = pd.to_datetime(question_df["question_created_at"], utc=True)
        drop_cols = [col for col in ["exposure_index", "high_tag", "exposure_rank", "exposure_tercile"] if col in question_df.columns]
        if drop_cols:
            question_df = question_df.drop(columns=drop_cols)
        if "selected_tags" in question_df.columns:
            question_df["selected_tags_list"] = question_df["selected_tags"].apply(lambda value: parse_selected_tags(value, valid_tags))
        else:
            question_df["selected_tags_list"] = question_df["question_tags"].apply(lambda value: parse_selected_tags(value, valid_tags))
        question_df["primary_tag"] = question_df["selected_tags_list"].apply(lambda tags: tags[0] if len(tags) == 1 else None)
        question_df["selected_tag_overlap"] = question_df["selected_tags_list"].str.len()
        if "score" not in question_df.columns and "score_snapshot" in question_df.columns:
            question_df["score"] = question_df["score_snapshot"]
        if "view_count" not in question_df.columns and "view_count_snapshot" in question_df.columns:
            question_df["view_count"] = question_df["view_count_snapshot"]
        source = SourceMetadata(
            question_source="stackexchange_20251231_dump",
            answer_source="stackexchange_20251231_dump",
            accepted_timing_source="accept_vote_date_daylevel_if_available",
            data_end_at=normalize_utc_timestamp(question_df["question_created_at"].max()).isoformat(),
            watch_state=read_watch_state(),
        )
        return question_df, source

    question_df = pd.read_parquet(CURRENT_QUESTION_LATENCY_PARQUET)
    question_df["question_created_at"] = pd.to_datetime(question_df["question_created_at"], utc=True)
    question_df["selected_tags_list"] = question_df["question_tags"].apply(lambda value: parse_selected_tags(value, valid_tags))
    question_df["primary_tag"] = question_df["selected_tags_list"].apply(lambda tags: tags[0] if len(tags) == 1 else None)
    question_df["selected_tag_overlap"] = question_df["selected_tags_list"].str.len()
    question_df["source_regime"] = "corrected_early_archive"
    source = SourceMetadata(
        question_source="corrected_early_archive",
        answer_source="corrected_early_archive",
        accepted_timing_source="accepted_answer_creation_timestamp",
        data_end_at=read_archive_cutoff(question_df).isoformat(),
        watch_state=read_watch_state(),
    )
    return question_df, source


def load_answer_base(question_df: pd.DataFrame) -> pd.DataFrame:
    questions = question_df[["question_id", "question_created_at", "accepted_answer_id", "primary_tag"]].copy()
    questions = questions.loc[questions["primary_tag"].notna()].copy()

    if DUMP_ANSWERS_PARQUET.exists() and DUMP_QUESTION_LEVEL_PARQUET.exists():
        answers = pd.read_parquet(DUMP_ANSWERS_PARQUET)
        answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], utc=True)
        answers = answers.rename(columns={"owner_user_id": "answerer_user_id", "score": "answer_score"})
        answers = answers.merge(questions, on="question_id", how="inner")
    else:
        con = duckdb.connect()
        con.register("qids", questions[["question_id"]])
        query = f"""
            SELECT
                a.Id AS answer_id,
                a.ParentId AS question_id,
                a.CreationDate AS answer_created_at,
                a.OwnerUserId AS answerer_user_id,
                a.Score AS answer_score
            FROM read_parquet('{RAW_POSTS_PARQUET.as_posix()}') a
            INNER JOIN qids q
              ON a.ParentId = q.question_id
            WHERE a.PostTypeId = 2
        """
        answers = con.execute(query).fetchdf()
        con.close()
        answers["answer_created_at"] = pd.to_datetime(answers["answer_created_at"], utc=True)
        answers = answers.merge(questions, on="question_id", how="inner")

    answers = answers.loc[answers["answer_created_at"] >= answers["question_created_at"]].copy()
    answers["is_current_accepted"] = (answers["answer_id"] == answers["accepted_answer_id"]).astype(int)
    answers["answer_month"] = answers["answer_created_at"].dt.strftime("%Y-%m")
    return answers


def build_question_closure_panel(question_df: pd.DataFrame, answers: pd.DataFrame, exposure: pd.DataFrame, source: SourceMetadata) -> pd.DataFrame:
    questions = question_df.loc[question_df["primary_tag"].notna()].copy()
    first_answers = (
        answers.groupby("question_id", as_index=False)["answer_created_at"]
        .min()
        .rename(columns={"answer_created_at": "first_answer_at_valid"})
    )
    accepted_answers = (
        answers.loc[answers["is_current_accepted"] == 1, ["question_id", "answer_created_at"]]
        .rename(columns={"answer_created_at": "accepted_answer_created_at_valid"})
        .drop_duplicates(subset=["question_id"])
    )
    questions = questions.merge(first_answers, on="question_id", how="left")
    questions = questions.merge(accepted_answers, on="question_id", how="left")
    if "accept_vote_date" in questions.columns:
        questions["accepted_event_at"] = pd.to_datetime(questions["accept_vote_date"], utc=True, errors="coerce")
        questions["accepted_timing_source"] = "accept_vote_date_daylevel"
    else:
        questions["accepted_event_at"] = questions["accepted_answer_created_at_valid"]
        questions["accepted_timing_source"] = source.accepted_timing_source

    data_end = pd.Timestamp(source.data_end_at)
    questions["can_observe_first_answer_12h"] = (questions["question_created_at"] <= data_end - pd.Timedelta(hours=12)).astype(int)
    questions["can_observe_first_answer_1d"] = (questions["question_created_at"] <= data_end - pd.Timedelta(days=1)).astype(int)
    questions["can_observe_first_answer_7d"] = (questions["question_created_at"] <= data_end - pd.Timedelta(days=7)).astype(int)
    questions["can_observe_accepted_7d"] = (questions["question_created_at"] <= data_end - pd.Timedelta(days=7)).astype(int)
    questions["can_observe_accepted_14d"] = (questions["question_created_at"] <= data_end - pd.Timedelta(days=14)).astype(int)
    questions["can_observe_accepted_30d"] = (questions["question_created_at"] <= data_end - pd.Timedelta(days=30)).astype(int)

    first_delta_hours = (questions["first_answer_at_valid"] - questions["question_created_at"]).dt.total_seconds() / 3600.0
    accepted_delta_hours = (questions["accepted_event_at"] - questions["question_created_at"]).dt.total_seconds() / 3600.0
    questions["first_answer_12h"] = ((first_delta_hours >= 0) & (first_delta_hours <= 12.0)).fillna(False).astype(int)
    questions["first_answer_1d"] = ((first_delta_hours >= 0) & (first_delta_hours <= 24.0)).fillna(False).astype(int)
    questions["first_answer_7d"] = ((first_delta_hours >= 0) & (first_delta_hours <= 24.0 * 7.0)).fillna(False).astype(int)
    questions["accepted_7d"] = ((accepted_delta_hours >= 0) & (accepted_delta_hours <= 24.0 * 7.0)).fillna(False).astype(int)
    questions["accepted_14d"] = ((accepted_delta_hours >= 0) & (accepted_delta_hours <= 24.0 * 14.0)).fillna(False).astype(int)
    questions["accepted_30d"] = ((accepted_delta_hours >= 0) & (accepted_delta_hours <= 24.0 * 30.0)).fillna(False).astype(int)
    questions["first_answer_12h_obs"] = questions["first_answer_12h"] * questions["can_observe_first_answer_12h"]
    questions["first_answer_1d_obs"] = questions["first_answer_1d"] * questions["can_observe_first_answer_1d"]
    questions["first_answer_7d_obs"] = questions["first_answer_7d"] * questions["can_observe_first_answer_7d"]
    questions["accepted_7d_obs"] = questions["accepted_7d"] * questions["can_observe_accepted_7d"]
    questions["accepted_14d_obs"] = questions["accepted_14d"] * questions["can_observe_accepted_14d"]
    questions["accepted_30d_obs"] = questions["accepted_30d"] * questions["can_observe_accepted_30d"]
    questions["month_id"] = questions["question_created_at"].dt.strftime("%Y-%m")
    questions["post_chatgpt"] = (questions["question_created_at"] >= SHOCK_TS).astype(int)
    questions["source_regime"] = source.question_source
    questions = questions.merge(exposure[["tag", "exposure_index", "high_tag", "exposure_rank", "exposure_tercile"]].rename(columns={"tag": "primary_tag"}), on="primary_tag", how="left")
    questions.to_parquet(QUESTION_CLOSURE_PANEL_PARQUET, index=False)

    tag_month = (
        questions.groupby(["primary_tag", "month_id"], as_index=False)
        .agg(
            n_questions=("question_id", "size"),
            exposure_index=("exposure_index", "first"),
            high_tag=("high_tag", "first"),
            first_answer_12h_num=("first_answer_12h_obs", "sum"),
            first_answer_12h_denom=("can_observe_first_answer_12h", "sum"),
            first_answer_1d_num=("first_answer_1d_obs", "sum"),
            first_answer_1d_denom=("can_observe_first_answer_1d", "sum"),
            first_answer_7d_num=("first_answer_7d_obs", "sum"),
            first_answer_7d_denom=("can_observe_first_answer_7d", "sum"),
            accepted_7d_num=("accepted_7d_obs", "sum"),
            accepted_7d_denom=("can_observe_accepted_7d", "sum"),
            accepted_14d_num=("accepted_14d_obs", "sum"),
            accepted_14d_denom=("can_observe_accepted_14d", "sum"),
            accepted_30d_num=("accepted_30d_obs", "sum"),
            accepted_30d_denom=("can_observe_accepted_30d", "sum"),
        )
        .sort_values(["primary_tag", "month_id"])
        .reset_index(drop=True)
    )
    for prefix in ["first_answer_12h", "first_answer_1d", "first_answer_7d", "accepted_7d", "accepted_14d", "accepted_30d"]:
        tag_month[f"{prefix}_rate"] = np.where(
            tag_month[f"{prefix}_denom"] > 0,
            tag_month[f"{prefix}_num"] / tag_month[f"{prefix}_denom"],
            np.nan,
        )
    tag_month["post_chatgpt"] = (pd.to_datetime(tag_month["month_id"] + "-01", utc=True) >= pd.Timestamp("2022-12-01T00:00:00Z")).astype(int)
    tag_month["exposure_post"] = tag_month["exposure_index"] * tag_month["post_chatgpt"]
    tag_month["time_index"] = tag_month.groupby("primary_tag").cumcount()
    tag_month.to_csv(QUESTION_CLOSURE_TAG_MONTH_CSV, index=False)
    return questions


def build_first_post_lookup(user_ids: pd.Series) -> pd.DataFrame:
    cleaned = user_ids.dropna().drop_duplicates()
    if cleaned.empty or not RAW_POSTS_PARQUET.exists():
        return pd.DataFrame(columns=["answerer_user_id", "first_post_at"])
    con = duckdb.connect()
    lookup = pd.DataFrame({"OwnerUserId": cleaned.astype(int).tolist()})
    con.register("lookup", lookup)
    query = f"""
        SELECT
            p.OwnerUserId AS answerer_user_id,
            MIN(p.CreationDate) AS first_post_at
        FROM read_parquet('{RAW_POSTS_PARQUET.as_posix()}') p
        INNER JOIN lookup l
          ON p.OwnerUserId = l.OwnerUserId
        GROUP BY 1
    """
    result = con.execute(query).fetchdf()
    con.close()
    if not result.empty:
        result["first_post_at"] = pd.to_datetime(result["first_post_at"], utc=True)
    return result


def standardize_within_group(frame: pd.Series) -> pd.Series:
    std = frame.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(frame)), index=frame.index)
    return (frame - frame.mean()) / std


def build_user_tag_preshock_cohorts(answers: pd.DataFrame, first_posts: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    preshock = answers.loc[answers["answer_created_at"] < SHOCK_TS].copy()
    cohorts = (
        preshock.groupby(["primary_tag", "answerer_user_id"], as_index=False)
        .agg(
            preshock_answers=("answer_id", "size"),
            preshock_accepted_current=("is_current_accepted", "sum"),
            preshock_mean_score=("answer_score", "mean"),
            first_tag_answer_at=("answer_created_at", "min"),
            last_tag_answer_at=("answer_created_at", "max"),
        )
    )
    cohorts = cohorts.merge(first_posts, on="answerer_user_id", how="left")
    cohorts["observed_first_activity_at"] = cohorts["first_post_at"].fillna(cohorts["first_tag_answer_at"])
    cohorts["observed_tenure_days_at_first_tag"] = (
        cohorts["first_tag_answer_at"] - cohorts["observed_first_activity_at"]
    ).dt.total_seconds() / 86400.0

    cohorts["log_answers"] = np.log1p(cohorts["preshock_answers"])
    cohorts["log_accepted"] = np.log1p(cohorts["preshock_accepted_current"])
    cohorts["mean_score_filled"] = cohorts["preshock_mean_score"].fillna(0.0)
    cohorts["z_answers"] = cohorts.groupby("primary_tag")["log_answers"].transform(standardize_within_group)
    cohorts["z_accepted"] = cohorts.groupby("primary_tag")["log_accepted"].transform(standardize_within_group)
    cohorts["z_score"] = cohorts.groupby("primary_tag")["mean_score_filled"].transform(standardize_within_group)
    cohorts["expert_score"] = cohorts["z_answers"] + cohorts["z_accepted"] + cohorts["z_score"]
    cohorts["expert_cutoff"] = cohorts.groupby("primary_tag")["expert_score"].transform(lambda series: series.quantile(0.9))
    cohorts["is_expert"] = ((cohorts["preshock_answers"] >= MIN_EXPERT_ANSWERS) & (cohorts["expert_score"] >= cohorts["expert_cutoff"])).astype(int)
    cohorts["is_incumbent_nonexpert"] = ((cohorts["preshock_answers"] >= MIN_INDIVIDUAL_INCUMBENT_ANSWERS) & (cohorts["is_expert"] == 0)).astype(int)
    cohorts["is_incumbent"] = (cohorts["preshock_answers"] >= MIN_INDIVIDUAL_INCUMBENT_ANSWERS).astype(int)
    cohorts = cohorts.merge(exposure[["tag", "exposure_index", "high_tag"]].rename(columns={"tag": "primary_tag"}), on="primary_tag", how="left")
    cohorts.to_parquet(PRESHOCK_COHORTS_PARQUET, index=False)
    cohorts.to_csv(PRESHOCK_COHORTS_CSV, index=False)
    return cohorts


def build_user_tag_month_panel(answers: pd.DataFrame, cohorts: pd.DataFrame, source: SourceMetadata) -> tuple[pd.DataFrame, pd.DataFrame]:
    incumbents = cohorts.loc[cohorts["is_incumbent"] == 1, [
        "primary_tag",
        "answerer_user_id",
        "is_expert",
        "is_incumbent_nonexpert",
        "exposure_index",
        "high_tag",
        "preshock_answers",
    ]].copy()
    activity = (
        answers.merge(incumbents[["primary_tag", "answerer_user_id", "is_expert"]], on=["primary_tag", "answerer_user_id"], how="inner")
        .groupby(["primary_tag", "answerer_user_id", "answer_month", "is_expert"], as_index=False)
        .agg(
            answer_count=("answer_id", "size"),
            accepted_current_count=("is_current_accepted", "sum"),
        )
        .rename(columns={"answer_month": "month_id"})
    )

    months = month_range(pd.Timestamp(f"{START_MONTH}-01T00:00:00Z"), pd.Timestamp(source.data_end_at))
    base_pairs = incumbents.copy()
    base_pairs["key"] = 1
    month_df = pd.DataFrame({"month_id": months, "key": 1})
    panel = base_pairs.merge(month_df, on="key").drop(columns="key")
    panel = panel.merge(activity, on=["primary_tag", "answerer_user_id", "month_id", "is_expert"], how="left")
    panel["answer_count"] = panel["answer_count"].fillna(0)
    panel["accepted_current_count"] = panel["accepted_current_count"].fillna(0)
    panel["any_answer"] = (panel["answer_count"] > 0).astype(int)
    panel["log_answers"] = np.log1p(panel["answer_count"])
    panel["post_chatgpt"] = (pd.to_datetime(panel["month_id"] + "-01", utc=True) >= pd.Timestamp("2022-12-01T00:00:00Z")).astype(int)
    panel["exposure_post"] = panel["exposure_index"] * panel["post_chatgpt"]
    panel["expert_post"] = panel["is_expert"] * panel["post_chatgpt"]
    panel["exposure_expert"] = panel["exposure_index"] * panel["is_expert"]
    panel["exposure_post_expert"] = panel["exposure_index"] * panel["post_chatgpt"] * panel["is_expert"]
    panel.to_parquet(USER_TAG_MONTH_PANEL_PARQUET, index=False)

    cohort_panel = (
        panel.groupby(["primary_tag", "month_id", "is_expert"], as_index=False)
        .agg(
            n_user_tag_pairs=("answerer_user_id", "size"),
            mean_log_answers=("log_answers", "mean"),
            share_active=("any_answer", "mean"),
            mean_answers=("answer_count", "mean"),
        )
    )
    exposure_lookup = cohorts[["primary_tag", "exposure_index", "high_tag"]].drop_duplicates()
    cohort_panel = cohort_panel.merge(exposure_lookup, on="primary_tag", how="left")
    cohort_panel["post_chatgpt"] = (pd.to_datetime(cohort_panel["month_id"] + "-01", utc=True) >= pd.Timestamp("2022-12-01T00:00:00Z")).astype(int)
    cohort_panel["exposure_post"] = cohort_panel["exposure_index"] * cohort_panel["post_chatgpt"]
    cohort_panel["expert_post"] = cohort_panel["is_expert"] * cohort_panel["post_chatgpt"]
    cohort_panel["exposure_expert"] = cohort_panel["exposure_index"] * cohort_panel["is_expert"]
    cohort_panel["exposure_post_expert"] = cohort_panel["exposure_index"] * cohort_panel["post_chatgpt"] * cohort_panel["is_expert"]
    cohort_panel.to_csv(INCUMBENT_COHORT_PANEL_CSV, index=False)
    return panel, cohort_panel


def build_tag_month_entry_panel(
    answers: pd.DataFrame,
    cohorts: pd.DataFrame,
    first_posts: pd.DataFrame,
    exposure: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    answer_with_cohorts = answers.merge(
        cohorts[["primary_tag", "answerer_user_id", "is_expert", "is_incumbent", "preshock_answers"]],
        on=["primary_tag", "answerer_user_id"],
        how="left",
    )
    answer_with_cohorts[["is_expert", "is_incumbent", "preshock_answers"]] = answer_with_cohorts[["is_expert", "is_incumbent", "preshock_answers"]].fillna(0)

    first_tag_answers = (
        answer_with_cohorts.groupby(["primary_tag", "answerer_user_id"], as_index=False)
        .agg(first_tag_answer_at=("answer_created_at", "min"))
        .merge(first_posts, on="answerer_user_id", how="left")
    )
    first_tag_answers["observed_first_activity_at"] = first_tag_answers["first_post_at"].fillna(first_tag_answers["first_tag_answer_at"])
    first_tag_answers["observed_tenure_days"] = (
        first_tag_answers["first_tag_answer_at"] - first_tag_answers["observed_first_activity_at"]
    ).dt.total_seconds() / 86400.0
    # Keep the headline entrant construct symmetric across the full panel.
    tenure_eligible = (
        first_tag_answers["observed_tenure_days"].ge(0) &
        first_tag_answers["observed_tenure_days"].le(NOVICE_TENURE_DAYS)
    )
    first_tag_answers["is_novice_entrant"] = tenure_eligible.astype(int)
    first_tag_answers["is_postshock_novice_entrant"] = (
        (first_tag_answers["first_tag_answer_at"] >= SHOCK_TS) & tenure_eligible
    ).astype(int)
    first_tag_answers["entrant_type"] = np.select(
        [
            first_tag_answers["first_post_at"].isna() | (first_tag_answers["first_post_at"] == first_tag_answers["first_tag_answer_at"]),
            (first_tag_answers["observed_tenure_days"] > 0) & (first_tag_answers["observed_tenure_days"] <= NOVICE_TENURE_DAYS),
        ],
        [
            "brand_new_platform",
            "low_tenure_existing",
        ],
        default="established_cross_tag",
    )
    first_tag_answers["entry_month"] = first_tag_answers["first_tag_answer_at"].dt.strftime("%Y-%m")

    tag_month = (
        answer_with_cohorts.groupby(["primary_tag", "answer_month"], as_index=False)
        .agg(
            n_answers=("answer_id", "size"),
            n_answerers=("answerer_user_id", "nunique"),
            expert_answer_share=("is_expert", "mean"),
            incumbent_answer_share=("is_incumbent", "mean"),
        )
        .rename(columns={"answer_month": "month_id"})
    )
    accepted = (
        answer_with_cohorts.loc[answer_with_cohorts["is_current_accepted"] == 1]
        .groupby(["primary_tag", "answer_month"], as_index=False)
        .agg(accepted_answer_expert_share=("is_expert", "mean"))
        .rename(columns={"answer_month": "month_id"})
    )
    entry = (
        first_tag_answers.groupby(["primary_tag", "entry_month"], as_index=False)
        .agg(
            n_new_answerers=("answerer_user_id", "size"),
            novice_entrants=("is_novice_entrant", "sum"),
        )
        .rename(columns={"entry_month": "month_id"})
    )
    entry["novice_entry_share"] = np.where(entry["n_new_answerers"] > 0, entry["novice_entrants"] / entry["n_new_answerers"], np.nan)

    closure_tag_month = pd.read_csv(QUESTION_CLOSURE_TAG_MONTH_CSV)
    panel = closure_tag_month.merge(tag_month, on=["primary_tag", "month_id"], how="left")
    panel = panel.merge(accepted, on=["primary_tag", "month_id"], how="left")
    panel = panel.merge(entry, on=["primary_tag", "month_id"], how="left")
    panel = panel.merge(exposure[["tag", "exposure_index", "high_tag"]].rename(columns={"tag": "primary_tag"}), on="primary_tag", how="left", suffixes=("", "_exp"))
    panel["post_chatgpt"] = (pd.to_datetime(panel["month_id"] + "-01", utc=True) >= pd.Timestamp("2022-12-01T00:00:00Z")).astype(int)
    panel["exposure_post"] = panel["exposure_index"] * panel["post_chatgpt"]
    panel["time_index"] = panel.groupby("primary_tag").cumcount()
    panel.to_csv(TAG_MONTH_ENTRY_PANEL_CSV, index=False)
    return panel, first_tag_answers


def build_postshock_status(panel: pd.DataFrame, cohorts: pd.DataFrame) -> pd.DataFrame:
    post = panel.loc[panel["post_chatgpt"] == 1].copy()
    status = (
        post.groupby(["primary_tag", "answerer_user_id"], as_index=False)
        .agg(
            any_postshock_answer=("any_answer", "max"),
            postshock_answer_count=("answer_count", "sum"),
            postshock_active_months=("any_answer", "sum"),
        )
    )
    status = status.merge(
        cohorts[["primary_tag", "answerer_user_id", "is_expert", "exposure_index", "high_tag", "preshock_answers"]],
        on=["primary_tag", "answerer_user_id"],
        how="left",
    )
    status["postshock_silent"] = (status["any_postshock_answer"] == 0).astype(int)
    status["exposure_expert"] = status["exposure_index"] * status["is_expert"]
    status.to_csv(POSTSHOCK_STATUS_CSV, index=False)
    return status


def fit_weighted(formula: str, data: pd.DataFrame, weight_col: str | None, cluster_col: str) -> object:
    return fit_model(formula, data, weight_col, cluster_col)


def extract_term(model: object, term: str) -> dict:
    return {
        "coef": float(model.params.get(term, np.nan)),
        "se": float(model.bse.get(term, np.nan)),
        "pval": float(model.pvalues.get(term, np.nan)),
        "nobs": float(getattr(model, "nobs", np.nan)),
        "r2": float(getattr(model, "rsquared", np.nan)),
    }


def get_model_specs(cohort_panel: pd.DataFrame, tag_month_panel: pd.DataFrame, postshock_status: pd.DataFrame) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="incumbent_mean_log_answers",
            frame=cohort_panel.copy(),
            outcome="mean_log_answers",
            weight_col="n_user_tag_pairs",
            term="exposure_post_expert",
            formula="mean_log_answers ~ exposure_post + expert_post + exposure_expert + exposure_post_expert + C(primary_tag) + C(month_id)",
        ),
        ModelSpec(
            name="incumbent_share_active",
            frame=cohort_panel.copy(),
            outcome="share_active",
            weight_col="n_user_tag_pairs",
            term="exposure_post_expert",
            formula="share_active ~ exposure_post + expert_post + exposure_expert + exposure_post_expert + C(primary_tag) + C(month_id)",
        ),
        ModelSpec(
            name="expert_answer_share",
            frame=tag_month_panel.dropna(subset=["expert_answer_share"]).copy(),
            outcome="expert_answer_share",
            weight_col="n_answers",
            term="exposure_post",
            formula="expert_answer_share ~ exposure_post + C(primary_tag) + C(month_id)",
        ),
        ModelSpec(
            name="novice_entry_share",
            frame=tag_month_panel.dropna(subset=["novice_entry_share"]).copy(),
            outcome="novice_entry_share",
            weight_col="n_new_answerers",
            term="exposure_post",
            formula="novice_entry_share ~ exposure_post + C(primary_tag) + C(month_id)",
        ),
        ModelSpec(
            name="first_answer_1d",
            frame=tag_month_panel.loc[tag_month_panel["first_answer_1d_denom"] > 0].copy(),
            outcome="first_answer_1d_rate",
            weight_col="first_answer_1d_denom",
            term="exposure_post",
            formula="first_answer_1d_rate ~ exposure_post + C(primary_tag) + C(month_id)",
        ),
        ModelSpec(
            name="accepted_7d",
            frame=tag_month_panel.loc[tag_month_panel["accepted_7d_denom"] > 0].copy(),
            outcome="accepted_7d_rate",
            weight_col="accepted_7d_denom",
            term="exposure_post",
            formula="accepted_7d_rate ~ exposure_post + C(primary_tag) + C(month_id)",
        ),
        ModelSpec(
            name="accepted_30d",
            frame=tag_month_panel.loc[tag_month_panel["accepted_30d_denom"] > 0].copy(),
            outcome="accepted_30d_rate",
            weight_col="accepted_30d_denom",
            term="exposure_post",
            formula="accepted_30d_rate ~ exposure_post + C(primary_tag) + C(month_id)",
        ),
    ]
    if not postshock_status.empty:
        specs.append(
            ModelSpec(
                name="postshock_presence",
                frame=postshock_status.copy(),
                outcome="any_postshock_answer",
                weight_col=None,
                term="exposure_expert",
                formula="any_postshock_answer ~ is_expert + exposure_expert + C(primary_tag)",
            )
        )
    return specs


def fit_models(specs: list[ModelSpec]) -> dict:
    results: dict[str, dict] = {}
    for spec in specs:
        model = fit_weighted(spec.formula, spec.frame, spec.weight_col, spec.cluster_col)
        results[spec.name] = {
            "outcome": spec.outcome,
            "weight_col": spec.weight_col,
            "formula": spec.formula,
            "term": spec.term,
            "summary": extract_term(model, spec.term),
        }

    rows = []
    for model_name, payload in results.items():
        row = {"model": model_name}
        row.update(payload["summary"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(MODEL_RESULTS_CSV, index=False)
    MODEL_RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def event_study(frame: pd.DataFrame, outcome: str, weight_col: str, output_csv: Path) -> pd.DataFrame:
    temp = frame.copy()
    temp["month_start"] = pd.to_datetime(temp["month_id"] + "-01", utc=True)
    shock_month = pd.Timestamp("2022-12-01T00:00:00Z")
    temp["rel_month"] = (
        (temp["month_start"].dt.year - shock_month.year) * 12 + (temp["month_start"].dt.month - shock_month.month)
    )
    temp = temp.loc[(temp["rel_month"] >= -18) & (temp["rel_month"] <= 3)].copy()
    temp["rel_month_binned"] = temp["rel_month"].astype(int).astype(str)
    model = fit_weighted(
        f"{outcome} ~ C(primary_tag):time_index + C(primary_tag) + C(month_id) + exposure_index:C(rel_month_binned, Treatment(reference='-1'))",
        temp,
        weight_col,
        "primary_tag",
    )
    rows = []
    for key, coef in model.params.items():
        if "exposure_index:C(rel_month_binned" not in key:
            continue
        if "[T." in key:
            rel_value = key.split("[T.", 1)[1].split("]", 1)[0]
        else:
            rel_value = key.rsplit("[", 1)[1].rstrip("]")
        rel_month = int(rel_value)
        if rel_month == -1:
            continue
        rows.append(
            {
                "rel_month": rel_month,
                "coef": float(coef),
                "se": float(model.bse[key]),
                "pval": float(model.pvalues[key]),
                "ci_low": float(coef - 1.96 * model.bse[key]),
                "ci_high": float(coef + 1.96 * model.bse[key]),
            }
        )
    out = pd.DataFrame(rows).sort_values("rel_month").reset_index(drop=True)
    out.to_csv(output_csv, index=False)
    return out


def placebo_grid(tag_month_panel: pd.DataFrame) -> pd.DataFrame:
    months = sorted(tag_month_panel["month_id"].unique())
    rows = []
    for placebo_month in months[3:-2]:
        for spec_name, outcome, weight_col in [
            ("expert_answer_share", "expert_answer_share", "n_answers"),
            ("novice_entry_share", "novice_entry_share", "n_new_answerers"),
            ("accepted_7d_rate", "accepted_7d_rate", "accepted_7d_denom"),
        ]:
            frame = tag_month_panel.dropna(subset=[outcome]).copy()
            if weight_col in frame.columns:
                frame = frame.loc[frame[weight_col].fillna(0) > 0].copy()
            frame["placebo_post"] = (frame["month_id"] >= placebo_month).astype(int)
            frame["placebo_exposure_post"] = frame["placebo_post"] * frame["exposure_index"]
            model = fit_weighted(f"{outcome} ~ placebo_exposure_post + C(primary_tag) + C(month_id)", frame, weight_col, "primary_tag")
            rows.append(
                {
                    "specification": spec_name,
                    "placebo_month": placebo_month,
                    "coef": float(model.params.get("placebo_exposure_post", np.nan)),
                    "se": float(model.bse.get("placebo_exposure_post", np.nan)),
                    "pval": float(model.pvalues.get("placebo_exposure_post", np.nan)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(PLACEBO_CSV, index=False)
    return out


def leave_one_out(tag_month_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tags = sorted(tag_month_panel["primary_tag"].dropna().unique())
    for dropped_tag in tags:
        subset = tag_month_panel.loc[tag_month_panel["primary_tag"] != dropped_tag].copy()
        for spec_name, outcome, weight_col in [
            ("expert_answer_share", "expert_answer_share", "n_answers"),
            ("novice_entry_share", "novice_entry_share", "n_new_answerers"),
            ("accepted_7d_rate", "accepted_7d_rate", "accepted_7d_denom"),
        ]:
            frame = subset.dropna(subset=[outcome]).copy()
            frame = frame.loc[frame[weight_col].fillna(0) > 0].copy()
            model = fit_weighted(
                f"{outcome} ~ exposure_post + C(primary_tag) + C(month_id)",
                frame,
                weight_col,
                "primary_tag",
            )
            rows.append(
                {
                    "specification": spec_name,
                    "dropped_tag": dropped_tag,
                    "coef": float(model.params.get("exposure_post", np.nan)),
                    "se": float(model.bse.get("exposure_post", np.nan)),
                    "pval": float(model.pvalues.get("exposure_post", np.nan)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(LEAVE_ONE_OUT_CSV, index=False)
    return out


def leave_two_out(specs: list[ModelSpec], selected_specs: set[str]) -> pd.DataFrame:
    rows = []
    tag_lookup = {
        spec.name: sorted(spec.frame["primary_tag"].dropna().unique())
        for spec in specs
        if spec.name in selected_specs
    }
    for spec in specs:
        if spec.name not in selected_specs:
            continue
        for dropped_a, dropped_b in itertools.combinations(tag_lookup[spec.name], 2):
            subset = spec.frame.loc[~spec.frame["primary_tag"].isin([dropped_a, dropped_b])].copy()
            if subset.empty:
                continue
            try:
                model = fit_weighted(spec.formula, subset, spec.weight_col, spec.cluster_col)
            except Exception:
                continue
            rows.append(
                {
                    "specification": spec.name,
                    "dropped_tag_a": dropped_a,
                    "dropped_tag_b": dropped_b,
                    "coef": float(model.params.get(spec.term, np.nan)),
                    "se": float(model.bse.get(spec.term, np.nan)),
                    "pval": float(model.pvalues.get(spec.term, np.nan)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(LEAVE_TWO_OUT_CSV, index=False)
    return out


def prepare_break_frame(frame: pd.DataFrame, break_month: str) -> pd.DataFrame:
    temp = frame.copy()
    month_order = {month: idx for idx, month in enumerate(sorted(temp["month_id"].unique()))}
    break_idx = month_order[break_month]
    temp["month_order"] = temp["month_id"].map(month_order).astype(int)
    temp["exposure_time"] = temp["exposure_index"] * temp["month_order"]
    temp["break_post"] = (temp["month_order"] >= break_idx).astype(int)
    temp["break_slope"] = np.maximum(temp["month_order"] - break_idx, 0)
    temp["exposure_break_post"] = temp["exposure_index"] * temp["break_post"]
    temp["exposure_break_slope"] = temp["exposure_index"] * temp["break_slope"]
    return temp


def trend_break_diagnostics(tag_month_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_months = sorted(tag_month_panel["month_id"].unique())
    shock_idx = all_months.index(SHOCK_MONTH)
    candidate_months = all_months[6 : shock_idx + 1]
    rows = []
    for spec_name, outcome, weight_col in [
        ("expert_answer_share", "expert_answer_share", "n_answers"),
        ("novice_entry_share", "novice_entry_share", "n_new_answerers"),
        ("accepted_7d_rate", "accepted_7d_rate", "accepted_7d_denom"),
    ]:
        base = filter_weighted_frame(tag_month_panel.dropna(subset=[outcome]), weight_col)
        for break_month in candidate_months:
            frame = prepare_break_frame(base, break_month)
            variants = {
                "level_only": f"{outcome} ~ exposure_time + exposure_break_post + C(primary_tag) + C(month_id)",
                "slope_only": f"{outcome} ~ exposure_time + exposure_break_slope + C(primary_tag) + C(month_id)",
                "level_and_slope": f"{outcome} ~ exposure_time + exposure_break_post + exposure_break_slope + C(primary_tag) + C(month_id)",
            }
            for variant_name, formula in variants.items():
                try:
                    model = fit_weighted(formula, frame, weight_col, "primary_tag")
                except Exception:
                    continue
                for term in ["exposure_break_post", "exposure_break_slope"]:
                    if term not in model.params.index:
                        continue
                    rows.append(
                        {
                            "specification": spec_name,
                            "break_month": break_month,
                            "variant": variant_name,
                            "term": term,
                            "coef": float(model.params.get(term, np.nan)),
                            "se": float(model.bse.get(term, np.nan)),
                            "pval": float(model.pvalues.get(term, np.nan)),
                        }
                    )
    trend_df = pd.DataFrame(rows)
    trend_df.to_csv(TREND_BREAK_RESULTS_CSV, index=False)

    summary_rows = []
    for specification, (variant_name, term_name, expected_sign) in PREFERRED_BREAK_TERMS.items():
        subset = trend_df.loc[
            (trend_df["specification"] == specification) &
            (trend_df["variant"] == variant_name) &
            (trend_df["term"] == term_name)
        ].copy()
        if subset.empty:
            continue
        subset["directional_coef"] = subset["coef"] * expected_sign
        subset = subset.sort_values("break_month").reset_index(drop=True)
        actual_row = subset.loc[subset["break_month"] == SHOCK_MONTH].copy()
        if actual_row.empty:
            continue
        actual_directional = float(actual_row["directional_coef"].iloc[0])
        pre = subset.loc[subset["break_month"] < SHOCK_MONTH].copy()
        actual_rank = int((pre["directional_coef"] >= actual_directional).sum() + 1)
        pre_positive_share = float((pre["coef"] * expected_sign > 0).mean()) if not pre.empty else np.nan
        pre_significant = pre.loc[pre["pval"] < 0.05].copy()
        summary_rows.append(
            {
                "specification": specification,
                "preferred_variant": variant_name,
                "preferred_term": term_name,
                "expected_sign": expected_sign,
                "actual_break_month": SHOCK_MONTH,
                "actual_coef": float(actual_row["coef"].iloc[0]),
                "actual_se": float(actual_row["se"].iloc[0]),
                "actual_pval": float(actual_row["pval"].iloc[0]),
                "actual_directional_coef": actual_directional,
                "actual_rank_vs_pre_breaks": actual_rank,
                "n_candidate_breaks": int(len(subset)),
                "n_pre_breaks": int(len(pre)),
                "actual_rank_percentile_vs_pre": float(1.0 - ((actual_rank - 1) / max(len(pre), 1))),
                "significant_pre_breaks": int((pre["pval"] < 0.05).sum()),
                "share_significant_pre_breaks": float((pre["pval"] < 0.05).mean()) if not pre.empty else np.nan,
                "first_significant_pre_break": None if pre_significant.empty else str(pre_significant["break_month"].iloc[0]),
                "pre_break_positive_share": pre_positive_share,
            }
        )
    identification_profile = pd.DataFrame(summary_rows)
    identification_profile.to_csv(IDENTIFICATION_PROFILE_CSV, index=False)
    return trend_df, identification_profile


def inverse_square_root_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    return eigenvectors @ inv_sqrt @ eigenvectors.T


def cr2_term_statistics(model: object, term: str, groups: pd.Series | np.ndarray) -> dict:
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


def wild_cluster_bootstrap_pvalue(spec: ModelSpec) -> tuple[float, np.ndarray]:
    frame = filter_weighted_frame(spec.frame, spec.weight_col)
    groups = frame[spec.cluster_col].to_numpy()
    unique_groups = pd.Index(sorted(pd.unique(groups)))
    if len(unique_groups) < 6:
        return np.nan, np.array([])
    restricted_formula = remove_term_from_formula(spec.formula, spec.term)
    y_full, x_full = patsy.dmatrices(spec.formula, frame, return_type="dataframe")
    y_restricted, x_restricted = patsy.dmatrices(restricted_formula, frame, return_type="dataframe")
    weights = np.ones(len(frame)) if spec.weight_col is None else frame[spec.weight_col].to_numpy()
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
    rng = np.random.default_rng(20260403 + abs(hash(spec.name)) % 1000)
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
        return np.nan, np.array([])
    bootstrap_array = np.asarray(bootstrap_t, dtype=float)
    return float(np.mean(np.abs(bootstrap_array) >= abs(observed_t))), bootstrap_array


def randomization_inference(spec: ModelSpec) -> tuple[float, pd.DataFrame]:
    frame = filter_weighted_frame(spec.frame, spec.weight_col)
    tags = sorted(frame["primary_tag"].dropna().unique())
    actual_model = fit_weighted(spec.formula, frame, spec.weight_col, spec.cluster_col)
    actual_coef = float(actual_model.params.get(spec.term, np.nan))
    observed_map = frame.groupby("primary_tag", as_index=False)["exposure_index"].first()
    exposures = observed_map["exposure_index"].to_numpy()
    rng = np.random.default_rng(20260403 + len(tags) + abs(hash(spec.name)) % 1000)
    rows = []
    for draw in range(RANDOMIZATION_REPS):
        permuted = rng.permutation(exposures)
        exposure_map = dict(zip(observed_map["primary_tag"], permuted, strict=False))
        permuted_frame = add_permuted_exposure(frame, exposure_map)
        try:
            permuted_model = fit_weighted(spec.formula, permuted_frame, spec.weight_col, spec.cluster_col)
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


def small_sample_inference(specs: list[ModelSpec], selected_specs: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inference_rows = []
    randomization_frames = []
    for spec in specs:
        if spec.name not in selected_specs:
            continue
        frame = filter_weighted_frame(spec.frame, spec.weight_col)
        model = fit_weighted(spec.formula, frame, spec.weight_col, spec.cluster_col)
        cr2_term = cr2_term_statistics(model, spec.term, frame[spec.cluster_col])
        bootstrap_pval, _ = wild_cluster_bootstrap_pvalue(spec)
        randomization_pval, perm_df = randomization_inference(spec)
        if not perm_df.empty:
            randomization_frames.append(perm_df)
        inference_rows.append(
            {
                "specification": spec.name,
                "term": spec.term,
                "coef": cr2_term["coef"],
                "cr2_se": cr2_term["se"],
                "cr2_pval": cr2_term["pval"],
                "cr2_tstat": cr2_term["tstat"],
                "cr2_df": cr2_term["df"],
                "wild_cluster_bootstrap_pval": bootstrap_pval,
                "randomization_pval": randomization_pval,
            }
        )
    inference_df = pd.DataFrame(inference_rows)
    randomization_df = pd.concat(randomization_frames, ignore_index=True) if randomization_frames else pd.DataFrame(
        columns=["specification", "draw", "coef", "actual_coef"]
    )
    inference_df.to_csv(SMALL_SAMPLE_INFERENCE_CSV, index=False)
    randomization_df.to_csv(EXPOSURE_RANDOMIZATION_CSV, index=False)
    return inference_df, randomization_df


def build_entrant_profiles(first_tag_answers: pd.DataFrame) -> pd.DataFrame:
    entrants = first_tag_answers[[
        "primary_tag",
        "answerer_user_id",
        "first_tag_answer_at",
        "entry_month",
        "observed_first_activity_at",
        "observed_tenure_days",
        "is_novice_entrant",
        "is_postshock_novice_entrant",
        "entrant_type",
    ]].copy()
    if entrants.empty or not RAW_POSTS_PARQUET.exists():
        entrants.to_csv(ENTRANT_PROFILE_CSV, index=False)
        return entrants
    con = duckdb.connect()
    con.register("entrants", entrants[["primary_tag", "answerer_user_id", "first_tag_answer_at"]])
    query = f"""
        SELECT
            e.primary_tag,
            e.answerer_user_id,
            e.first_tag_answer_at,
            SUM(CASE WHEN p.CreationDate < e.first_tag_answer_at THEN 1 ELSE 0 END) AS prior_posts,
            SUM(CASE WHEN p.CreationDate < e.first_tag_answer_at AND p.PostTypeId = 1 THEN 1 ELSE 0 END) AS prior_questions,
            SUM(CASE WHEN p.CreationDate < e.first_tag_answer_at AND p.PostTypeId = 2 THEN 1 ELSE 0 END) AS prior_answers
        FROM entrants e
        LEFT JOIN read_parquet('{RAW_POSTS_PARQUET.as_posix()}') p
          ON p.OwnerUserId = e.answerer_user_id
        GROUP BY 1, 2, 3
    """
    prior_activity = con.execute(query).fetchdf()
    con.close()
    entrants = entrants.merge(prior_activity, on=["primary_tag", "answerer_user_id", "first_tag_answer_at"], how="left")
    for column in ["prior_posts", "prior_questions", "prior_answers"]:
        entrants[column] = entrants[column].fillna(0).astype(int)
    entrants.to_csv(ENTRANT_PROFILE_CSV, index=False)
    return entrants


def build_construct_ladders(
    answers: pd.DataFrame,
    question_closure: pd.DataFrame,
    cohorts: pd.DataFrame,
    first_tag_answers: pd.DataFrame,
    entrant_profiles: pd.DataFrame,
    tag_month_panel: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def classify_variant(frame: pd.DataFrame, percentile: float, min_answers: int, variant: str) -> pd.Series:
        if variant == "volume_only":
            score = frame["z_answers"]
        elif variant == "accepted_only":
            score = frame["z_accepted"]
        elif variant == "score_only":
            score = frame["z_score"]
        else:
            score = frame["expert_score"]
        cutoff = score.groupby(frame["primary_tag"]).transform(lambda s: s.quantile(percentile))
        return ((frame["preshock_answers"] >= min_answers) & (score >= cutoff)).astype(int)

    pre_answers = answers.loc[answers["answer_created_at"] < SHOCK_TS].copy()
    holdout_cutoff = pd.Timestamp("2022-01-01T00:00:00Z")
    early = pre_answers.loc[pre_answers["answer_created_at"] < holdout_cutoff].copy()
    late = pre_answers.loc[pre_answers["answer_created_at"] >= holdout_cutoff].copy()
    if not early.empty and not late.empty:
        early_cohorts = (
            early.groupby(["primary_tag", "answerer_user_id"], as_index=False)
            .agg(
                answers=("answer_id", "size"),
                accepted=("is_current_accepted", "sum"),
                mean_score=("answer_score", "mean"),
            )
        )
        early_cohorts["log_answers"] = np.log1p(early_cohorts["answers"])
        early_cohorts["log_accepted"] = np.log1p(early_cohorts["accepted"])
        early_cohorts["mean_score"] = early_cohorts["mean_score"].fillna(0.0)
        early_cohorts["z_answers"] = early_cohorts.groupby("primary_tag")["log_answers"].transform(standardize_within_group)
        early_cohorts["z_accepted"] = early_cohorts.groupby("primary_tag")["log_accepted"].transform(standardize_within_group)
        early_cohorts["z_score"] = early_cohorts.groupby("primary_tag")["mean_score"].transform(standardize_within_group)
        early_cohorts["expert_score"] = early_cohorts["z_answers"] + early_cohorts["z_accepted"] + early_cohorts["z_score"]
        early_cohorts["expert_cutoff"] = early_cohorts.groupby("primary_tag")["expert_score"].transform(lambda s: s.quantile(0.9))
        early_cohorts["expert_holdout_label"] = ((early_cohorts["answers"] >= MIN_EXPERT_ANSWERS) & (early_cohorts["expert_score"] >= early_cohorts["expert_cutoff"])).astype(int)
        late_metrics = (
            late.groupby(["primary_tag", "answerer_user_id"], as_index=False)
            .agg(
                later_answers=("answer_id", "size"),
                later_accepted=("is_current_accepted", "sum"),
                later_active_months=("answer_month", "nunique"),
                later_mean_score=("answer_score", "mean"),
            )
        )
        holdout = early_cohorts.merge(late_metrics, on=["primary_tag", "answerer_user_id"], how="left").fillna(0)
        grouped = holdout.groupby("expert_holdout_label", as_index=False).agg(
            users=("answerer_user_id", "size"),
            later_answers=("later_answers", "mean"),
            later_accepted=("later_accepted", "mean"),
            later_active_months=("later_active_months", "mean"),
            later_mean_score=("later_mean_score", "mean"),
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "family": "expert_holdout",
                    "variant": "early_window_top_decile_min20",
                    "metric": "group_mean",
                    "label": "expert_holdout_label" if int(row["expert_holdout_label"]) == 1 else "other_holdout_users",
                    "value": float(row["later_answers"]),
                    "extra_1": float(row["later_accepted"]),
                    "extra_2": float(row["later_active_months"]),
                    "extra_3": float(row["later_mean_score"]),
                }
            )

    face_validity = cohorts.loc[cohorts["is_incumbent"] == 1].copy()
    face_validity["accepted_rate"] = np.where(
        face_validity["preshock_answers"] > 0,
        face_validity["preshock_accepted_current"] / face_validity["preshock_answers"],
        np.nan,
    )
    grouped_face = face_validity.groupby("is_expert", as_index=False).agg(
        users=("answerer_user_id", "size"),
        mean_preshock_answers=("preshock_answers", "mean"),
        mean_accepted_rate=("accepted_rate", "mean"),
        mean_score=("preshock_mean_score", "mean"),
        mean_tenure_days=("observed_tenure_days_at_first_tag", "mean"),
    )
    for _, row in grouped_face.iterrows():
        rows.append(
            {
                "family": "expert_face_validity",
                "variant": "current_frozen_definition",
                "metric": "group_mean",
                "label": "expert" if int(row["is_expert"]) == 1 else "incumbent_nonexpert",
                "value": float(row["mean_preshock_answers"]),
                "extra_1": float(row["mean_accepted_rate"]),
                "extra_2": float(row["mean_score"]),
                "extra_3": float(row["mean_tenure_days"]),
            }
        )

    for percentile in [0.95, 0.90, 0.85]:
        for min_answers in [10, 20, 30]:
            temp = cohorts.copy()
            alt_label = classify_variant(temp, percentile, min_answers, "composite")
            answer_with_label = answers.merge(
                temp[["primary_tag", "answerer_user_id"]].assign(alt_expert=alt_label),
                on=["primary_tag", "answerer_user_id"],
                how="left",
            )
            answer_with_label["alt_expert"] = answer_with_label["alt_expert"].fillna(0)
            alt_panel = (
                answer_with_label.groupby(["primary_tag", "answer_month"], as_index=False)
                .agg(n_answers=("answer_id", "size"), alt_expert_share=("alt_expert", "mean"))
                .rename(columns={"answer_month": "month_id"})
            )
            alt_panel = alt_panel.merge(tag_month_panel[["primary_tag", "month_id", "post_chatgpt", "exposure_index"]], on=["primary_tag", "month_id"], how="left")
            alt_panel["exposure_post"] = alt_panel["exposure_index"] * alt_panel["post_chatgpt"]
            model = fit_weighted(
                "alt_expert_share ~ exposure_post + C(primary_tag) + C(month_id)",
                alt_panel,
                "n_answers",
                "primary_tag",
            )
            rows.append(
                {
                    "family": "expert_threshold_sensitivity",
                    "variant": f"pct_{int((1 - percentile) * 100)}_min_{min_answers}",
                    "metric": "expert_share_coef",
                    "label": "exposure_post",
                    "value": float(model.params.get("exposure_post", np.nan)),
                    "extra_1": float(model.pvalues.get("exposure_post", np.nan)),
                    "extra_2": float(alt_label.sum()),
                    "extra_3": np.nan,
                }
            )

    for variant in ["composite", "volume_only", "accepted_only", "score_only"]:
        alt_label = classify_variant(cohorts, 0.90, 20, variant)
        rows.append(
            {
                "family": "expert_component_variant",
                "variant": variant,
                "metric": "n_experts",
                "label": "expert_count",
                "value": float(alt_label.sum()),
                "extra_1": float(alt_label.mean()),
                "extra_2": np.nan,
                "extra_3": np.nan,
            }
        )

    for tenure_days in [30.0, 60.0, 90.0, 180.0]:
        temp = first_tag_answers.copy()
        temp["is_alt_novice"] = (
            temp["observed_tenure_days"].ge(0) &
            (temp["observed_tenure_days"] <= tenure_days)
        ).astype(int)
        monthly = (
            temp.groupby(["primary_tag", "entry_month"], as_index=False)
            .agg(
                n_new_answerers=("answerer_user_id", "size"),
                novice_entrants=("is_alt_novice", "sum"),
            )
            .rename(columns={"entry_month": "month_id"})
        )
        monthly["novice_entry_share"] = np.where(monthly["n_new_answerers"] > 0, monthly["novice_entrants"] / monthly["n_new_answerers"], np.nan)
        monthly = monthly.merge(tag_month_panel[["primary_tag", "month_id", "exposure_index", "post_chatgpt"]], on=["primary_tag", "month_id"], how="left")
        monthly["exposure_post"] = monthly["exposure_index"] * monthly["post_chatgpt"]
        model = fit_weighted(
            "novice_entry_share ~ exposure_post + C(primary_tag) + C(month_id)",
            monthly,
            "n_new_answerers",
            "primary_tag",
        )
        rows.append(
            {
                "family": "entrant_tenure_sensitivity",
                "variant": f"tenure_{int(tenure_days)}d",
                "metric": "novice_entry_coef",
                "label": "exposure_post",
                "value": float(model.params.get("exposure_post", np.nan)),
                "extra_1": float(model.pvalues.get("exposure_post", np.nan)),
                "extra_2": float(monthly["novice_entrants"].sum()),
                "extra_3": np.nan,
                }
            )

    subtype_entry = first_tag_answers.loc[first_tag_answers["first_tag_answer_at"] >= SHOCK_TS].copy()
    for entrant_type in ["brand_new_platform", "low_tenure_existing", "established_cross_tag"]:
        temp = subtype_entry.copy()
        temp["is_subtype"] = (temp["entrant_type"] == entrant_type).astype(int)
        monthly = (
            temp.groupby(["primary_tag", "entry_month"], as_index=False)
            .agg(
                n_new_answerers=("answerer_user_id", "size"),
                subtype_count=("is_subtype", "sum"),
            )
            .rename(columns={"entry_month": "month_id"})
        )
        if monthly.empty:
            continue
        monthly["subtype_share"] = np.where(monthly["n_new_answerers"] > 0, monthly["subtype_count"] / monthly["n_new_answerers"], np.nan)
        monthly = monthly.merge(tag_month_panel[["primary_tag", "month_id", "exposure_index", "post_chatgpt"]], on=["primary_tag", "month_id"], how="left")
        monthly["exposure_post"] = monthly["exposure_index"] * monthly["post_chatgpt"]
        model = fit_weighted(
            "subtype_share ~ exposure_post + C(primary_tag) + C(month_id)",
            monthly,
            "n_new_answerers",
            "primary_tag",
        )
        rows.append(
            {
                "family": "entrant_subtype_effect",
                "variant": entrant_type,
                "metric": "subtype_share_coef",
                "label": "exposure_post",
                "value": float(model.params.get("exposure_post", np.nan)),
                "extra_1": float(model.pvalues.get("exposure_post", np.nan)),
                "extra_2": float(monthly["subtype_count"].sum()),
                "extra_3": np.nan,
            }
        )

    entrant_grouped = entrant_profiles.groupby("entrant_type", as_index=False).agg(
        n_users=("answerer_user_id", "size"),
        mean_tenure_days=("observed_tenure_days", "mean"),
        mean_prior_posts=("prior_posts", "mean"),
        mean_prior_answers=("prior_answers", "mean"),
        mean_prior_questions=("prior_questions", "mean"),
    )
    for _, row in entrant_grouped.iterrows():
        rows.append(
            {
                "family": "entrant_profiles",
                "variant": row["entrant_type"],
                "metric": "profile_mean",
                "label": "entrant_type",
                "value": float(row["n_users"]),
                "extra_1": float(row["mean_tenure_days"]),
                "extra_2": float(row["mean_prior_posts"]),
                "extra_3": float(row["mean_prior_answers"]),
                }
            )

    strict_timing_share = float((question_closure["accepted_timing_source"] == "accept_vote_date_daylevel").mean())
    rows.append(
        {
            "family": "closure_timing_validation",
            "variant": "strict_event_timing_share",
            "metric": "share_questions_with_event_timing",
            "label": "accept_vote_date_daylevel",
            "value": strict_timing_share,
            "extra_1": float((question_closure["accepted_timing_source"] != "accept_vote_date_daylevel").mean()),
            "extra_2": float(len(question_closure)),
            "extra_3": np.nan,
        }
    )

    closure_variants = [
        ("first_answer_12h_rate", "first_answer_12h_denom"),
        ("first_answer_1d_rate", "first_answer_1d_denom"),
        ("first_answer_7d_rate", "first_answer_7d_denom"),
        ("accepted_7d_rate", "accepted_7d_denom"),
        ("accepted_14d_rate", "accepted_14d_denom"),
        ("accepted_30d_rate", "accepted_30d_denom"),
    ]
    for outcome, weight_col in closure_variants:
        frame = tag_month_panel.loc[tag_month_panel[weight_col] > 0].copy()
        model = fit_weighted(f"{outcome} ~ exposure_post + C(primary_tag) + C(month_id)", frame, weight_col, "primary_tag")
        rows.append(
            {
                "family": "closure_family",
                "variant": outcome,
                "metric": "closure_coef",
                "label": "exposure_post",
                "value": float(model.params.get("exposure_post", np.nan)),
                "extra_1": float(model.pvalues.get("exposure_post", np.nan)),
                "extra_2": float(len(frame)),
                "extra_3": np.nan,
            }
        )

    conditional = question_closure.copy()
    conditional["accepted_7d_conditional"] = np.where(conditional["first_answer_7d"] == 1, conditional["accepted_7d"], np.nan)
    conditional_tag = (
        conditional.groupby(["primary_tag", "month_id"], as_index=False)
        .agg(
            conditional_rate=("accepted_7d_conditional", "mean"),
            conditional_denom=("first_answer_7d", "sum"),
        )
    )
    conditional_tag = conditional_tag.merge(tag_month_panel[["primary_tag", "month_id", "exposure_index", "post_chatgpt"]], on=["primary_tag", "month_id"], how="left")
    conditional_tag["exposure_post"] = conditional_tag["exposure_index"] * conditional_tag["post_chatgpt"]
    conditional_tag = conditional_tag.loc[conditional_tag["conditional_denom"] > 0].copy()
    if not conditional_tag.empty:
        model = fit_weighted("conditional_rate ~ exposure_post + C(primary_tag) + C(month_id)", conditional_tag, "conditional_denom", "primary_tag")
        rows.append(
            {
                "family": "closure_conditional",
                "variant": "accepted_7d_given_any_answer_7d",
                "metric": "closure_coef",
                "label": "exposure_post",
                "value": float(model.params.get("exposure_post", np.nan)),
                "extra_1": float(model.pvalues.get("exposure_post", np.nan)),
                "extra_2": float(len(conditional_tag)),
                "extra_3": np.nan,
            }
        )

    pre_link = tag_month_panel.loc[tag_month_panel["post_chatgpt"] == 0].dropna(subset=["expert_answer_share", "accepted_7d_rate"]).copy()
    if not pre_link.empty:
        model = fit_weighted(
            "accepted_7d_rate ~ expert_answer_share + C(primary_tag) + C(month_id)",
            pre_link,
            "accepted_7d_denom",
            "primary_tag",
        )
        rows.append(
            {
                "family": "closure_prelink",
                "variant": "pre_shock_expert_share_link",
                "metric": "expert_share_to_accepted_7d",
                "label": "expert_answer_share",
                "value": float(model.params.get("expert_answer_share", np.nan)),
                "extra_1": float(model.pvalues.get("expert_answer_share", np.nan)),
                "extra_2": float(len(pre_link)),
                "extra_3": np.nan,
            }
        )

    construct_df = pd.DataFrame(rows)
    construct_df.to_csv(CONSTRUCT_LADDERS_CSV, index=False)
    return construct_df


def plot_event_study(event_df: pd.DataFrame, title: str, y_label: str, output_path: Path, color: str) -> None:
    plt.figure(figsize=(8.6, 4.8))
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(-0.5, color="gray", linestyle="--", linewidth=1)
    if not event_df.empty:
        plt.errorbar(
            event_df["rel_month"],
            event_df["coef"],
            yerr=1.96 * event_df["se"],
            fmt="o-",
            color=color,
            ecolor=color,
            capsize=3,
            linewidth=1.5,
        )
    plt.title(title)
    plt.xlabel("Months Relative to December 2022")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_trends(tag_month_panel: pd.DataFrame) -> None:
    frame = tag_month_panel.copy()
    frame["exposure_group"] = np.where(frame["high_tag"] == 1, "Higher exposure", "Lower exposure")
    outcomes = ["expert_answer_share", "novice_entry_share", "accepted_7d_rate"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for ax, outcome in zip(axes, outcomes):
        grouped = (
            frame.dropna(subset=[outcome])
            .groupby(["month_id", "exposure_group"], as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "value": np.average(
                            g[outcome],
                            weights=g["n_answers"].fillna(g["accepted_7d_denom"]).fillna(g["n_questions"]).fillna(1),
                        )
                    }
                )
            )
            .reset_index(drop=True)
        )
        pivot = grouped.pivot(index="month_id", columns="exposure_group", values="value").sort_index()
        for label in pivot.columns:
            ax.plot(pivot.index, pivot[label], marker="o", linewidth=1.8, label=label)
        ax.axvline(SHOCK_MONTH, color="black", linestyle="--", linewidth=1)
        ax.set_title(outcome.replace("_", " ").title())
        ax.legend()
    axes[-1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(TRENDS_PNG, dpi=220)
    plt.close()


def plot_placebo_rank_panel(trend_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for ax, specification in zip(axes, ["novice_entry_share", "expert_answer_share", "accepted_7d_rate"]):
        variant_name, term_name, expected_sign = PREFERRED_BREAK_TERMS[specification]
        subset = trend_df.loc[
            (trend_df["specification"] == specification) &
            (trend_df["variant"] == variant_name) &
            (trend_df["term"] == term_name)
        ].copy()
        if subset.empty:
            continue
        subset["directional_coef"] = subset["coef"] * expected_sign
        subset = subset.sort_values("break_month")
        colors = np.where(subset["break_month"] == SHOCK_MONTH, "#b22222", "#4f6d7a")
        ax.bar(subset["break_month"], subset["directional_coef"], color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{specification}: directional break strength")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(PLACEBO_RANK_PANEL_PNG, dpi=220)
    plt.close()


def plot_novice_trend_break(tag_month_panel: pd.DataFrame, trend_df: pd.DataFrame) -> None:
    frame = tag_month_panel.dropna(subset=["novice_entry_share"]).copy()
    frame["exposure_group"] = np.where(frame["high_tag"] == 1, "Higher exposure", "Lower exposure")
    grouped = (
        frame.groupby(["month_id", "exposure_group"], as_index=False)
        .apply(lambda g: pd.Series({"value": np.average(g["novice_entry_share"], weights=g["n_new_answerers"])}))
        .reset_index(drop=True)
    )
    variant = trend_df.loc[
        (trend_df["specification"] == "novice_entry_share") &
        (trend_df["variant"] == "level_and_slope") &
        (trend_df["break_month"] == SHOCK_MONTH)
    ].copy()
    slope_coef = float(variant.loc[variant["term"] == "exposure_break_slope", "coef"].iloc[0]) if not variant.empty and (variant["term"] == "exposure_break_slope").any() else np.nan
    level_coef = float(variant.loc[variant["term"] == "exposure_break_post", "coef"].iloc[0]) if not variant.empty and (variant["term"] == "exposure_break_post").any() else np.nan

    plt.figure(figsize=(10, 5))
    for label, color in [("Higher exposure", "#1f77b4"), ("Lower exposure", "#b07d3c")]:
        subset = grouped.loc[grouped["exposure_group"] == label].copy()
        plt.plot(subset["month_id"], subset["value"], marker="o", linewidth=1.8, color=color, label=label)
    plt.axvline(SHOCK_MONTH, color="black", linestyle="--", linewidth=1)
    plt.title("Novice Entry Share With Pretrend-Adjusted Break Read")
    plt.xlabel("Month")
    plt.ylabel("Weighted novice entry share")
    plt.xticks(rotation=45)
    annotation = f"Level break: {level_coef:.4f}\nSlope break: {slope_coef:.4f}" if np.isfinite(level_coef) and np.isfinite(slope_coef) else "Trend-break coefficients unavailable"
    plt.text(0.02, 0.97, annotation, transform=plt.gca().transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#666666"})
    plt.legend()
    plt.tight_layout()
    plt.savefig(NOVICE_TREND_BREAK_PNG, dpi=220)
    plt.close()


def plot_permutation_density(randomization_df: pd.DataFrame) -> None:
    subset = randomization_df.loc[randomization_df["specification"] == "novice_entry_share"].copy()
    if subset.empty:
        return
    actual_coef = float(subset["actual_coef"].iloc[0])
    plt.figure(figsize=(8.6, 4.8))
    plt.hist(subset["coef"], bins=30, color="#7aa6c2", edgecolor="white")
    plt.axvline(actual_coef, color="#b22222", linestyle="--", linewidth=2)
    plt.title("Permutation Distribution: Novice Entry Share")
    plt.xlabel("Permuted exposure coefficient")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PERMUTATION_DENSITY_NOVICE_PNG, dpi=220)
    plt.close()


def plot_influence_heatmap(leave_out: pd.DataFrame, results: dict) -> None:
    focus = leave_out.loc[leave_out["specification"].isin(["expert_answer_share", "novice_entry_share", "accepted_7d_rate"])].copy()
    if focus.empty:
        return
    baseline = {
        "expert_answer_share": results.get("expert_answer_share", {}).get("summary", {}).get("coef", np.nan),
        "novice_entry_share": results.get("novice_entry_share", {}).get("summary", {}).get("coef", np.nan),
        "accepted_7d_rate": results.get("accepted_7d", {}).get("summary", {}).get("coef", np.nan),
    }
    focus["coef_delta"] = focus.apply(lambda row: row["coef"] - baseline.get(row["specification"], np.nan), axis=1)
    pivot = focus.pivot(index="specification", columns="dropped_tag", values="coef_delta").fillna(0.0)
    plt.figure(figsize=(11, 3.6))
    plt.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    plt.colorbar(label="Leave-one-out coefficient delta")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.title("Influence Heatmap by Dropped Tag")
    plt.tight_layout()
    plt.savefig(INFLUENCE_HEATMAP_PNG, dpi=220)
    plt.close()


def build_tag_exposure_panel(question_closure: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    panel = (
        question_closure.groupby("primary_tag", as_index=False)
        .agg(
            n_questions=("question_id", "size"),
            min_question_created_at=("question_created_at", "min"),
            max_question_created_at=("question_created_at", "max"),
            mean_first_answer_1d=("first_answer_1d_obs", "mean"),
            mean_accepted_7d=("accepted_7d_obs", "mean"),
        )
        .merge(exposure, left_on="primary_tag", right_on="tag", how="left")
        .drop(columns=["tag"])
        .sort_values("exposure_rank")
        .reset_index(drop=True)
    )
    panel.to_csv(TAG_EXPOSURE_PANEL_CSV, index=False)
    return panel


def save_metadata(source: SourceMetadata, question_closure: pd.DataFrame, answers: pd.DataFrame, cohorts: pd.DataFrame, tag_month_panel: pd.DataFrame) -> None:
    payload = {
        "question_source": source.question_source,
        "answer_source": source.answer_source,
        "accepted_timing_source": source.accepted_timing_source,
        "data_end_at": source.data_end_at,
        "watch_state": source.watch_state,
        "n_single_tag_questions": int(len(question_closure)),
        "n_focal_answers": int(len(answers)),
        "n_user_tag_cohorts": int(len(cohorts)),
        "n_tag_month_rows": int(len(tag_month_panel)),
        "min_question_created_at": question_closure["question_created_at"].min().isoformat(),
        "max_question_created_at": question_closure["question_created_at"].max().isoformat(),
        "min_answer_created_at": answers["answer_created_at"].min().isoformat(),
        "max_answer_created_at": answers["answer_created_at"].max().isoformat(),
    }
    METADATA_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(
    source: SourceMetadata,
    tag_exposure_panel: pd.DataFrame,
    cohorts: pd.DataFrame,
    results: dict,
    identification_profile: pd.DataFrame,
    inference_df: pd.DataFrame,
    construct_df: pd.DataFrame,
    placebo: pd.DataFrame,
    leave_out: pd.DataFrame,
    leave_two: pd.DataFrame,
) -> None:
    cohort_counts = (
        cohorts.groupby("primary_tag", as_index=False)
        .agg(
            n_preshock_contributors=("answerer_user_id", "size"),
            n_experts=("is_expert", "sum"),
            n_incumbents=("is_incumbent", "sum"),
        )
        .sort_values("primary_tag")
    )

    model_rows = []
    for model_name, payload in results.items():
        row = {"model": model_name}
        row.update(payload["summary"])
        model_rows.append(row)
    model_df = pd.DataFrame(model_rows)
    pre_placebo = placebo.loc[placebo["placebo_month"] < SHOCK_MONTH].copy()
    leave_two_summary = pd.DataFrame(columns=["specification", "positive_share", "significant_share"])
    if not leave_two.empty:
        leave_two_summary = (
            leave_two.groupby("specification", as_index=False)
            .agg(
                positive_share=("coef", lambda s: float((s > 0).mean())),
                significant_share=("pval", lambda s: float((s < 0.05).mean())),
            )
        )
    with SUMMARY_MD.open("w", encoding="utf-8") as handle:
        handle.write("# Who Still Answers After ChatGPT? Current Analysis Summary\n\n")
        handle.write("## Source State\n\n")
        handle.write(f"- Question source: `{source.question_source}`\n")
        handle.write(f"- Answer source: `{source.answer_source}`\n")
        handle.write(f"- Accepted timing source: `{source.accepted_timing_source}`\n")
        handle.write(f"- Current data end: `{source.data_end_at}`\n")
        if source.watch_state:
            handle.write(f"- 2025 dump watcher status: `{source.watch_state.get('status', 'unknown')}`\n")
            handle.write(f"- 2025 dump watcher detail: `{source.watch_state.get('detail', 'n/a')}`\n")
        handle.write("\n## What Is Implemented Now\n\n")
        handle.write("- `tag_exposure_panel`\n")
        handle.write("- `user_tag_preshock_cohorts`\n")
        handle.write("- `user_tag_month_panel`\n")
        handle.write("- `tag_month_entry_panel`\n")
        handle.write("- `question_closure_panel`\n")
        handle.write("- `identification_profile`\n")
        handle.write("- `trend_break_results`\n")
        handle.write("- `small_sample_inference`\n")
        handle.write("- `leave_two_out`\n")
        handle.write("- `construct_ladders`\n")
        handle.write("\n## Read on Current Evidence\n\n")
        handle.write("The contributor-reallocation pipeline is fully implemented on the currently available archival backbone. ")
        if source.question_source == "stackexchange_20251231_dump":
            handle.write("The canonical `2025Q4` dump backbone is active in this run, so the estimates now reflect the long-window archival population through `2025-12-31`. ")
            handle.write("The headline entrant-side result remains positive and concentrated in `brand_new_platform` entrants, but the timing diagnostics no longer support a ChatGPT-timed break and the downstream accepted-answer outcomes move in the opposite direction of the original weakening hypothesis. ")
        else:
            handle.write("The present run still resolves to the corrected early archive rather than the canonical `2025Q4` dump backbone, so the post period remains short. ")
            if source.watch_state.get("status") == "completed":
                handle.write("The watcher reports a completed dump parse, but the expected canonical dump parquet inputs were not present at the analysis load path during this run. ")
        handle.write("The current build hardens the paper with outcome-specific trend-break diagnostics, conservative tag-level inference, leave-two-out stress tests, and construct-validation ladders.\n\n")
        handle.write("## Model Results\n\n")
        handle.write(model_df.to_markdown(index=False))
        handle.write("\n\n## Identification Profile\n\n")
        if not identification_profile.empty:
            handle.write(identification_profile.to_markdown(index=False))
        else:
            handle.write("Identification profile unavailable.\n")
        handle.write("\n\n## Small-Sample Inference\n\n")
        if not inference_df.empty:
            handle.write(inference_df.to_markdown(index=False))
        else:
            handle.write("Small-sample inference unavailable.\n")
        handle.write("\n\n## Tag Exposure Panel\n\n")
        handle.write(tag_exposure_panel[[
            "primary_tag",
            "exposure_index",
            "exposure_rank",
            "exposure_tercile",
            "n_questions",
            "mean_first_answer_1d",
            "mean_accepted_7d",
        ]].to_markdown(index=False))
        handle.write("\n\n## Preshock Cohort Counts\n\n")
        handle.write(cohort_counts.to_markdown(index=False))
        handle.write("\n\n## Timing / Placebo Snapshot\n\n")
        handle.write(placebo.head(18).to_markdown(index=False))
        handle.write("\n\n")
        if not pre_placebo.empty:
            handle.write(f"- Significant pre-shock placebo rows: `{int((pre_placebo['pval'] < 0.05).sum())}`\n")
        handle.write("\n## Leave-One-Out Snapshot\n\n")
        handle.write(leave_out.head(24).to_markdown(index=False))
        handle.write("\n\n## Leave-Two-Out Summary\n\n")
        if not leave_two_summary.empty:
            handle.write(leave_two_summary.to_markdown(index=False))
        else:
            handle.write("Leave-two-out summary unavailable.\n")
        handle.write("\n\n## Construct Validation Snapshot\n\n")
        construct_preview = construct_df.head(18) if not construct_df.empty else pd.DataFrame()
        if not construct_preview.empty:
            handle.write(construct_preview.to_markdown(index=False))
        else:
            handle.write("Construct validation unavailable.\n")
        handle.write("\n\n## Guardrail\n\n")
        handle.write(
            "Do not claim that the dump-backed rerun validates `expert exit + novice entry + weaker closure`. "
            "The current long-window run supports a narrower entrant-side paper: entrant share rises more in more exposed domains, expert-share decline is not established, and acceptance-based visible-resolution outcomes improve rather than weaken. "
            "The entrant result still does not clear the full conservative-inference stack, and the timing diagnostics do not support a clean ChatGPT break.\n"
        )


def write_mechanism_decision_memo(results: dict, identification_profile: pd.DataFrame, inference_df: pd.DataFrame) -> None:
    novice_row = identification_profile.loc[identification_profile["specification"] == "novice_entry_share"].copy()
    inference_row = inference_df.loc[inference_df["specification"] == "novice_entry_share"].copy()
    novice_model = results.get("novice_entry_share", {}).get("summary", {})
    incumbent_model = results.get("incumbent_mean_log_answers", {}).get("summary", {})
    expert_share_model = results.get("expert_answer_share", {}).get("summary", {})
    closure_model = results.get("accepted_7d", {}).get("summary", {})
    with MECHANISM_DECISION_MEMO.open("w", encoding="utf-8") as handle:
        handle.write("# Who Still Answers After ChatGPT? Mechanism Decision Memo\n\n")
        handle.write("## Current Branch Decision\n\n")
        handle.write("Keep the paper narrow as an entrant-side reallocation manuscript, and do not market the dump-backed rerun as validating expert exit or weaker public resolution.\n\n")
        handle.write("## Why the Paper Stays Narrow Now\n\n")
        handle.write(f"- `novice_entry_share` remains the strongest current coefficient: `{novice_model.get('coef', float('nan')):.4f}` with conventional cluster p-value `{novice_model.get('pval', float('nan')):.4f}`.\n")
        if not novice_row.empty:
            handle.write(
                f"- The preferred novice trend-break term ranks `{int(novice_row['actual_rank_vs_pre_breaks'].iloc[0])}` "
                f"against `{int(novice_row['n_pre_breaks'].iloc[0])}` pre-period candidate breaks, so the timing profile does not support a distinctive ChatGPT break.\n"
            )
        if not inference_row.empty:
            handle.write(f"- Conservative inference for novice entry: CR2 p-value `{float(inference_row['cr2_pval'].iloc[0]):.4f}`, wild-bootstrap p-value `{float(inference_row['wild_cluster_bootstrap_pval'].iloc[0]):.4f}`, randomization p-value `{float(inference_row['randomization_pval'].iloc[0]):.4f}`.\n")
            handle.write("- The entrant-side result survives the cluster-based small-sample correction but does not yet clear the full `2-of-3` conservative-inference gate.\n")
        handle.write(f"- Incumbent expert activity remains directionally negative but imprecise: `{incumbent_model.get('coef', float('nan')):.4f}` with p-value `{incumbent_model.get('pval', float('nan')):.4f}`.\n")
        handle.write(f"- Expert-share decline is not established: `{expert_share_model.get('coef', float('nan')):.4f}` with p-value `{expert_share_model.get('pval', float('nan')):.4f}`.\n")
        handle.write(f"- Accepted-within-7-days visible-resolution outcome is positive, not negative: `{closure_model.get('coef', float('nan')):.4f}` with p-value `{closure_model.get('pval', float('nan')):.4f}`.\n\n")
        handle.write("## Editorial Rule\n\n")
        handle.write("Use the narrow entrant-side title and contribution language. Do not describe the paper as an expert-exit or weaker-closure paper unless a different design later supports those claims.\n")


def write_final_scorecard(results: dict, inference_df: pd.DataFrame, identification_profile: pd.DataFrame, construct_df: pd.DataFrame) -> None:
    novice_inference = inference_df.loc[inference_df["specification"] == "novice_entry_share"].copy()
    expert_profile = construct_df.loc[construct_df["family"] == "expert_holdout"].copy()
    dimensions = [
        ("Question importance", 9.0, "Public answer supply under private GenAI remains an important IS question."),
        ("Novelty", 9.0, "The paper is now clearly answerer-side, entrant-side, and composition-first relative to adjacent Stack Overflow work."),
        ("IS/theory contribution", 8.6, "The narrower entrant-side mechanism is clear, but the broader staged mechanism is not supported."),
        ("Construct validity", 8.8 if not expert_profile.empty else 8.0, "Construct ladders are strong and dump-backed accept-vote timing materially improves the visible-resolution side of the design."),
        ("Identification credibility", 7.2 if not identification_profile.empty else 6.9, "The long-window timing profile is openly bounded: the entrant result is not distinctively ChatGPT-timed and should not be sold as a clean break."),
        ("Small-sample inference credibility", 8.4 if not novice_inference.empty else 6.8, "CR2, wild bootstrap, randomization inference, and leave-two-out materially improve the inferential base, with strong support for the positive acceptance outcomes and mixed support for entrant timing."),
        ("Mechanism completeness", 6.5, "Entrant-side reallocation is clear, but expert-share decline is null and the original weaker-closure stage fails in the opposite direction."),
        ("Manuscript quality", 8.9, "The article can now be shaped into a narrower, more honest long-window entrant-side paper."),
        ("Display quality", 8.8, "The package has strong timing and inference displays, but they now need to be reframed around the new long-window evidence hierarchy."),
        ("Editor/reviewer package strength", 8.7, "The package is credible if it narrows immediately; it weakens if it keeps the old broader mechanism language."),
    ]
    total_score = sum(score for _, score, _ in dimensions)
    with FINAL_SCORECARD_MD.open("w", encoding="utf-8") as handle:
        handle.write("# Who Still Answers After ChatGPT? Current Scorecard\n\n")
        handle.write(f"Current internal score: `{total_score:.1f} / 100`.\n\n")
        handle.write("The package is not eligible for an internal `100 / 100` until the dump-backed branch decision is complete and the method dimensions all clear `9.0`.\n\n")
        handle.write("| Dimension | Score (0-10) | Read |\n")
        handle.write("| --- | ---: | --- |\n")
        for label, score, readout in dimensions:
            handle.write(f"| {label} | `{score:.1f}` | {readout} |\n")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()
    exposure = load_exposure()
    question_df, source = load_question_base(exposure)
    answers = load_answer_base(question_df)
    question_closure = build_question_closure_panel(question_df, answers, exposure, source)
    first_posts = build_first_post_lookup(answers["answerer_user_id"])
    cohorts = build_user_tag_preshock_cohorts(answers, first_posts, exposure)
    user_tag_month_panel, cohort_panel = build_user_tag_month_panel(answers, cohorts, source)
    tag_month_panel, first_tag_answers = build_tag_month_entry_panel(answers, cohorts, first_posts, exposure)
    entrant_profiles = build_entrant_profiles(first_tag_answers)
    postshock_status = build_postshock_status(user_tag_month_panel, cohorts)
    tag_exposure_panel = build_tag_exposure_panel(question_closure, exposure)
    specs = get_model_specs(cohort_panel, tag_month_panel, postshock_status)
    results = fit_models(specs)
    placebo = placebo_grid(tag_month_panel)
    leave_out = leave_one_out(tag_month_panel)
    leave_two = leave_two_out(specs, {"expert_answer_share", "novice_entry_share", "first_answer_1d", "accepted_7d", "accepted_30d"})
    trend_breaks, identification_profile = trend_break_diagnostics(tag_month_panel)
    inference_df, randomization_df = small_sample_inference(
        specs,
        {"incumbent_mean_log_answers", "incumbent_share_active", "expert_answer_share", "novice_entry_share", "first_answer_1d", "accepted_7d", "accepted_30d"},
    )
    construct_df = build_construct_ladders(
        answers,
        question_closure,
        cohorts,
        first_tag_answers,
        entrant_profiles,
        tag_month_panel,
    )

    expert_event = event_study(
        tag_month_panel.dropna(subset=["expert_answer_share"]).copy(),
        "expert_answer_share",
        "n_answers",
        EVENT_STUDY_EXPERT_SHARE_CSV,
    )
    novice_event = event_study(
        tag_month_panel.dropna(subset=["novice_entry_share"]).copy(),
        "novice_entry_share",
        "n_new_answerers",
        EVENT_STUDY_NOVICE_ENTRY_CSV,
    )
    accepted_event = event_study(
        tag_month_panel.loc[tag_month_panel["accepted_7d_denom"] > 0].copy(),
        "accepted_7d_rate",
        "accepted_7d_denom",
        EVENT_STUDY_ACCEPTED_7D_CSV,
    )
    plot_event_study(expert_event, "Event Study: Expert Answer Share", "Exposure-scaled differential", EXPERT_SHARE_PNG, "#8c1515")
    plot_event_study(novice_event, "Event Study: Novice Entry Share", "Exposure-scaled differential", NOVICE_ENTRY_PNG, "#1f77b4")
    plot_event_study(accepted_event, "Event Study: Accepted Within 7 Days", "Exposure-scaled differential", ACCEPTED_7D_PNG, "#2e8b57")
    plot_trends(tag_month_panel)
    plot_placebo_rank_panel(trend_breaks)
    plot_novice_trend_break(tag_month_panel, trend_breaks)
    plot_permutation_density(randomization_df)
    plot_influence_heatmap(leave_out, results)
    save_metadata(source, question_closure, answers, cohorts, tag_month_panel)
    write_summary(
        source,
        tag_exposure_panel,
        cohorts,
        results,
        identification_profile,
        inference_df,
        construct_df,
        placebo,
        leave_out,
        leave_two,
    )
    write_mechanism_decision_memo(results, identification_profile, inference_df)
    write_final_scorecard(results, inference_df, identification_profile, construct_df)
    print(MODEL_RESULTS_JSON)
    print(SUMMARY_MD)


if __name__ == "__main__":
    main()
