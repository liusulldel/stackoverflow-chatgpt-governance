from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
FIGURES_DIR = BASE_DIR / "figures"

POSTS_PARQUET = RAW_DIR / "stackoverflow_2023_05_posts.parquet"
EXTENDED_SAMPLE_PARQUET = PROCESSED_DIR / "strengthened_question_latency_sample.parquet"
EXTENDED_SAMPLE_CSV = PROCESSED_DIR / "strengthened_question_latency_sample.csv"
PRIMARY_PANEL_CSV = PROCESSED_DIR / "strengthened_primary_panel.csv"
FRACTIONAL_PANEL_CSV = PROCESSED_DIR / "strengthened_fractional_panel.csv"
EXPOSURE_TAG_CSV = PROCESSED_DIR / "strengthened_exposure_tag_scores.csv"
RESULTS_JSON = PROCESSED_DIR / "strengthened_results.json"
SUMMARY_MD = PROCESSED_DIR / "strengthened_summary.md"
EVENT_STUDY_CSV = PROCESSED_DIR / "strengthened_event_study_accepted_7d.csv"
PLACEBO_GRID_CSV = PROCESSED_DIR / "strengthened_placebo_grid_accepted_7d.csv"
ARCHIVE_META_JSON = PROCESSED_DIR / "strengthened_archive_metadata.json"
EVENT_STUDY_PNG = FIGURES_DIR / "strengthened_event_study_accepted_7d.png"

START_DATE = "2020-01-01"
SHOCK_DATE = pd.Timestamp("2022-11-30T00:00:00Z")
SHOCK_MONTH = "2022-12"
PRE_EXPOSURE_END = "2022-11"

SELECTED_TAGS = [
    "apache-spark",
    "android",
    "bash",
    "docker",
    "excel",
    "firebase",
    "kubernetes",
    "linux",
    "memory-management",
    "multithreading",
    "numpy",
    "pandas",
    "python",
    "regex",
    "javascript",
    "sql",
]

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

ROUTINE_TERMS = [
    "how to",
    "convert",
    "parse",
    "replace",
    "split",
    "sort",
    "merge",
    "join",
    "filter",
    "extract",
    "regex",
    "formula",
    "query",
    "dataframe",
    "array",
    "list",
    "string",
    "json",
    "csv",
    "xml",
    "datetime",
    "date format",
    "round",
    "group by",
    "select",
]

CONTEXT_TERMS = [
    "version",
    "install",
    "deployment",
    "deploy",
    "configure",
    "configuration",
    "environment",
    "runtime",
    "thread",
    "memory",
    "device",
    "emulator",
    "sdk",
    "cluster",
    "container",
    "pod",
    "service",
    "server",
    "kernel",
    "permission",
    "socket",
    "network",
    "gradle",
    "ubuntu",
    "android",
    "docker",
    "kubernetes",
    "firebase",
]

ROUTINE_REGEX = re.compile("|".join(re.escape(term) for term in ROUTINE_TERMS), flags=re.IGNORECASE)
CONTEXT_REGEX = re.compile("|".join(re.escape(term) for term in CONTEXT_TERMS), flags=re.IGNORECASE)
VERSION_REGEX = re.compile(r"\b\d+(?:\.\d+){1,}\b")
CONFIG_REGEX = re.compile(r"\b(?:config|settings|yaml|json file|dockerfile|gradle|manifest|permission)\b", flags=re.IGNORECASE)

WINDOWS = {
    "first_answer_1d": 24.0,
    "first_answer_7d": 24.0 * 7.0,
    "accepted_7d": 24.0 * 7.0,
    "accepted_30d": 24.0 * 30.0,
}


@dataclass
class ArchiveMetadata:
    archive_cutoff_iso: str
    archive_cutoff_for_7d_iso: str
    archive_cutoff_for_30d_iso: str
    archive_cutoff_for_1d_iso: str
    rows: int


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def tag_condition(tags: Iterable[str]) -> str:
    parts = []
    for tag in tags:
        escaped = tag.replace("'", "''")
        parts.append(f"Tags LIKE '%<{escaped}>%'")
    return " OR ".join(parts)


def extract_extended_sample() -> ArchiveMetadata:
    con = duckdb.connect()
    archive_cutoff = con.execute(
        f"SELECT max(CreationDate) FROM read_parquet('{POSTS_PARQUET.as_posix()}')"
    ).fetchone()[0]

    condition = tag_condition(SELECTED_TAGS)
    query = f"""
        WITH questions AS (
            SELECT
                Id AS question_id,
                CreationDate AS question_created_at,
                AcceptedAnswerId AS accepted_answer_id,
                Score AS score,
                ViewCount AS view_count,
                AnswerCount AS answer_count_archive,
                OwnerUserId AS owner_user_id,
                Title AS title,
                regexp_replace(regexp_replace(Tags, '^<|>$', '', 'g'), '><', ';', 'g') AS question_tags
            FROM read_parquet('{POSTS_PARQUET.as_posix()}')
            WHERE PostTypeId = 1
              AND CreationDate >= TIMESTAMP '{START_DATE}'
              AND ({condition})
        ),
        answers AS (
            SELECT
                Id AS answer_id,
                ParentId AS question_id,
                CreationDate AS answer_created_at
            FROM read_parquet('{POSTS_PARQUET.as_posix()}')
            WHERE PostTypeId = 2
        ),
        first_answers AS (
            SELECT
                question_id,
                min(answer_created_at) AS first_answer_at
            FROM answers
            GROUP BY 1
        )
        SELECT
            q.question_id,
            q.question_created_at,
            q.accepted_answer_id,
            q.score,
            q.view_count,
            q.answer_count_archive,
            q.owner_user_id,
            q.title,
            q.question_tags,
            fa.first_answer_at,
            aa.answer_created_at AS accepted_answer_at
        FROM questions q
        LEFT JOIN first_answers fa
          ON q.question_id = fa.question_id
        LEFT JOIN answers aa
          ON q.accepted_answer_id = aa.answer_id
    """

    con.execute(f"COPY ({query}) TO '{EXTENDED_SAMPLE_PARQUET.as_posix()}' (FORMAT PARQUET)")
    con.execute(f"COPY ({query}) TO '{EXTENDED_SAMPLE_CSV.as_posix()}' (HEADER, DELIMITER ',')")
    row_count = con.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
    con.close()

    cutoff = pd.Timestamp(archive_cutoff, tz="UTC")
    metadata = ArchiveMetadata(
        archive_cutoff_iso=cutoff.isoformat(),
        archive_cutoff_for_1d_iso=(cutoff - pd.Timedelta(days=1)).isoformat(),
        archive_cutoff_for_7d_iso=(cutoff - pd.Timedelta(days=7)).isoformat(),
        archive_cutoff_for_30d_iso=(cutoff - pd.Timedelta(days=30)).isoformat(),
        rows=int(row_count),
    )
    ARCHIVE_META_JSON.write_text(json.dumps(metadata.__dict__, indent=2), encoding="utf-8")
    return metadata


def parse_selected_tags(tag_string: str) -> list[str]:
    if pd.isna(tag_string):
        return []
    return [tag for tag in str(tag_string).split(";") if tag in SELECTED_TAGS]


def load_and_prepare_questions(archive_cutoff_iso: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(EXTENDED_SAMPLE_PARQUET)
    df["question_created_at"] = pd.to_datetime(df["question_created_at"], utc=True)
    for col in ["first_answer_at", "accepted_answer_at"]:
        df[col] = pd.to_datetime(df[col], utc=True)

    archive_cutoff = pd.Timestamp(archive_cutoff_iso)
    df["followup_hours"] = (archive_cutoff - df["question_created_at"]).dt.total_seconds() / 3600.0
    df["month_id"] = df["question_created_at"].dt.strftime("%Y-%m")
    df["question_tags_list"] = df["question_tags"].apply(parse_selected_tags)
    df["selected_tags_in_order"] = df["question_tags_list"]
    df["selected_tag_overlap"] = df["selected_tags_in_order"].str.len()
    df["keep_single_focal"] = (df["selected_tag_overlap"] == 1).astype(int)

    df["first_answer_event"] = df["first_answer_at"].notna().astype(int)
    df["accepted_event"] = df["accepted_answer_at"].notna().astype(int)
    df["first_answer_hours"] = np.where(
        df["first_answer_event"] == 1,
        (df["first_answer_at"] - df["question_created_at"]).dt.total_seconds() / 3600.0,
        df["followup_hours"],
    )
    df["accepted_hours"] = np.where(
        df["accepted_event"] == 1,
        (df["accepted_answer_at"] - df["question_created_at"]).dt.total_seconds() / 3600.0,
        df["followup_hours"],
    )

    title_lower = df["title"].fillna("").astype(str).str.lower()
    df["routine_hits"] = title_lower.str.count(ROUTINE_REGEX)
    df["context_hits"] = title_lower.str.count(CONTEXT_REGEX)
    df["version_hits"] = title_lower.str.count(VERSION_REGEX)
    df["config_hits"] = title_lower.str.count(CONFIG_REGEX)
    df["question_exposure_raw"] = (
        df["routine_hits"]
        - df["context_hits"]
        - 0.5 * df["version_hits"]
        - 0.5 * df["config_hits"]
    )

    for outcome, hours in WINDOWS.items():
        eligible_col = f"{outcome}_eligible"
        event_col = f"{outcome}_event"
        if outcome.startswith("first_answer"):
            base_hours = df["first_answer_hours"]
            base_event = df["first_answer_event"]
        else:
            base_hours = df["accepted_hours"]
            base_event = df["accepted_event"]
        df[eligible_col] = (df["followup_hours"] >= hours).astype(int)
        df[event_col] = ((base_event == 1) & (base_hours <= hours) & (df[eligible_col] == 1)).astype(int)

    primary = df.loc[df["keep_single_focal"] == 1].copy()
    primary["tag"] = primary["selected_tags_in_order"].str[0]
    primary["question_weight"] = 1.0

    def add_common_fields(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["group"] = np.where(out["tag"].isin(HIGH_EXPOSURE_TAGS), "high", "low")
        out["high_tag"] = (out["group"] == "high").astype(int)
        out["post_chatgpt"] = (out["month_id"] >= SHOCK_MONTH).astype(int)
        out["high_post"] = out["high_tag"] * out["post_chatgpt"]
        out["log_questions_weight"] = out["question_weight"]
        return out

    primary = add_common_fields(primary)

    repeated_index = np.repeat(df.index.to_numpy(), df["selected_tag_overlap"].clip(lower=0).to_numpy())
    fractional = df.loc[repeated_index].copy()
    fractional = fractional.loc[fractional["selected_tag_overlap"] >= 1].copy()
    fractional["tag"] = [tag for tags in df.loc[df["selected_tag_overlap"] >= 1, "selected_tags_in_order"] for tag in tags]
    fractional["question_weight"] = 1.0 / fractional["selected_tag_overlap"]
    fractional = add_common_fields(fractional)
    return df, primary, fractional


def build_interpretable_exposure(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pre = primary.loc[primary["month_id"] <= PRE_EXPOSURE_END].copy()
    exposure = (
        pre.groupby("tag", as_index=False)
        .agg(
            pre_questions=("question_id", "count"),
            mean_exposure_raw=("question_exposure_raw", "mean"),
            mean_routine_hits=("routine_hits", "mean"),
            mean_context_hits=("context_hits", "mean"),
            mean_version_hits=("version_hits", "mean"),
            mean_config_hits=("config_hits", "mean"),
        )
        .sort_values("tag")
        .reset_index(drop=True)
    )
    exposure["manual_high"] = exposure["tag"].isin(HIGH_EXPOSURE_TAGS).astype(int)
    exposure_values = exposure["mean_exposure_raw"].to_numpy(dtype=float)
    if exposure.loc[exposure["manual_high"] == 1, "mean_exposure_raw"].mean() < exposure.loc[exposure["manual_high"] == 0, "mean_exposure_raw"].mean():
        exposure_values = -exposure_values
    exposure["exposure_index"] = (exposure_values - exposure_values.mean()) / exposure_values.std(ddof=0)
    exposure["exposure_rank"] = exposure["exposure_index"].rank(ascending=False, method="first").astype(int)
    exposure["exposure_tercile"] = pd.qcut(exposure["exposure_index"], q=3, labels=["Low", "Middle", "High"], duplicates="drop")

    diagnostics = {
        "mean_exposure_high_tags": float(exposure.loc[exposure["manual_high"] == 1, "exposure_index"].mean()),
        "mean_exposure_low_tags": float(exposure.loc[exposure["manual_high"] == 0, "exposure_index"].mean()),
        "correlation_with_manual_high": float(np.corrcoef(exposure["manual_high"], exposure["exposure_index"])[0, 1]),
        "top_exposure_tags": exposure.sort_values("exposure_index", ascending=False).head(5)[["tag", "exposure_index"]].to_dict(orient="records"),
        "bottom_exposure_tags": exposure.sort_values("exposure_index", ascending=True).head(5)[["tag", "exposure_index"]].to_dict(orient="records"),
    }
    exposure.to_csv(EXPOSURE_TAG_CSV, index=False)
    return exposure, diagnostics


def attach_exposure(frame: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(exposure[["tag", "exposure_index", "exposure_rank", "exposure_tercile"]], on="tag", how="left")
    merged["exposure_post"] = merged["exposure_index"] * merged["post_chatgpt"]
    return merged


def build_monthly_panel(frame: pd.DataFrame) -> pd.DataFrame:
    month_order = {month: idx + 1 for idx, month in enumerate(sorted(frame["month_id"].unique()))}
    shock_period = pd.Period(SHOCK_MONTH)

    def summarize(g: pd.DataFrame) -> pd.Series:
        def rate(event_col: str, eligible_col: str) -> tuple[float, float]:
            denom = float((g["question_weight"] * g[eligible_col]).sum())
            if denom <= 0:
                return np.nan, 0.0
            numerator = float((g["question_weight"] * g[event_col]).sum())
            return numerator / denom, denom

        first_1d_rate, first_1d_denom = rate("first_answer_1d_event", "first_answer_1d_eligible")
        first_7d_rate, first_7d_denom = rate("first_answer_7d_event", "first_answer_7d_eligible")
        acc_7d_rate, acc_7d_denom = rate("accepted_7d_event", "accepted_7d_eligible")
        acc_30d_rate, acc_30d_denom = rate("accepted_30d_event", "accepted_30d_eligible")
        accepted_archive_rate = float(np.average(g["accepted_event"], weights=g["question_weight"]))
        return pd.Series(
            {
                "n_questions": float(g["question_weight"].sum()),
                "exposure_index": float(np.average(g["exposure_index"], weights=g["question_weight"])) if "exposure_index" in g else np.nan,
                "accepted_archive_rate": accepted_archive_rate,
                "first_answer_1d_rate": first_1d_rate,
                "first_answer_1d_denom": first_1d_denom,
                "first_answer_7d_rate": first_7d_rate,
                "first_answer_7d_denom": first_7d_denom,
                "accepted_7d_rate": acc_7d_rate,
                "accepted_7d_denom": acc_7d_denom,
                "accepted_30d_rate": acc_30d_rate,
                "accepted_30d_denom": acc_30d_denom,
            }
        )

    monthly = (
        frame.groupby(["tag", "group", "month_id", "high_tag", "post_chatgpt"], as_index=False)
        .apply(summarize)
        .reset_index(drop=True)
    )
    monthly["high_post"] = monthly["high_tag"] * monthly["post_chatgpt"]
    monthly["exposure_post"] = monthly["exposure_index"] * monthly["post_chatgpt"]
    monthly["time_index"] = monthly["month_id"].map(month_order)
    monthly["rel_month"] = monthly["month_id"].apply(
        lambda m: (pd.Period(m).year - shock_period.year) * 12 + (pd.Period(m).month - shock_period.month)
    )
    return monthly


def fit_weighted(formula: str, data: pd.DataFrame, weight_col: str):
    return smf.wls(formula, data=data, weights=data[weight_col]).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["tag"]},
    )


def fit_plain(formula: str, data: pd.DataFrame, weight_col: str):
    return smf.wls(formula, data=data, weights=data[weight_col]).fit()


def extract_term(model, term: str) -> dict:
    return {
        "coef": float(model.params.get(term, np.nan)),
        "se": float(model.bse.get(term, np.nan)),
        "pval": float(model.pvalues.get(term, np.nan)),
        "nobs": float(model.nobs),
        "r2": float(getattr(model, "rsquared", np.nan)),
    }


def fit_panel_suite(primary_panel: pd.DataFrame, fractional_panel: pd.DataFrame) -> dict:
    specs = [
        ("archive_primary_binary", primary_panel, "accepted_archive_rate", "n_questions", "high_post"),
        ("archive_primary_continuous", primary_panel, "accepted_archive_rate", "n_questions", "exposure_post"),
        ("accepted_7d_primary_binary", primary_panel.loc[primary_panel["accepted_7d_denom"] > 0].copy(), "accepted_7d_rate", "accepted_7d_denom", "high_post"),
        ("accepted_7d_primary_continuous", primary_panel.loc[primary_panel["accepted_7d_denom"] > 0].copy(), "accepted_7d_rate", "accepted_7d_denom", "exposure_post"),
        ("accepted_30d_primary_binary", primary_panel.loc[primary_panel["accepted_30d_denom"] > 0].copy(), "accepted_30d_rate", "accepted_30d_denom", "high_post"),
        ("accepted_30d_primary_continuous", primary_panel.loc[primary_panel["accepted_30d_denom"] > 0].copy(), "accepted_30d_rate", "accepted_30d_denom", "exposure_post"),
        ("first_answer_1d_primary_binary", primary_panel.loc[primary_panel["first_answer_1d_denom"] > 0].copy(), "first_answer_1d_rate", "first_answer_1d_denom", "high_post"),
        ("accepted_7d_fractional_binary", fractional_panel.loc[fractional_panel["accepted_7d_denom"] > 0].copy(), "accepted_7d_rate", "accepted_7d_denom", "high_post"),
        ("accepted_30d_fractional_binary", fractional_panel.loc[fractional_panel["accepted_30d_denom"] > 0].copy(), "accepted_30d_rate", "accepted_30d_denom", "high_post"),
    ]
    results = {}
    for name, data, outcome, weight_col, term in specs:
        formula = f"{outcome} ~ {term} + C(tag):time_index + C(tag) + C(month_id)"
        model = fit_weighted(formula, data, weight_col)
        results[name] = {
            "outcome": outcome,
            "weight_col": weight_col,
            "term": term,
            "summary": extract_term(model, term),
            "formula": formula,
        }
    return results


def event_study(primary_panel: pd.DataFrame) -> pd.DataFrame:
    panel = primary_panel.loc[primary_panel["accepted_7d_denom"] > 0].copy()
    rel = panel["rel_month"].astype(int)
    rel = rel.clip(lower=-18, upper=2)
    rel = rel.where(rel != -1, other=-1)
    panel["rel_month_binned"] = rel.astype(int).astype(str)
    model = fit_weighted(
        "accepted_7d_rate ~ C(tag):time_index + C(tag) + C(month_id) + high_tag:C(rel_month_binned, Treatment(reference='-1'))",
        panel,
        "accepted_7d_denom",
    )
    rows = []
    for key, coef in model.params.items():
        if "high_tag:C(rel_month_binned" not in key:
            continue
        match = re.search(r"\[([-\d]+)\]", key)
        if not match:
            continue
        rel_month = int(match.group(1))
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
    out.to_csv(EVENT_STUDY_CSV, index=False)
    return out


def placebo_grid(primary_panel: pd.DataFrame) -> pd.DataFrame:
    panel = primary_panel.loc[primary_panel["accepted_7d_denom"] > 0].copy()
    months = sorted(panel["month_id"].unique())
    rows = []
    for placebo_month in months[3:-2]:
        temp = panel.copy()
        temp["placebo_post"] = (temp["month_id"] >= placebo_month).astype(int)
        model = fit_weighted(
            "accepted_7d_rate ~ high_tag:placebo_post + C(tag):time_index + C(tag) + C(month_id)",
            temp,
            "accepted_7d_denom",
        )
        rows.append(
            {
                "placebo_month": placebo_month,
                "coef": float(model.params.get("high_tag:placebo_post", np.nan)),
                "se": float(model.bse.get("high_tag:placebo_post", np.nan)),
                "pval": float(model.pvalues.get("high_tag:placebo_post", np.nan)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(PLACEBO_GRID_CSV, index=False)
    return out


def plot_event_study(event_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8.6, 4.8))
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(-0.5, color="gray", linestyle="--", linewidth=1)
    plt.errorbar(
        event_df["rel_month"],
        event_df["coef"],
        yerr=1.96 * event_df["se"],
        fmt="o-",
        color="#8C1515",
        ecolor="#c76b6b",
        capsize=3,
        linewidth=1.5,
    )
    plt.title("Event Study: Accepted Within 7 Days")
    plt.xlabel("Months Relative to December 2022")
    plt.ylabel("High-Exposure Differential")
    plt.tight_layout()
    plt.savefig(EVENT_STUDY_PNG, dpi=220)
    plt.close()


def save_summary(
    metadata: ArchiveMetadata,
    exposure: pd.DataFrame,
    exposure_diagnostics: dict,
    results: dict,
    event_df: pd.DataFrame,
    placebo_df: pd.DataFrame,
) -> None:
    payload = {
        "archive_metadata": metadata.__dict__,
        "exposure_diagnostics": exposure_diagnostics,
        "model_results": results,
        "event_study_summary": {
            "n_coefficients": int(len(event_df)),
            "pre_period_significant_months": event_df.loc[(event_df["rel_month"] <= -2) & (event_df["pval"] < 0.05), "rel_month"].tolist(),
            "post_period_coefficients": event_df.loc[event_df["rel_month"] >= 0, ["rel_month", "coef", "pval"]].to_dict(orient="records"),
        },
        "placebo_grid_summary": {
            "n_months": int(len(placebo_df)),
            "pre_shock_significant_months": placebo_df.loc[(placebo_df["placebo_month"] < SHOCK_MONTH) & (placebo_df["pval"] < 0.05), "placebo_month"].tolist(),
            "min_pre_shock_pval": float(placebo_df.loc[placebo_df["placebo_month"] < SHOCK_MONTH, "pval"].min()),
        },
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for name, spec in results.items():
        row = {"model": name}
        row.update({k: v for k, v in spec["summary"].items() if k in {"coef", "se", "pval", "nobs", "r2"}})
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    top_exposure = exposure.sort_values("exposure_index", ascending=False)[["tag", "exposure_index"]]
    placebo_pre = placebo_df.loc[placebo_df["placebo_month"] < SHOCK_MONTH].copy()

    with SUMMARY_MD.open("w", encoding="utf-8") as handle:
        handle.write("# Strengthening Summary\n\n")
        handle.write("## Archive Cutoff\n\n")
        handle.write(f"- Exact archive cutoff: `{metadata.archive_cutoff_iso}`\n")
        handle.write(f"- Extended selected-tag questions: `{metadata.rows}`\n\n")
        handle.write("## Step 1: Interpretable Continuous Exposure\n\n")
        handle.write(f"- Mean exposure for manual high tags: `{exposure_diagnostics['mean_exposure_high_tags']:.3f}`\n")
        handle.write(f"- Mean exposure for manual low tags: `{exposure_diagnostics['mean_exposure_low_tags']:.3f}`\n")
        handle.write(f"- Correlation with manual high split: `{exposure_diagnostics['correlation_with_manual_high']:.3f}`\n\n")
        handle.write(top_exposure.to_markdown(index=False))
        handle.write("\n\n## Step 2: Latency / Windowed Closure Models\n\n")
        handle.write(summary_df.to_markdown(index=False))
        handle.write("\n\n## Step 3: Timing Redesign\n\n")
        handle.write("### Event Study on Accepted Within 7 Days\n\n")
        handle.write(event_df.to_markdown(index=False))
        handle.write("\n\n### Dense Placebo Grid on Accepted Within 7 Days\n\n")
        handle.write(placebo_df.to_markdown(index=False))
        handle.write("\n\n### Timing Read\n\n")
        handle.write(
            f"- Significant pre-shock placebo months: `{', '.join(placebo_pre.loc[placebo_pre['pval'] < 0.05, 'placebo_month']) if not placebo_pre.loc[placebo_pre['pval'] < 0.05].empty else 'none'}`\n"
        )


def main() -> None:
    ensure_dirs()
    metadata = extract_extended_sample()
    raw_df, primary_questions, fractional_questions = load_and_prepare_questions(metadata.archive_cutoff_iso)
    exposure, exposure_diagnostics = build_interpretable_exposure(primary_questions)

    primary_questions = attach_exposure(primary_questions, exposure)
    fractional_questions = attach_exposure(fractional_questions, exposure)

    primary_panel = build_monthly_panel(primary_questions)
    fractional_panel = build_monthly_panel(fractional_questions)
    primary_panel.to_csv(PRIMARY_PANEL_CSV, index=False)
    fractional_panel.to_csv(FRACTIONAL_PANEL_CSV, index=False)

    results = fit_panel_suite(primary_panel, fractional_panel)
    event_df = event_study(primary_panel)
    placebo_df = placebo_grid(primary_panel)
    plot_event_study(event_df)
    save_summary(metadata, exposure, exposure_diagnostics, results, event_df, placebo_df)


if __name__ == "__main__":
    main()
