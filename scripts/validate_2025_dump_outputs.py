from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw" / "stackexchange_20251231"
PROCESSED_DIR = BASE_DIR / "processed"

MANIFEST_JSON = RAW_DIR / "stackexchange_20251231_manifest.json"
FOCAL_SUMMARY_JSON = PROCESSED_DIR / "stackexchange_20251231_focal_summary.json"
PANEL_SUMMARY_JSON = PROCESSED_DIR / "stackexchange_20251231_panel_summary.json"
QUESTIONS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_questions.parquet"
ANSWERS_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_answers.parquet"
ACCEPT_VOTES_PARQUET = PROCESSED_DIR / "stackexchange_20251231_focal_accept_votes.parquet"
QUESTION_LEVEL_PARQUET = PROCESSED_DIR / "stackexchange_20251231_question_level_enriched.parquet"
PRIMARY_PANEL_CSV = PROCESSED_DIR / "stackexchange_20251231_primary_panel.csv"
FRACTIONAL_PANEL_CSV = PROCESSED_DIR / "stackexchange_20251231_fractional_panel.csv"

REPORT_JSON = PROCESSED_DIR / "stackexchange_20251231_validation_report.json"
REPORT_MD = PROCESSED_DIR / "stackexchange_20251231_validation_report.md"

SELECTED_TAGS = {
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
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def add_result(results: list[CheckResult], name: str, condition: bool, success: str, failure: str) -> None:
    results.append(CheckResult(name=name, status="pass" if condition else "fail", detail=success if condition else failure))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_reports(results: list[CheckResult], metadata: dict) -> None:
    payload = {
        "metadata": metadata,
        "checks": [asdict(result) for result in results],
        "n_pass": sum(result.status == "pass" for result in results),
        "n_fail": sum(result.status == "fail" for result in results),
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# stackexchange_20251231 Validation Report", ""]
    lines.append("## Metadata")
    for key, value in metadata.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Checks")
    for result in results:
        marker = "PASS" if result.status == "pass" else "FAIL"
        lines.append(f"- `{marker}` {result.name}: {result.detail}")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results: list[CheckResult] = []
    required_paths = [
        MANIFEST_JSON,
        FOCAL_SUMMARY_JSON,
        PANEL_SUMMARY_JSON,
        QUESTIONS_PARQUET,
        ANSWERS_PARQUET,
        ACCEPT_VOTES_PARQUET,
        QUESTION_LEVEL_PARQUET,
        PRIMARY_PANEL_CSV,
        FRACTIONAL_PANEL_CSV,
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required outputs: " + "; ".join(missing))

    manifest = read_json(MANIFEST_JSON)
    focal_summary = read_json(FOCAL_SUMMARY_JSON)
    panel_summary = read_json(PANEL_SUMMARY_JSON)

    questions = pd.read_parquet(QUESTIONS_PARQUET)
    answers = pd.read_parquet(ANSWERS_PARQUET)
    accept_votes = pd.read_parquet(ACCEPT_VOTES_PARQUET)
    question_level = pd.read_parquet(QUESTION_LEVEL_PARQUET)
    primary_panel = pd.read_csv(PRIMARY_PANEL_CSV)
    fractional_panel = pd.read_csv(FRACTIONAL_PANEL_CSV)

    question_level["question_created_at"] = pd.to_datetime(question_level["question_created_at"], utc=True)
    if "observation_cutoff_at" in question_level.columns:
        question_level["observation_cutoff_at"] = pd.to_datetime(question_level["observation_cutoff_at"], utc=True)

    min_created = question_level["question_created_at"].min()
    max_created = question_level["question_created_at"].max()
    observation_cutoff = question_level["observation_cutoff_at"].max() if "observation_cutoff_at" in question_level.columns else max_created
    observed_tags = set(primary_panel["tag"].dropna().astype(str))
    primary_duplicates = int(primary_panel.duplicated(["tag", "month_id"]).sum())
    fractional_duplicates = int(fractional_panel.duplicated(["tag", "month_id"]).sum())

    metadata = {
        "n_questions": len(questions),
        "n_answers": len(answers),
        "n_accept_votes": len(accept_votes),
        "n_question_level_rows": len(question_level),
        "n_primary_panel_rows": len(primary_panel),
        "n_fractional_panel_rows": len(fractional_panel),
        "min_question_created_at": min_created.isoformat(),
        "max_question_created_at": max_created.isoformat(),
        "observation_cutoff_at": observation_cutoff.isoformat(),
        "n_primary_tags": len(observed_tags),
    }

    add_result(
        results,
        "manifest_has_summary",
        manifest.get("summary") is not None,
        "Manifest includes a non-null summary payload.",
        "Manifest summary is null.",
    )
    add_result(
        results,
        "date_range_starts_in_2020",
        min_created >= pd.Timestamp("2020-01-01T00:00:00Z"),
        f"Question-level min date is {min_created.isoformat()}.",
        f"Question-level min date is too early: {min_created.isoformat()}.",
    )
    add_result(
        results,
        "date_range_reaches_2025",
        max_created >= pd.Timestamp("2025-12-01T00:00:00Z"),
        f"Question-level max date reaches {max_created.isoformat()}.",
        f"Question-level max date does not reach late 2025: {max_created.isoformat()}.",
    )
    add_result(
        results,
        "question_ids_unique",
        question_level["question_id"].is_unique,
        "Question-level data have unique question_id rows.",
        "Question-level data contain duplicate question_id rows.",
    )
    add_result(
        results,
        "primary_panel_unique_tag_month",
        primary_duplicates == 0,
        "Primary panel has unique tag-month rows.",
        f"Primary panel has {primary_duplicates} duplicate tag-month rows.",
    )
    add_result(
        results,
        "fractional_panel_unique_tag_month",
        fractional_duplicates == 0,
        "Fractional panel has unique tag-month rows.",
        f"Fractional panel has {fractional_duplicates} duplicate tag-month rows.",
    )
    add_result(
        results,
        "tag_set_matches_selected_tags",
        observed_tags == SELECTED_TAGS,
        f"Observed tag set matches the 16 selected tags.",
        f"Observed tags differ from the selected set: {sorted(observed_tags.symmetric_difference(SELECTED_TAGS))}.",
    )
    add_result(
        results,
        "panel_summary_has_observation_cutoff",
        "observation_cutoff" in panel_summary,
        f"Panel summary includes observation_cutoff={panel_summary.get('observation_cutoff')}.",
        "Panel summary is missing observation_cutoff.",
    )
    add_result(
        results,
        "accepted_7d_le_accepted_30d_question_level",
        bool((question_level["accepted_7d"] <= question_level["accepted_30d"]).all()),
        "Question-level accepted_7d never exceeds accepted_30d.",
        "Question-level accepted_7d exceeds accepted_30d for at least one row.",
    )
    add_result(
        results,
        "first_answer_1d_le_first_answer_7d_question_level",
        bool((question_level["first_answer_1d"] <= question_level["first_answer_7d"]).all()),
        "Question-level first_answer_1d never exceeds first_answer_7d.",
        "Question-level first_answer_1d exceeds first_answer_7d for at least one row.",
    )
    add_result(
        results,
        "accepted_7d_le_accepted_30d_primary_panel",
        bool(
            (
                primary_panel.loc[
                    primary_panel["accepted_7d_denom"] == primary_panel["accepted_30d_denom"],
                    "accepted_7d_rate",
                ]
                <= primary_panel.loc[
                    primary_panel["accepted_7d_denom"] == primary_panel["accepted_30d_denom"],
                    "accepted_30d_rate",
                ]
            ).all()
        ),
        "Primary panel preserves accepted_7d_rate <= accepted_30d_rate whenever the two rates share the same risk set.",
        "Primary panel violates accepted_7d_rate <= accepted_30d_rate within at least one equal-risk-set row.",
    )
    add_result(
        results,
        "first_answer_1d_le_first_answer_7d_primary_panel",
        bool(
            (
                primary_panel.loc[
                    primary_panel["first_answer_1d_denom"] == primary_panel["first_answer_7d_denom"],
                    "first_answer_1d_rate",
                ]
                <= primary_panel.loc[
                    primary_panel["first_answer_1d_denom"] == primary_panel["first_answer_7d_denom"],
                    "first_answer_7d_rate",
                ]
            ).all()
        ),
        "Primary panel preserves first_answer_1d_rate <= first_answer_7d_rate whenever the two rates share the same risk set.",
        "Primary panel violates first_answer_1d_rate <= first_answer_7d_rate within at least one equal-risk-set row.",
    )
    add_result(
        results,
        "accepted_7d_le_accepted_30d_fractional_panel",
        bool(
            (
                fractional_panel.loc[
                    fractional_panel["accepted_7d_denom"] == fractional_panel["accepted_30d_denom"],
                    "accepted_7d_rate",
                ]
                <= fractional_panel.loc[
                    fractional_panel["accepted_7d_denom"] == fractional_panel["accepted_30d_denom"],
                    "accepted_30d_rate",
                ]
            ).all()
        ),
        "Fractional panel preserves accepted_7d_rate <= accepted_30d_rate whenever the two rates share the same risk set.",
        "Fractional panel violates accepted_7d_rate <= accepted_30d_rate within at least one equal-risk-set row.",
    )
    add_result(
        results,
        "first_answer_1d_le_first_answer_7d_fractional_panel",
        bool(
            (
                fractional_panel.loc[
                    fractional_panel["first_answer_1d_denom"] == fractional_panel["first_answer_7d_denom"],
                    "first_answer_1d_rate",
                ]
                <= fractional_panel.loc[
                    fractional_panel["first_answer_1d_denom"] == fractional_panel["first_answer_7d_denom"],
                    "first_answer_7d_rate",
                ]
            ).all()
        ),
        "Fractional panel preserves first_answer_1d_rate <= first_answer_7d_rate whenever the two rates share the same risk set.",
        "Fractional panel violates first_answer_1d_rate <= first_answer_7d_rate within at least one equal-risk-set row.",
    )
    add_result(
        results,
        "primary_panel_has_denominators",
        all(column in primary_panel.columns for column in ["accepted_7d_denom", "accepted_30d_denom", "first_answer_1d_denom", "first_answer_7d_denom"]),
        "Primary panel includes denominator columns for eligibility-aware short-horizon rates.",
        "Primary panel is missing one or more denominator columns.",
    )
    add_result(
        results,
        "fractional_panel_has_denominators",
        all(column in fractional_panel.columns for column in ["accepted_7d_denom", "accepted_30d_denom", "first_answer_1d_denom", "first_answer_7d_denom"]),
        "Fractional panel includes denominator columns for eligibility-aware short-horizon rates.",
        "Fractional panel is missing one or more denominator columns.",
    )
    add_result(
        results,
        "question_level_has_eligibility_flags",
        all(column in question_level.columns for column in ["accepted_7d_eligible", "accepted_30d_eligible", "first_answer_1d_eligible", "first_answer_7d_eligible"]),
        "Question-level data include eligibility flags for all short-horizon outcomes.",
        "Question-level data are missing one or more eligibility flags.",
    )
    add_result(
        results,
        "eligibility_flags_respect_cutoff",
        bool(
            (question_level.loc[question_level["accepted_7d_eligible"] == 1, "question_created_at"] <= observation_cutoff - pd.Timedelta(days=7)).all()
            and (question_level.loc[question_level["accepted_30d_eligible"] == 1, "question_created_at"] <= observation_cutoff - pd.Timedelta(days=30)).all()
            and (question_level.loc[question_level["first_answer_1d_eligible"] == 1, "question_created_at"] <= observation_cutoff - pd.Timedelta(days=1)).all()
            and (question_level.loc[question_level["first_answer_7d_eligible"] == 1, "question_created_at"] <= observation_cutoff - pd.Timedelta(days=7)).all()
        ),
        "Eligibility flags respect the observation cutoff for all short-horizon outcomes.",
        "At least one eligibility flag is inconsistent with the observation cutoff.",
    )
    add_result(
        results,
        "late_2025_primary_denominators_drop_below_total",
        bool(
            ((primary_panel["accepted_7d_denom"] < primary_panel["n_questions"]) | (primary_panel["accepted_30d_denom"] < primary_panel["n_questions"])).any()
        ),
        "Late-window censoring is visible in the primary panel denominators.",
        "Primary panel denominators never fall below total questions; censoring guard may be missing.",
    )
    add_result(
        results,
        "focal_summary_matches_question_rows",
        int(focal_summary["n_questions"]) == len(questions),
        f"Focal summary n_questions matches parquet row count ({len(questions)}).",
        f"Focal summary n_questions={focal_summary['n_questions']} but parquet has {len(questions)} rows.",
    )
    add_result(
        results,
        "panel_summary_matches_question_level_rows",
        int(panel_summary["n_question_level_rows"]) == len(question_level),
        f"Panel summary n_question_level_rows matches parquet row count ({len(question_level)}).",
        f"Panel summary n_question_level_rows={panel_summary['n_question_level_rows']} but parquet has {len(question_level)} rows.",
    )

    write_reports(results, metadata)
    print(json.dumps({"metadata": metadata, "n_pass": sum(r.status == 'pass' for r in results), "n_fail": sum(r.status == 'fail' for r in results)}, indent=2))


if __name__ == "__main__":
    main()
