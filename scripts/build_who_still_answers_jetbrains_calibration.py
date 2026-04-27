from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
RAW = ROOT / "raw" / "external_validation" / "jetbrains_deveco_2025"
PROCESSED = ROOT / "processed"
PAPER = ROOT / "paper"

QUESTION_PATH = next(RAW.glob("*\\developer_ecosystem_2025_external_questions.csv"))
DATA_PATH = next(RAW.glob("*\\developer_ecosystem_2025_external.csv"))
EXPOSURE_PATH = PROCESSED / "who_still_answers_tag_exposure_panel.csv"

TABLE_CSV = PROCESSED / "who_still_answers_jetbrains_calibration_table.csv"
TABLE_MD = PAPER / "who_still_answers_jetbrains_calibration_table_2026-04-04.md"
READOUT_MD = PAPER / "who_still_answers_jetbrains_calibration_readout_2026-04-04.md"
SUMMARY_JSON = PROCESSED / "who_still_answers_jetbrains_calibration_summary.json"


def load_data() -> pd.DataFrame:
    usecols = [
        "main_lang",
        "answers_platform::ChatGPT or other AI chatbots",
        "answers_platform::Stack Overflow",
        "ai_benefits::Less time spent searching for information",
        "ai_coding_tasks_freq::Debugging code",
        "development_type::Infrastructure / DevOps",
        "development_type::Mobile",
        "development_type::Data science and ML",
        "platform::Server / Infrastructure / Cloud",
        "platform::Mobile",
        "data_role::Data Analyst",
        "data_role::Business Analyst",
    ]
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c in usecols, low_memory=False)
    for col in df.columns:
        if col != "main_lang":
            df[col] = df[col].notna() & (df[col].astype(str).str.strip() != "")
    return df


def build_cluster_table(df: pd.DataFrame) -> pd.DataFrame:
    tag_exposure = pd.read_csv(EXPOSURE_PATH, usecols=["primary_tag", "exposure_index"])
    clusters = {
        "SQL / analytics": {
            "tags": ["sql", "excel", "regex"],
            "mask": (
                df["main_lang"].eq("SQL (PL/SQL, T-SQL, or other programming extensions of SQL)")
                | df["data_role::Data Analyst"]
                | df["data_role::Business Analyst"]
            ),
        },
        "Python / data": {
            "tags": ["python", "pandas", "numpy", "apache-spark"],
            "mask": df["main_lang"].eq("Python") | df["development_type::Data science and ML"],
        },
        "JavaScript / web": {
            "tags": ["javascript", "firebase"],
            "mask": df["main_lang"].isin(["JavaScript", "TypeScript", "HTML / CSS"]),
        },
        "Android / mobile": {
            "tags": ["android", "multithreading", "memory-management"],
            "mask": df["development_type::Mobile"] | df["platform::Mobile"],
        },
        "Shell / infra / cloud": {
            "tags": ["bash", "linux", "docker", "kubernetes"],
            "mask": (
                df["main_lang"].eq("Shell scripting languages (Bash, Shell, PowerShell, etc.)")
                | df["development_type::Infrastructure / DevOps"]
                | df["platform::Server / Infrastructure / Cloud"]
            ),
        },
    }
    rows = []
    for cluster_name, spec in clusters.items():
        sub = df[spec["mask"].fillna(False)].copy()
        exposure_sub = tag_exposure[tag_exposure["primary_tag"].isin(spec["tags"])]
        rows.append(
            {
                "cluster": cluster_name,
                "focal_tags": ", ".join(spec["tags"]),
                "mean_exposure_index": float(exposure_sub["exposure_index"].mean()),
                "min_exposure_index": float(exposure_sub["exposure_index"].min()),
                "max_exposure_index": float(exposure_sub["exposure_index"].max()),
                "respondent_n": int(len(sub)),
                "chatgpt_answers_share": float(sub["answers_platform::ChatGPT or other AI chatbots"].mean()),
                "stackoverflow_answers_share": float(sub["answers_platform::Stack Overflow"].mean()),
                "private_public_gap": float(
                    sub["answers_platform::ChatGPT or other AI chatbots"].mean()
                    - sub["answers_platform::Stack Overflow"].mean()
                ),
                "ai_search_saving_share": float(
                    sub["ai_benefits::Less time spent searching for information"].mean()
                ),
                "ai_debugging_share": float(sub["ai_coding_tasks_freq::Debugging code"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_exposure_index", ascending=False).reset_index(drop=True)


def write_markdown_table(table: pd.DataFrame) -> None:
    display = table.copy()
    for col in [
        "mean_exposure_index",
        "chatgpt_answers_share",
        "stackoverflow_answers_share",
        "private_public_gap",
        "ai_search_saving_share",
        "ai_debugging_share",
    ]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    TABLE_MD.write_text(
        "# JetBrains Calibration Table\n\n" + display.to_markdown(index=False),
        encoding="utf-8",
    )


def write_readout(table: pd.DataFrame) -> None:
    best_gap = table.sort_values("private_public_gap", ascending=False).iloc[0]
    lines = [
        "# Who Still Answers: JetBrains Calibration Readout",
        "",
        "Date: April 4, 2026",
        "",
        "## What This Layer Does",
        "",
        "This layer uses the official JetBrains Developer Ecosystem 2025 raw survey as an external calibration source.",
        "It does not observe Stack Overflow users directly, but it does provide developer-level evidence on answer-source substitution and AI-assisted coding tasks in matched technical subgroups.",
        "",
        "## Main Calibration Table",
        "",
        f"- rows in the main calibration table: `{len(table)}`",
        f"- largest private-vs-public answer-source gap: `{best_gap['cluster']}` at `{best_gap['private_public_gap']:.3f}`",
        "",
        "## Safe Interpretation",
        "",
        "Across all matched technical clusters, the share selecting `ChatGPT or other AI chatbots` as an answers platform exceeds the share selecting `Stack Overflow`.",
        "That does not mechanically validate the exact tag-level exposure ordering, but it does strengthen the core public-vs-private answer-source substitution story with official developer-survey microdata.",
        "This is therefore a main-text calibration layer, not a substitute for the main behavioral design.",
    ]
    READOUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_data()
    table = build_cluster_table(df)
    table.to_csv(TABLE_CSV, index=False)
    write_markdown_table(table)
    write_readout(table)

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "source_csv": str(DATA_PATH),
                "source_questions_csv": str(QUESTION_PATH),
                "n_rows": int(len(df)),
                "clusters": table["cluster"].tolist(),
                "mean_private_public_gap": float(table["private_public_gap"].mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
