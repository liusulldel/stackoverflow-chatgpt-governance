from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper" / "staged_public_resolution"
FIGURES_DIR = BASE_DIR / "figures"

QUESTION_LEVEL_PARQUET = PROCESSED_DIR / "stackexchange_20251231_question_level_enriched.parquet"
COMPLEXITY_PARQUET = PROCESSED_DIR / "stackexchange_20251231_question_complexity_features.parquet"
OUTPUT_PANEL_CSV = PROCESSED_DIR / "selection_composition_primary_panel.csv"
OUTPUT_RESULTS_CSV = PROCESSED_DIR / "selection_composition_model_results.csv"
OUTPUT_RESULTS_JSON = PROCESSED_DIR / "selection_composition_results.json"
OUTPUT_MD = PAPER_DIR / "selection_composition_evidence.md"
FIGURE_PNG = FIGURES_DIR / "selection_composition_complexity_distribution.png"


def load_inputs() -> pd.DataFrame:
    df = pd.read_parquet(QUESTION_LEVEL_PARQUET)
    if COMPLEXITY_PARQUET.exists():
        complexity = pd.read_parquet(COMPLEXITY_PARQUET)
        df = df.merge(complexity, on="question_id", how="left")
    else:
        raise FileNotFoundError(f"Complexity features not found: {COMPLEXITY_PARQUET}")
    return df


def add_composite(df: pd.DataFrame) -> pd.DataFrame:
    work = df.loc[df["keep_single_focal"] == 1].copy()
    proxy_cols = [
        "title_length_chars",
        "body_length_chars",
        "body_word_count",
        "code_block_count",
        "code_char_count",
        "error_keyword_density",
        "comment_count",
        "has_edit",
        "tag_count_full",
    ]
    for col in proxy_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        col_mean = work[col].mean()
        col_std = work[col].std(ddof=0)
        work[f"z_{col}"] = 0.0 if pd.isna(col_std) or col_std == 0 else (work[col] - col_mean) / col_std
    broad_z_cols = [f"z_{col}" for col in proxy_cols]
    focused_proxy_cols = ["body_word_count", "tag_count_full", "has_edit"]
    focused_z_cols = [f"z_{col}" for col in focused_proxy_cols]
    work["complexity_index_broad"] = work[broad_z_cols].mean(axis=1)
    work["residual_queue_complexity_index"] = work[focused_z_cols].mean(axis=1)
    work["has_code_block"] = (work["code_block_count"].fillna(0) > 0).astype(int)
    work["code_share"] = (work["code_char_count"] / work["body_length_chars"].replace(0, np.nan)).fillna(0.0)
    return work


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (tag, month_id), g in df.groupby(["primary_tag", "month_id"], sort=True):
        rows.append(
            {
                "tag": tag,
                "month_id": month_id,
                "high_tag": int(g["high_tag"].iloc[0]),
                "post_chatgpt": int(g["post_chatgpt"].iloc[0]),
                "n_questions": int(len(g)),
                "title_length_chars_mean": float(g["title_length_chars"].mean()),
                "body_length_chars_mean": float(g["body_length_chars"].mean()),
                "body_word_count_mean": float(g["body_word_count"].mean()),
                "code_block_count_mean": float(g["code_block_count"].mean()),
                "code_char_count_mean": float(g["code_char_count"].mean()),
                "error_keyword_density_mean": float(g["error_keyword_density"].mean()),
                "comment_count_mean": float(g["comment_count"].mean()),
                "has_edit_mean": float(g["has_edit"].mean()),
                "tag_count_full_mean": float(g["tag_count_full"].mean()),
                "has_code_block_mean": float(g["has_code_block"].mean()),
                "code_share_mean": float(g["code_share"].mean()),
                "complexity_index_broad_mean": float(g["complexity_index_broad"].mean()),
                "residual_queue_complexity_index_mean": float(g["residual_queue_complexity_index"].mean()),
            }
        )
    panel = pd.DataFrame(rows)
    panel["high_post"] = panel["high_tag"] * panel["post_chatgpt"]
    panel["time_index"] = pd.factorize(panel["month_id"], sort=True)[0] + 1
    return panel


def fit_results(panel: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "title_length_chars_mean",
        "body_length_chars_mean",
        "body_word_count_mean",
        "code_block_count_mean",
        "code_char_count_mean",
        "error_keyword_density_mean",
        "comment_count_mean",
        "has_edit_mean",
        "tag_count_full_mean",
        "has_code_block_mean",
        "code_share_mean",
        "complexity_index_broad_mean",
        "residual_queue_complexity_index_mean",
    ]
    rows: list[dict] = []
    for outcome in outcomes:
        model = smf.wls(
            f"{outcome} ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
            data=panel,
            weights=panel["n_questions"],
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["tag"], "use_correction": True, "df_correction": True},
        )
        rows.append(
            {
                "outcome": outcome,
                "coef": float(model.params.get("high_post", np.nan)),
                "se": float(model.bse.get("high_post", np.nan)),
                "pval": float(model.pvalues.get("high_post", np.nan)),
                "nobs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def build_figure(df: pd.DataFrame) -> None:
    monthly = (
        df.groupby(["month_id", "high_tag"], as_index=False)
        .agg(residual_queue_complexity_index_mean=("residual_queue_complexity_index", "mean"))
    )
    monthly["month_start"] = pd.to_datetime(monthly["month_id"] + "-01", utc=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for high_tag, label, color in [(1, "Higher substitutability", "#B3472E"), (0, "Lower substitutability", "#1F5A7A")]:
        subset = monthly.loc[monthly["high_tag"] == high_tag].sort_values("month_start")
        ax.plot(subset["month_start"], subset["residual_queue_complexity_index_mean"], label=label, color=color, linewidth=2)
    ax.axvline(pd.Timestamp("2022-11-30T00:00:00Z"), color="#444444", linestyle="--", linewidth=1)
    ax.set_title("Queue Complexity Index Over Time")
    ax.set_ylabel("Mean residual-queue complexity index")
    ax.set_xlabel("Month")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(results: pd.DataFrame) -> None:
    selected = results.loc[
        results["outcome"].isin(
            [
                "body_word_count_mean",
                "tag_count_full_mean",
                "has_edit_mean",
                "residual_queue_complexity_index_mean",
            ]
        )
    ].copy()
    lines = [
        "# Selection and Composition Evidence",
        "",
        "## Read",
        "",
        "This memo evaluates whether the post-period public queue in higher-substitutability domains becomes more context-heavy and operationally messy.",
        "",
        "### Main promotion table",
        "",
        selected.to_markdown(index=False),
        "",
        "### Full proxy battery",
        "",
        results.to_markdown(index=False),
        "",
        "## Promotion Rule",
        "",
        "Promote this evidence into main text if the residual-queue complexity index is directionally aligned and at least two underlying proxies move in the predicted direction. Treat the broader code-centric battery as secondary because it is noisier and less central to the queue-composition claim.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    df = add_composite(load_inputs())
    panel = build_panel(df)
    results = fit_results(panel)
    panel.to_csv(OUTPUT_PANEL_CSV, index=False)
    results.to_csv(OUTPUT_RESULTS_CSV, index=False)
    OUTPUT_RESULTS_JSON.write_text(
        json.dumps(
            {
                "results": results.to_dict(orient="records"),
                "n_question_rows": int(len(df)),
                "n_panel_rows": int(len(panel)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    build_figure(df)
    write_summary(results)


if __name__ == "__main__":
    main()
