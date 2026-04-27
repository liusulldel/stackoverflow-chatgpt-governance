from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper" / "staged_public_resolution"
FIGURES_DIR = BASE_DIR / "figures"

QUESTION_FILE = PROCESSED_DIR / "stackexchange_20251231_question_level_enriched.parquet"
COMPLEXITY_FILE = PROCESSED_DIR / "stackexchange_20251231_question_complexity_features.parquet"
CALL_LEDGER = PAPER_DIR / "exposure_call_ledger_2026-04-04.csv"

SCORED_FILE = PROCESSED_DIR / "question_level_exposure_fullsample_scores.parquet"
VALIDATION_JSON = PROCESSED_DIR / "question_level_exposure_model_validation.json"
VALIDATION_REG_CSV = PROCESSED_DIR / "question_level_exposure_validation_regressions.csv"
SUMMARY_MD = PAPER_DIR / "question_level_exposure_index.md"
FIGURE_FILE = FIGURES_DIR / "question_level_exposure_validation.png"

NUMERIC_COLS = [
    "title_length_chars",
    "body_length_chars",
    "body_word_count",
    "code_block_count",
    "inline_code_count",
    "code_char_count",
    "error_keyword_count",
    "error_keyword_density",
    "comment_count",
    "has_edit",
    "tag_count_full",
]
TEXT_COL = "combined_text"
TAG_COL = "primary_tag"
TARGET_COL = "rubric_exposure_score"
ALPHAS = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
SEED = 42


def load_base() -> pd.DataFrame:
    questions = pd.read_parquet(
        QUESTION_FILE,
        columns=[
            "question_id",
            "question_created_at",
            "title",
            "question_tags",
            "primary_tag",
            "month_id",
            "post_chatgpt",
            "keep_single_focal",
            "high_tag",
            "exposure_index",
            "first_answer_1d",
            "first_answer_7d",
            "accepted_30d",
        ],
    )
    complexity = pd.read_parquet(COMPLEXITY_FILE, columns=["question_id", *NUMERIC_COLS])
    df = questions.merge(complexity, on="question_id", how="left")
    df = df.loc[df["primary_tag"].notna()].copy()
    df["combined_text"] = (
        df["title"].fillna("").astype(str).str.strip()
        + " [TAGS] "
        + df["question_tags"].fillna("").astype(str).str.replace("|", " ", regex=False)
    )
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def load_labels() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(PROCESSED_DIR.glob("question_level_exposure*_labels.csv"))]
    if not frames:
        raise RuntimeError("No question-level exposure label files found.")
    labels = pd.concat(frames, ignore_index=True)
    question_level = (
        labels.groupby("question_id", as_index=False)
        .agg(
            rubric_exposure_score=("rubric_exposure_score", "mean"),
            overall_private_ai_substitutability=("overall_private_ai_substitutability", "mean"),
            mean_confidence=("confidence", "mean"),
            n_labels=("question_id", "size"),
            primary_tag=("primary_tag", "first"),
            post_chatgpt=("post_chatgpt", "first"),
        )
    )
    return question_level


def load_raw_labels() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(PROCESSED_DIR.glob("question_level_exposure*_labels.csv"))]
    if not frames:
        raise RuntimeError("No question-level exposure label files found.")
    return pd.concat(frames, ignore_index=True)


def load_summary_files() -> list[Path]:
    return sorted(PROCESSED_DIR.glob("question_level_exposure*_summary.json"))


def load_audited_cumulative_calls() -> int:
    if not CALL_LEDGER.exists():
        return 0
    max_total = 0
    with CALL_LEDGER.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\n").split(",", 5)
            if len(parts) < 4:
                continue
            try:
                max_total = max(max_total, int(parts[3]))
            except ValueError:
                continue
    return max_total


def build_design(df: pd.DataFrame, vectorizer: TfidfVectorizer | None = None, encoder: OneHotEncoder | None = None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=15,
            strip_accents="unicode",
        )
        x_text = vectorizer.fit_transform(df[TEXT_COL])
    else:
        x_text = vectorizer.transform(df[TEXT_COL])

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore")
        x_tag = encoder.fit_transform(df[[TAG_COL]])
    else:
        x_tag = encoder.transform(df[[TAG_COL]])

    x_num = sp.csr_matrix(df[NUMERIC_COLS].to_numpy(dtype=float))
    x = sp.hstack([x_text, x_tag, x_num], format="csr")
    return x, vectorizer, encoder


def select_alpha(x_train: sp.csr_matrix, y_train: np.ndarray) -> tuple[float, dict]:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores: dict[float, list[float]] = {alpha: [] for alpha in ALPHAS}
    for train_idx, test_idx in kf.split(np.arange(len(y_train))):
        x_tr = x_train[train_idx]
        x_te = x_train[test_idx]
        y_tr = y_train[train_idx]
        y_te = y_train[test_idx]
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha, random_state=SEED)
            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            scores[alpha].append(mean_absolute_error(y_te, pred))
    mean_scores = {alpha: float(np.mean(vals)) for alpha, vals in scores.items()}
    best_alpha = min(mean_scores, key=mean_scores.get)
    return best_alpha, mean_scores


def cross_validated_predictions(x_train: sp.csr_matrix, y_train: np.ndarray, alpha: float) -> np.ndarray:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = np.zeros(len(y_train))
    for train_idx, test_idx in kf.split(np.arange(len(y_train))):
        model = Ridge(alpha=alpha, random_state=SEED)
        model.fit(x_train[train_idx], y_train[train_idx])
        preds[test_idx] = model.predict(x_train[test_idx])
    return preds


def build_panel_validation(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, dict]:
    panel = (
        df.groupby(["primary_tag", "month_id", "exposure_quintile", "post_chatgpt"], observed=False)
        .agg(
            outcome_mean=(outcome, "mean"),
            n=("question_id", "size"),
            exposure_mean=("predicted_exposure_within_tag_z", "mean"),
        )
        .reset_index()
    )
    panel["top_quintile"] = (panel["exposure_quintile"] == 5).astype(int)
    panel["month_id"] = panel["month_id"].astype(str)
    model = smf.wls(
        "outcome_mean ~ post_chatgpt * top_quintile + C(primary_tag) + C(month_id)",
        data=panel,
        weights=panel["n"],
    ).fit(cov_type="HC1")
    row = model.params
    pvals = model.pvalues
    result = {
        "outcome": outcome,
        "coef_post_x_top_quintile": float(row.get("post_chatgpt:top_quintile", np.nan)),
        "p_post_x_top_quintile": float(pvals.get("post_chatgpt:top_quintile", np.nan)),
        "n_cells": int(len(panel)),
    }
    return panel, result


def build_composition_shift_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    panel = (
        df.groupby(["primary_tag", "month_id", "post_chatgpt", "high_tag"], observed=False)
        .agg(
            predicted_exposure_mean=("predicted_exposure_score", "mean"),
            n=("question_id", "size"),
        )
        .reset_index()
    )
    panel["month_id"] = panel["month_id"].astype(str)
    model = smf.wls(
        "predicted_exposure_mean ~ post_chatgpt * high_tag + C(primary_tag) + C(month_id)",
        data=panel,
        weights=panel["n"],
    ).fit(cov_type="HC1")
    result = {
        "outcome": "predicted_exposure_score",
        "coef_post_x_high_tag": float(model.params.get("post_chatgpt:high_tag", np.nan)),
        "p_post_x_high_tag": float(model.pvalues.get("post_chatgpt:high_tag", np.nan)),
        "n_cells": int(len(panel)),
    }
    return panel, result


def make_figure(df: pd.DataFrame) -> None:
    plot_df = df.loc[:, ["predicted_exposure_within_tag_z", "post_chatgpt", "first_answer_1d", "high_tag"]].copy()
    plot_df["period"] = np.where(plot_df["post_chatgpt"] == 1, "Post", "Pre")
    plot_df["manual_group"] = np.where(plot_df["high_tag"] == 1, "High-tag proxy", "Low-tag proxy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for idx, period in enumerate(["Pre", "Post"]):
        subset = plot_df.loc[plot_df["period"] == period]
        high = subset.loc[subset["manual_group"] == "High-tag proxy", "predicted_exposure_within_tag_z"].to_numpy()
        low = subset.loc[subset["manual_group"] == "Low-tag proxy", "predicted_exposure_within_tag_z"].to_numpy()
        axes[idx].hist(low, bins=40, alpha=0.6, label="Low-tag proxy", density=True)
        axes[idx].hist(high, bins=40, alpha=0.6, label="High-tag proxy", density=True)
        axes[idx].set_title(f"{period} predicted within-tag exposure")
        axes[idx].set_xlabel("Predicted exposure z-score")
        axes[idx].set_ylabel("Density")
        axes[idx].legend(frameon=False)
    fig.tight_layout()
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_FILE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(summary: dict) -> None:
    n_waves = int(summary.get("n_completed_waves", 0))
    wave_noun = "wave" if n_waves == 1 else "waves"
    lines = [
        "# Question-Level Exposure Index",
        "",
        "## API Labeling Summary",
        "",
        f"- Successful API labeling calls across all completed waves: `{summary['api_calls']}`",
        f"- Target met (`>=500`): `{summary['api_calls'] >= 500}`",
        f"- Strict audited cumulative completed-call total for this branch: `{summary.get('audited_cumulative_calls', summary['api_calls'])}`",
        f"- Unique labeled questions: `{summary['n_labeled_questions']}`",
        f"- Mean within-tag SD in labeled sample: `{summary['labeled_within_tag_sd_mean']:.4f}`",
        "",
        "## Calibration Agreement",
        "",
        f"- Paired calibration items: `{summary['agreement']['n_paired']}`",
        f"- Exact overall match rate: `{summary['agreement']['exact_overall_match_rate']:.4f}`",
        f"- Mean absolute overall difference: `{summary['agreement']['mean_abs_overall_diff']:.4f}`",
        f"- Mean absolute rubric difference: `{summary['agreement']['mean_abs_rubric_diff']:.4f}`",
        "",
        "## Full-Sample Scoring",
        "",
        f"- Canonical focal questions scored: `{summary['n_scored_questions']}`",
        f"- Chosen Ridge alpha: `{summary['best_alpha']}`",
        f"- Cross-validated MAE on labeled sample: `{summary['cv_mae']:.4f}`",
        f"- Cross-validated R^2 on labeled sample: `{summary['cv_r2']:.4f}`",
        f"- Cross-validated correlation on labeled sample: `{summary['cv_corr']:.4f}`",
        f"- Correlation with legacy continuous tag proxy: `{summary['corr_with_legacy_exposure_index']:.4f}`",
        f"- Mean within-tag SD of predicted exposure: `{summary['predicted_within_tag_sd_mean']:.4f}`",
        "",
        "## Outcome-Facing Validation",
        "",
        f"- `first_answer_1d`: top-quintile x post = `{summary['validation']['first_answer_1d']['coef_post_x_top_quintile']:.4f}`, `p={summary['validation']['first_answer_1d']['p_post_x_top_quintile']:.4f}`",
        f"- `first_answer_7d`: top-quintile x post = `{summary['validation']['first_answer_7d']['coef_post_x_top_quintile']:.4f}`, `p={summary['validation']['first_answer_7d']['p_post_x_top_quintile']:.4f}`",
        f"- `accepted_30d`: top-quintile x post = `{summary['validation']['accepted_30d']['coef_post_x_top_quintile']:.4f}`, `p={summary['validation']['accepted_30d']['p_post_x_top_quintile']:.4f}`",
        f"- `predicted_exposure_score`: high-tag x post = `{summary['validation']['predicted_exposure_score']['coef_post_x_high_tag']:.4f}`, `p={summary['validation']['predicted_exposure_score']['p_post_x_high_tag']:.4f}`",
        "",
        "## Read",
        "",
        f"This round upgrades the project from a pure tag-level exposure split to a question-level continuous exposure score that varies within tag. The index is still a modeled proxy rather than direct AI-use observation, so it should be positioned as a measurement-strengthening layer and a bounded validation device rather than a replacement natural experiment. With {n_waves} completed labeling {wave_noun}, the strongest read is still two-part: higher predicted exposure questions receive faster public response, but the average exposure of questions remaining in the legacy high-tag queue declines after ChatGPT.",
        "",
        f"Validation figure: `{FIGURE_FILE.name}`",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base = load_base()
    score_base = base.copy()
    labels = load_labels()
    raw_labels = load_raw_labels()

    train_df = base.merge(labels, on=["question_id", "primary_tag", "post_chatgpt"], how="inner")
    train_df = train_df.drop_duplicates(subset=["question_id"]).copy()

    if train_df.empty:
        raise RuntimeError("Training frame is empty after merging labels onto the primary-tag universe.")

    x_all, vectorizer, encoder = build_design(score_base)
    x_train, _, _ = build_design(train_df, vectorizer=vectorizer, encoder=encoder)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)

    best_alpha, alpha_scores = select_alpha(x_train, y_train)
    cv_pred = cross_validated_predictions(x_train, y_train, best_alpha)
    cv_mae = mean_absolute_error(y_train, cv_pred)
    cv_r2 = r2_score(y_train, cv_pred)
    cv_corr = float(np.corrcoef(y_train, cv_pred)[0, 1])

    final_model = Ridge(alpha=best_alpha, random_state=SEED)
    final_model.fit(x_train, y_train)
    score_base["predicted_exposure_score"] = final_model.predict(x_all)
    score_base["predicted_exposure_within_tag_z"] = (
        score_base.groupby("primary_tag", observed=False)["predicted_exposure_score"]
        .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if float(s.std(ddof=0)) > 0 else 1.0))
    )
    score_base["exposure_quintile"] = (
        score_base.groupby("primary_tag", observed=False)["predicted_exposure_within_tag_z"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop") + 1)
        .astype(int)
    )

    _, reg_first = build_panel_validation(score_base.dropna(subset=["first_answer_1d"]), "first_answer_1d")
    _, reg_seven = build_panel_validation(score_base.dropna(subset=["first_answer_7d"]), "first_answer_7d")
    _, reg_accept = build_panel_validation(score_base.dropna(subset=["accepted_30d"]), "accepted_30d")
    _, reg_composition = build_composition_shift_validation(score_base.dropna(subset=["high_tag"]))

    validation_rows = pd.DataFrame([reg_first, reg_seven, reg_accept, reg_composition])
    validation_rows.to_csv(VALIDATION_REG_CSV, index=False)

    scored_cols = [
        "question_id",
        "question_created_at",
        "primary_tag",
        "month_id",
        "post_chatgpt",
        "high_tag",
        "exposure_index",
        "predicted_exposure_score",
        "predicted_exposure_within_tag_z",
        "exposure_quintile",
        "first_answer_1d",
        "first_answer_7d",
        "accepted_30d",
    ]
    score_base.loc[:, scored_cols].to_parquet(SCORED_FILE, index=False)

    make_figure(score_base)

    total_api_calls = 0
    within_tag_values = []
    summary_files = load_summary_files()
    for path in summary_files:
        if not path.exists():
            continue
        current = json.loads(path.read_text(encoding="utf-8"))
        total_api_calls += int(current.get("successful_api_calls", 0))
        within_tag_values.append(float(current.get("within_tag_sd_mean", np.nan)))
    main_raw = raw_labels.loc[raw_labels["label_pass"] == 1].copy()
    cal_raw = raw_labels.loc[raw_labels["label_pass"] == 2].copy()
    if not cal_raw.empty:
        paired = main_raw.merge(
            cal_raw[["question_id", "overall_private_ai_substitutability", "rubric_exposure_score"]],
            on="question_id",
            suffixes=("_p1", "_p2"),
            how="inner",
        )
        agreement_source = {
            "n_paired": int(len(paired)),
            "exact_overall_match_rate": float(np.mean(paired["overall_private_ai_substitutability_p1"] == paired["overall_private_ai_substitutability_p2"])),
            "mean_abs_overall_diff": float(np.mean(np.abs(paired["overall_private_ai_substitutability_p1"] - paired["overall_private_ai_substitutability_p2"]))),
            "mean_abs_rubric_diff": float(np.mean(np.abs(paired["rubric_exposure_score_p1"] - paired["rubric_exposure_score_p2"]))),
        }
    else:
        agreement_source = {}
    summary = {
        "api_calls": int(total_api_calls),
        "audited_cumulative_calls": load_audited_cumulative_calls(),
        "n_completed_waves": int(len(summary_files)),
        "n_labeled_questions": int(labels["question_id"].nunique()),
        "labeled_within_tag_sd_mean": float(np.nanmean(within_tag_values)),
        "agreement": agreement_source,
        "best_alpha": float(best_alpha),
        "alpha_cv_mae": alpha_scores,
        "cv_mae": float(cv_mae),
        "cv_r2": float(cv_r2),
        "cv_corr": float(cv_corr),
        "n_scored_questions": int(len(score_base)),
        "corr_with_legacy_exposure_index": float(score_base[["predicted_exposure_score", "exposure_index"]].corr().iloc[0, 1]),
        "predicted_within_tag_sd_mean": float(score_base.groupby("primary_tag", observed=False)["predicted_exposure_score"].std().mean()),
        "validation": {
            "first_answer_1d": reg_first,
            "first_answer_7d": reg_seven,
            "accepted_30d": reg_accept,
            "predicted_exposure_score": reg_composition,
        },
    }
    VALIDATION_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
