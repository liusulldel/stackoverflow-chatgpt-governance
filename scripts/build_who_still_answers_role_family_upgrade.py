from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
PAPER_DIR = BASE_DIR / "paper"

ROLE_QUESTION_PANEL = PROCESSED_DIR / "who_still_answers_answer_role_question_panel.parquet"
QUESTION_PANEL = PROCESSED_DIR / "who_still_answers_question_closure_panel.parquet"

OUT_COMMON_SUPPORT = PROCESSED_DIR / "who_still_answers_role_common_support_results.csv"
OUT_STAGE_WEIGHTED = PROCESSED_DIR / "who_still_answers_role_stage_weighted_results.csv"
OUT_GRADIENT = PROCESSED_DIR / "who_still_answers_role_pipeline_gradient_tests.csv"
OUT_SUPPORT = PROCESSED_DIR / "who_still_answers_role_support_counts.csv"
OUT_SUMMARY = PROCESSED_DIR / "who_still_answers_role_family_upgrade_summary.json"

READOUT_MD = PAPER_DIR / "who_still_answers_role_family_upgrade_readout_2026-04-13.md"

ROLE_ORDER = ["first_answer", "first_positive", "top_score", "accepted_current"]
PRIMARY_OUTCOME_COL = "recent_entrant_90d"


@dataclass(frozen=True)
class PanelSpec:
    name: str
    question_filter: str
    weight_col: str | None


def _safe_title_word_count(s: pd.Series) -> pd.Series:
    # Keep it deterministic and cheap; do not depend on language models.
    return s.fillna("").astype(str).str.split().str.len().clip(lower=0, upper=200)


def _safe_title_char_len(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.len().clip(lower=0, upper=400)


def load_role_questions() -> pd.DataFrame:
    df = pd.read_parquet(ROLE_QUESTION_PANEL)
    df["post_chatgpt"] = df["post_chatgpt"].astype(int)
    df["time_index"] = df["time_index"].astype(int)
    df["high_post"] = df["high_post"].astype(int)
    # Convenience: the role-specific outcome is already per-question (one row per role per question).
    return df


def load_question_features() -> pd.DataFrame:
    cols = [
        "question_id",
        "primary_tag",
        "month_id",
        "post_chatgpt",
        "exposure_index",
        "high_tag",
        "title",
        "selected_tags_list",
        "answer_count",
        "accepted_archive",
    ]
    q = pd.read_parquet(QUESTION_PANEL, columns=cols).drop_duplicates("question_id")
    q["post_chatgpt"] = q["post_chatgpt"].astype(int)
    q["answer_count"] = pd.to_numeric(q["answer_count"], errors="coerce").fillna(0).astype(int)
    q["accepted_archive"] = pd.to_numeric(q["accepted_archive"], errors="coerce").fillna(0).astype(int)
    q["n_tags"] = q["selected_tags_list"].apply(lambda x: len(x) if isinstance(x, list) else 0).astype(int)
    q["title_word_count"] = _safe_title_word_count(q["title"])
    q["title_char_len"] = _safe_title_char_len(q["title"])
    return q


def build_role_eligibility_flags(role_q: pd.DataFrame) -> pd.DataFrame:
    flags = (
        role_q[["question_id", "role"]]
        .drop_duplicates()
        .assign(v=1)
        .pivot_table(index="question_id", columns="role", values="v", fill_value=0, aggfunc="max")
        .reset_index()
    )
    for r in ROLE_ORDER:
        if r not in flags.columns:
            flags[r] = 0
    flags = flags[["question_id", *ROLE_ORDER]]
    flags = flags.rename(columns={r: f"has_{r}" for r in ROLE_ORDER})
    flags["common_support_all4"] = (
        (flags["has_first_answer"] == 1)
        & (flags["has_first_positive"] == 1)
        & (flags["has_top_score"] == 1)
        & (flags["has_accepted_current"] == 1)
    ).astype(int)
    flags["common_support_endorsed_cert"] = (
        (flags["has_first_positive"] == 1)
        & (flags["has_top_score"] == 1)
        & (flags["has_accepted_current"] == 1)
    ).astype(int)
    return flags


def fit_stage_eligibility_models(q: pd.DataFrame) -> tuple[object, object, object]:
    """
    Stage eligibility models are pre-period-only and use only posting-time-ish features.

    We intentionally keep these models simple: they are for inverse-probability weighting to
    reduce stage-eligibility comparability concerns, not for causal claims.
    """
    pre = q.loc[q["post_chatgpt"] == 0].copy()
    # Outcome flags live in q as has_* columns
    # We include tag FE because the tag universe is small (16) and stage eligibility is tag-shaped.
    answered_m = smf.logit(
        "has_first_answer ~ exposure_index + high_tag + n_tags + title_word_count + C(primary_tag)",
        data=pre,
    ).fit(disp=0)
    positive_m = smf.logit(
        "has_first_positive ~ exposure_index + high_tag + n_tags + title_word_count + C(primary_tag)",
        data=pre,
    ).fit(disp=0)
    accepted_m = smf.logit(
        "has_accepted_current ~ exposure_index + high_tag + n_tags + title_word_count + C(primary_tag)",
        data=pre,
    ).fit(disp=0)
    return answered_m, positive_m, accepted_m


def add_stage_weights(role_q: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    answered_m, positive_m, accepted_m = fit_stage_eligibility_models(q)
    q = q.copy()
    q["p_answered"] = answered_m.predict(q).clip(1e-3, 1 - 1e-3)
    q["p_positive"] = positive_m.predict(q).clip(1e-3, 1 - 1e-3)
    q["p_accepted"] = accepted_m.predict(q).clip(1e-3, 1 - 1e-3)

    merged = role_q.merge(q[["question_id", "p_answered", "p_positive", "p_accepted"]], on="question_id", how="left")

    merged["w_stage_later_only"] = 1.0
    merged.loc[merged["role"] == "first_positive", "w_stage_later_only"] = 1.0 / merged["p_positive"]
    merged.loc[merged["role"] == "accepted_current", "w_stage_later_only"] = 1.0 / merged["p_accepted"]

    merged["w_stage_all"] = 1.0
    merged.loc[merged["role"].isin(["first_answer", "top_score"]), "w_stage_all"] = 1.0 / merged["p_answered"]
    merged.loc[merged["role"] == "first_positive", "w_stage_all"] = 1.0 / merged["p_positive"]
    merged.loc[merged["role"] == "accepted_current", "w_stage_all"] = 1.0 / merged["p_accepted"]

    # Conservative trim to avoid a few tiny p's dominating.
    for c in ["w_stage_later_only", "w_stage_all"]:
        merged[c] = merged[c].clip(lower=0.0, upper=float(np.nanpercentile(merged[c].dropna(), 99)))
    return merged


def aggregate_tag_month_role(role_q: pd.DataFrame, weight_col: str | None) -> pd.DataFrame:
    df = role_q.copy()
    if weight_col is None:
        df["_w"] = 1.0
    else:
        df["_w"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
    # Weighted mean recent-entrant indicator within each tag-month-role cell.
    out = (
        df.groupby(["primary_tag", "month_id", "role"], as_index=False)
        .agg(
            exposure_index=("exposure_index", "first"),
            post_chatgpt=("post_chatgpt", "first"),
            time_index=("time_index", "first"),
            residual_queue_complexity_index_mean=("residual_queue_complexity_index_mean", "first"),
            n_role_questions=("question_id", "nunique"),
            sum_w=("_w", "sum"),
            y_w=(PRIMARY_OUTCOME_COL, lambda s: float(np.nan)),  # placeholder
        )
    )
    # compute weighted mean y
    tmp = (
        df.groupby(["primary_tag", "month_id", "role"], as_index=False)
        .apply(lambda g: pd.Series({"y_w": float((g["_w"] * g[PRIMARY_OUTCOME_COL]).sum() / g["_w"].sum())}))
        .reset_index(drop=True)
    )
    out = out.drop(columns=["y_w"]).merge(tmp, on=["primary_tag", "month_id", "role"], how="left")
    out = out.rename(columns={"y_w": "recent_entrant_90d_share"})
    return out


def fit_role_wls(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for role in ROLE_ORDER:
        sub = panel.loc[panel["role"] == role].copy()
        formula = (
            "recent_entrant_90d_share ~ exposure_index * post_chatgpt + residual_queue_complexity_index_mean + "
            "C(primary_tag):time_index + C(primary_tag) + C(month_id)"
        )
        model = smf.wls(formula, data=sub, weights=sub["n_role_questions"]).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["primary_tag"], "use_correction": True, "df_correction": True},
        )
        rows.append(
            {
                "role": role,
                "coef": float(model.params.get("exposure_index:post_chatgpt", np.nan)),
                "se": float(model.bse.get("exposure_index:post_chatgpt", np.nan)),
                "pval": float(model.pvalues.get("exposure_index:post_chatgpt", np.nan)),
                "n_cells": int(len(sub)),
                "mean_outcome": float(sub["recent_entrant_90d_share"].mean()),
                "mean_weight": float(sub["n_role_questions"].mean()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_tag_diff(panel: pd.DataFrame, role_a: str, role_b: str, n_boot: int = 200, seed: int = 7) -> dict[str, float]:
    """
    Few-cluster friendly difference test via tag bootstrap (resample tags with replacement).
    This is not a full wild-cluster implementation; it’s a pragmatic differential check for the ordered-gradient story.
    """
    rng = np.random.default_rng(seed)
    tags = sorted(panel["primary_tag"].unique().tolist())
    if len(tags) < 8:
        return {"diff": float("nan"), "p_boot_two_sided": float("nan"), "n_tags": float(len(tags))}
    base = fit_role_wls(panel).set_index("role")
    diff0 = float(base.loc[role_a, "coef"] - base.loc[role_b, "coef"])
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(tags, size=len(tags), replace=True)
        boot = panel.loc[panel["primary_tag"].isin(sampled)].copy()
        # Duplicate tags can matter; approximate by reweighting rows by tag multiplicity.
        mult = pd.Series(sampled).value_counts()
        boot["_boot_mult"] = boot["primary_tag"].map(mult).astype(float)
        boot["n_role_questions"] = boot["n_role_questions"] * boot["_boot_mult"]
        b = fit_role_wls(boot).set_index("role")
        diffs.append(float(b.loc[role_a, "coef"] - b.loc[role_b, "coef"]))
    diffs = np.asarray(diffs)
    p = float(np.mean(np.abs(diffs) >= abs(diff0)))
    return {"diff": diff0, "p_boot_two_sided": p, "n_tags": float(len(tags))}


def main() -> None:
    role_q = load_role_questions()
    q = load_question_features()
    flags = build_role_eligibility_flags(role_q)
    q = q.merge(flags, on="question_id", how="left")

    # Support counts audit
    support = q[["question_id", "primary_tag", "month_id", "post_chatgpt", "common_support_all4", "common_support_endorsed_cert"]].copy()
    support_counts = (
        support.groupby(["post_chatgpt"], as_index=False)
        .agg(
            n_questions=("question_id", "count"),
            n_common_all4=("common_support_all4", "sum"),
            n_common_endorsed_cert=("common_support_endorsed_cert", "sum"),
        )
    )
    support_counts.to_csv(OUT_SUPPORT, index=False)

    # Build common-support panels
    role_q = role_q.merge(flags[["question_id", "common_support_all4", "common_support_endorsed_cert"]], on="question_id", how="left")
    panel_specs = [
        PanelSpec("common_all4", "common_support_all4 == 1", None),
        PanelSpec("common_endorsed_cert", "common_support_endorsed_cert == 1", None),
    ]
    common_rows = []
    for spec in panel_specs:
        sub = role_q.query(spec.question_filter).copy()
        panel = aggregate_tag_month_role(sub, spec.weight_col)
        res = fit_role_wls(panel)
        res["panel"] = spec.name
        common_rows.append(res)
    common = pd.concat(common_rows, ignore_index=True)
    common.to_csv(OUT_COMMON_SUPPORT, index=False)

    # Build stage-weighted panels (native eligibility samples, but eligibility-weighted)
    role_q_w = add_stage_weights(role_q, q)
    weighted_specs = [
        PanelSpec("native_unweighted", "role == role", None),
        PanelSpec("native_w_later_only", "role == role", "w_stage_later_only"),
        PanelSpec("native_w_all", "role == role", "w_stage_all"),
    ]
    weighted_rows = []
    for spec in weighted_specs:
        sub = role_q_w.copy()
        panel = aggregate_tag_month_role(sub, spec.weight_col)
        res = fit_role_wls(panel)
        res["panel"] = spec.name
        weighted_rows.append(res)
    weighted = pd.concat(weighted_rows, ignore_index=True)
    weighted.to_csv(OUT_STAGE_WEIGHTED, index=False)

    # Ordered pipeline gradient tests (vs accepted_current)
    gradient_rows = []
    for name, df in [
        ("common_all4", role_q.query("common_support_all4 == 1")),
        ("common_endorsed_cert", role_q.query("common_support_endorsed_cert == 1")),
        ("native_unweighted", role_q),
        ("native_w_later_only", role_q_w),
    ]:
        panel = aggregate_tag_month_role(df, "w_stage_later_only" if name == "native_w_later_only" else None)
        for r in ["first_answer", "first_positive", "top_score"]:
            out = bootstrap_tag_diff(panel, r, "accepted_current", n_boot=200, seed=11)
            out.update({"panel": name, "role_a": r, "role_b": "accepted_current"})
            gradient_rows.append(out)
    grad = pd.DataFrame(gradient_rows)
    grad.to_csv(OUT_GRADIENT, index=False)

    summary = {
        "support_counts_by_post": support_counts.to_dict(orient="records"),
        "common_support_panels": common.to_dict(orient="records"),
        "stage_weighted_panels": weighted.to_dict(orient="records"),
        "gradient_tests": grad.to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Role-Family Upgrade Readout",
        "",
        "This file summarizes the `common-support` and `stage-eligibility weighting` upgrades to the answer-role mechanism.",
        "It is intended to reduce the reviewer objection that role comparisons reflect different eligible sets rather than pipeline reallocation.",
        "",
        "## Outputs",
        "",
        f"- `{OUT_COMMON_SUPPORT.name}`",
        f"- `{OUT_STAGE_WEIGHTED.name}`",
        f"- `{OUT_GRADIENT.name}`",
        f"- `{OUT_SUPPORT.name}`",
        "",
        "## Notes",
        "",
        "- Common-support panels restrict to questions eligible for multiple roles (all-4 and endorsed+certification).",
        "- Stage-eligibility weighting uses pre-period-only logit models with posting-time-ish question features (title length, tag count, exposure, tag FE).",
        "- Gradient tests use a tag bootstrap difference check; treat as supportive, not definitive conservative inference.",
        "",
    ]
    READOUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

