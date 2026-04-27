import json
import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

RAW_FILE = os.path.join(RAW_DIR, "stackoverflow_large_design_questions_raw.csv")
METADATA_FILE = os.path.join(RAW_DIR, "stackoverflow_large_design_tag_metadata.json")

SHOCK_MONTH = "2022-12"
PRE_EXPOSURE_END = "2022-11"
PLACEBO_MONTHS = ["2022-07", "2022-09", "2023-03"]
WILD_BOOTSTRAP_REPS = 999
WILD_BOOTSTRAP_SEED = 42

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
LOW_EXPOSURE_TAGS = {
    "android",
    "apache-spark",
    "docker",
    "firebase",
    "kubernetes",
    "linux",
    "memory-management",
    "multithreading",
}

ROUTINE_PATTERN = re.compile(
    r"\b(?:how|error|exception|convert|parse|replace|split|sort|join|merge|regex|"
    r"formula|string|array|list|column|row|csv|format|plot|numpy|data)\b",
    flags=re.IGNORECASE,
)


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    return pd.read_csv(RAW_FILE)


def load_metadata():
    with open(METADATA_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_text_blob(df):
    return (
        df["title"].fillna("").astype(str).str.strip()
        + " "
        + df["body_text"].fillna("").astype(str).str.strip()
    ).str.strip()


def add_common_fields(frame):
    data = frame.copy(deep=False)
    data["group"] = np.where(data["tag"].isin(HIGH_EXPOSURE_TAGS), "high", "low")
    data["high_tag"] = (data["group"] == "high").astype(int)
    data["post_chatgpt"] = (data["month_id"] >= SHOCK_MONTH).astype(int)
    data["high_post"] = data["high_tag"] * data["post_chatgpt"]
    data["log_answers"] = np.log1p(data["answer_count"].fillna(0))
    data["log_views"] = np.log1p(data["view_count"].fillna(0))
    data["analysis_window"] = "2022-01_to_2023-05"
    if "question_weight" not in data.columns:
        data["question_weight"] = 1.0
    return data


def prepare_question_data(raw):
    selected_set = HIGH_EXPOSURE_TAGS | LOW_EXPOSURE_TAGS
    df = raw.copy()
    df["created_at"] = pd.to_datetime(df["created_at_iso"], utc=True)
    df["month_id"] = df["created_at"].dt.strftime("%Y-%m")
    df["text_blob"] = build_text_blob(df)
    df["routine_keyword_hit"] = df["text_blob"].str.contains(ROUTINE_PATTERN, regex=True, na=False).astype(int)
    df["question_tags_list"] = df["question_tags"].fillna("").astype(object).apply(
        lambda value: [tag for tag in str(value).split(";") if tag]
    )
    df["selected_tags_in_order"] = df["question_tags_list"].apply(
        lambda tags: [tag for tag in tags if tag in selected_set]
    )
    df["selected_tags_on_question"] = df["selected_tags_in_order"].apply(sorted)
    df["selected_tag_overlap"] = df["selected_tags_in_order"].str.len()
    df["keep_single_focal"] = (df["selected_tag_overlap"] == 1).astype(int)

    filtered_columns = [
        "question_id",
        "created_at_iso",
        "score",
        "view_count",
        "answer_count",
        "accepted",
        "owner_user_id",
        "title",
        "body_text",
        "question_tags",
        "created_at",
        "month_id",
        "text_blob",
        "routine_keyword_hit",
        "selected_tags_in_order",
        "selected_tag_overlap",
    ]
    filtered = df.loc[df["keep_single_focal"] == 1, filtered_columns].copy(deep=False)
    filtered["tag"] = [tags[0] for tags in filtered["selected_tags_in_order"].tolist()]
    filtered = filtered.loc[filtered["tag"].isin(selected_set)]
    filtered["question_weight"] = 1.0
    filtered = add_common_fields(filtered)
    return filtered, df


def build_first_focal_assignment(raw_flags):
    frame = raw_flags.loc[raw_flags["selected_tag_overlap"] >= 1].copy(deep=False)
    frame["tag"] = [tags[0] for tags in frame["selected_tags_in_order"].tolist()]
    frame["question_weight"] = 1.0
    return add_common_fields(frame)


def build_fractional_assignment(raw_flags):
    frame = raw_flags.loc[raw_flags["selected_tag_overlap"] >= 1].copy(deep=False)
    repeated_index = np.repeat(frame.index.to_numpy(), frame["selected_tag_overlap"].to_numpy())
    expanded = frame.loc[repeated_index].copy(deep=False)
    expanded["tag"] = [tag for tags in frame["selected_tags_in_order"] for tag in tags]
    frame = expanded
    frame["question_weight"] = 1.0 / frame["selected_tag_overlap"]
    return add_common_fields(frame)


def build_exposure_index(raw_flags):
    base = raw_flags.loc[(raw_flags["selected_tag_overlap"] >= 1) & (raw_flags["month_id"] <= PRE_EXPOSURE_END)].copy(deep=False)
    repeated_index = np.repeat(base.index.to_numpy(), base["selected_tag_overlap"].to_numpy())
    frame = base.loc[repeated_index, ["question_id", "selected_tags_in_order", "text_blob", "routine_keyword_hit"]].copy(deep=False)
    frame["tag"] = [tag for tags in base["selected_tags_in_order"] for tag in tags]
    tag_docs = (
        frame.groupby("tag", as_index=False)
        .agg(
            document=("text_blob", lambda values: " ".join(values.astype(str))),
            pre_questions=("question_id", "count"),
            routine_keyword_share=("routine_keyword_hit", "mean"),
        )
        .sort_values("tag")
        .reset_index(drop=True)
    )
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2500)
    matrix = vectorizer.fit_transform(tag_docs["document"])
    n_components = 2 if matrix.shape[1] > 2 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    components = svd.fit_transform(matrix)
    exposure_raw = components[:, 0]
    routine_share = tag_docs["routine_keyword_share"].to_numpy()
    corr = np.corrcoef(exposure_raw, routine_share)[0, 1] if len(tag_docs) > 1 else 1.0
    if np.isnan(corr) or corr < 0:
        exposure_raw = -exposure_raw
        component_weights = -svd.components_[0]
    else:
        component_weights = svd.components_[0]
    exposure_z = (exposure_raw - exposure_raw.mean()) / exposure_raw.std(ddof=0)
    tag_docs["exposure_raw"] = exposure_raw
    tag_docs["exposure_index"] = exposure_z
    tag_docs["exposure_rank"] = tag_docs["exposure_index"].rank(ascending=False, method="first").astype(int)
    tag_docs["exposure_tercile"] = pd.qcut(
        tag_docs["exposure_index"],
        q=3,
        labels=["Low", "Middle", "High"],
        duplicates="drop",
    )
    feature_names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(component_weights)
    diagnostics = {
        "orientation_correlation_with_routine_share": None if np.isnan(corr) else float(abs(corr)),
        "top_positive_terms": feature_names[order[-15:][::-1]].tolist(),
        "top_negative_terms": feature_names[order[:15]].tolist(),
    }
    return tag_docs, diagnostics


def attach_exposure(frame, exposure):
    merged = frame.merge(
        exposure[
            [
                "tag",
                "exposure_index",
                "exposure_rank",
                "exposure_tercile",
                "pre_questions",
                "routine_keyword_share",
            ]
        ],
        on="tag",
        how="left",
    )
    merged["exposure_post"] = merged["exposure_index"] * merged["post_chatgpt"]
    return merged


def build_panels(questions, weight_col="question_weight"):
    month_map = {month: idx + 1 for idx, month in enumerate(sorted(questions["month_id"].unique()))}
    monthly = (
        questions.groupby(["tag", "group", "month_id", "high_tag", "post_chatgpt"], as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "n_questions": float(frame[weight_col].sum()),
                    "accepted_rate": float(np.average(frame["accepted"], weights=frame[weight_col])),
                    "mean_log_answers": float(np.average(frame["log_answers"], weights=frame[weight_col])),
                    "mean_log_views": float(np.average(frame["log_views"], weights=frame[weight_col])),
                }
            )
        )
        .reset_index(drop=True)
    )
    monthly["log_questions"] = np.log1p(monthly["n_questions"])
    monthly["high_post"] = monthly["high_tag"] * monthly["post_chatgpt"]
    monthly["time_index"] = monthly["month_id"].map(month_map)
    return monthly


def fit_weighted(formula, data, weight_col):
    return smf.wls(formula, data=data, weights=data[weight_col]).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["tag"]},
    )


def fit_plain(formula, data, weight_col):
    return smf.wls(formula, data=data, weights=data[weight_col]).fit()


def extract_term(model, term):
    return {
        "coef": float(model.params.get(term, np.nan)),
        "se": float(model.bse.get(term, np.nan)),
        "pval": float(model.pvalues.get(term, np.nan)),
        "nobs": float(model.nobs),
        "r2": float(getattr(model, "rsquared", np.nan)),
    }


def fit_models(monthly, questions):
    panel_specs = {
        "accepted_rate_primary": "accepted_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
        "mean_log_answers_primary": "mean_log_answers ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
        "log_questions_secondary": "log_questions ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
        "mean_log_views_secondary": "mean_log_views ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
    }
    results = {}
    for label, formula in panel_specs.items():
        model = fit_weighted(formula, monthly, "n_questions")
        results[label] = {"formula": formula, "summary": extract_term(model, "high_post"), "fit": model}

    question_specs = {
        "accepted_question_hc1": "accepted ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
        "log_answers_question_hc1": "log_answers ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
    }
    for label, formula in question_specs.items():
        model = smf.ols(formula, data=questions.assign(time_index=questions["month_id"].map({m: i + 1 for i, m in enumerate(sorted(questions['month_id'].unique()))}))).fit(cov_type="HC1")
        results[label] = {"formula": formula, "summary": extract_term(model, "high_post"), "fit": model}
    return results


def fit_continuous_exposure_models(monthly, fractional_monthly):
    results = {}
    specs = {
        "accepted_rate_continuous_primary": {
            "data": monthly,
            "formula": "accepted_rate ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
            "term": "exposure_post",
        },
        "mean_log_answers_continuous_primary": {
            "data": monthly,
            "formula": "mean_log_answers ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
            "term": "exposure_post",
        },
        "accepted_rate_continuous_fractional": {
            "data": fractional_monthly,
            "formula": "accepted_rate ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
            "term": "exposure_post",
        },
        "mean_log_answers_continuous_fractional": {
            "data": fractional_monthly,
            "formula": "mean_log_answers ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
            "term": "exposure_post",
        },
    }
    for label, spec in specs.items():
        model = fit_weighted(spec["formula"], spec["data"], "n_questions")
        results[label] = {"formula": spec["formula"], "summary": extract_term(model, spec["term"]), "fit": model}
    return results


def fit_assignment_robustness(first_focal_questions, fractional_questions):
    first_monthly = build_panels(first_focal_questions)
    fractional_monthly = build_panels(fractional_questions)
    results = {}
    for label, formula in {
        "accepted_rate_first_focal": "accepted_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
        "mean_log_answers_first_focal": "mean_log_answers ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
        "accepted_rate_fractional_all_tag": "accepted_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
        "mean_log_answers_fractional_all_tag": "mean_log_answers ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
    }.items():
        data = first_monthly if "first_focal" in label else fractional_monthly
        model = fit_weighted(formula, data, "n_questions")
        results[label] = {"formula": formula, "summary": extract_term(model, "high_post"), "fit": model}
    return results, {"first_focal": first_monthly, "fractional": fractional_monthly}


def placebo_tests(monthly):
    rows = []
    for placebo_month in PLACEBO_MONTHS:
        frame = monthly.copy()
        frame["placebo_post"] = (frame["month_id"] >= placebo_month).astype(int)
        for outcome in ["accepted_rate", "mean_log_answers", "log_questions"]:
            model = fit_weighted(
                f"{outcome} ~ high_tag:placebo_post + C(tag):time_index + C(tag) + C(month_id)",
                frame,
                "n_questions",
            )
            rows.append(
                {
                    "outcome": outcome,
                    "placebo_month": placebo_month,
                    "coef": float(model.params.get("high_tag:placebo_post", np.nan)),
                    "se": float(model.bse.get("high_tag:placebo_post", np.nan)),
                    "pval": float(model.pvalues.get("high_tag:placebo_post", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def wild_cluster_bootstrap(specs):
    rng = np.random.default_rng(WILD_BOOTSTRAP_SEED)
    rows = []
    for spec in specs:
        data = spec["data"]
        term = spec["term"]
        clusters = data["tag"].astype(str).to_numpy()
        unique_clusters = sorted(set(clusters))
        full = fit_weighted(spec["full_formula"], data, "n_questions")
        restricted = fit_plain(spec["restricted_formula"], data, "n_questions")
        observed_t = float(full.params[term] / full.bse[term])
        t_stars = []
        for _ in range(WILD_BOOTSTRAP_REPS):
            weights = {cluster: rng.choice([-1.0, 1.0]) for cluster in unique_clusters}
            boot = data.copy()
            boot[spec["outcome"]] = (
                restricted.fittedvalues.to_numpy()
                + restricted.resid.to_numpy() * np.array([weights[c] for c in clusters])
            )
            try:
                boot_model = fit_weighted(spec["full_formula"], boot, "n_questions")
                t_star = float(boot_model.params[term] / boot_model.bse[term])
                if np.isfinite(t_star):
                    t_stars.append(t_star)
            except Exception:
                continue
        rows.append(
            {
                "specification": spec["specification"],
                "coef": float(full.params[term]),
                "cluster_pval": float(full.pvalues[term]),
                "wild_cluster_pval": float(np.mean(np.abs(t_stars) >= abs(observed_t))),
                "successful_draws": int(len(t_stars)),
            }
        )
    return pd.DataFrame(rows)


def leave_one_out_robustness(monthly):
    rows = []
    formula = "accepted_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)"
    for tag in sorted(monthly["tag"].unique()):
        data = monthly.loc[monthly["tag"] != tag].copy()
        model = fit_weighted(formula, data, "n_questions")
        rows.append(
            {
                "dropped_tag": tag,
                "dropped_group": "high" if tag in HIGH_EXPOSURE_TAGS else "low",
                "coef": float(model.params["high_post"]),
                "se": float(model.bse["high_post"]),
                "pval": float(model.pvalues["high_post"]),
            }
        )
    return pd.DataFrame(rows)


def partition_permutation(monthly):
    tags = sorted(monthly["tag"].unique())
    anchor_tag = "bash"
    remaining = [tag for tag in tags if tag != anchor_tag]
    rows = []
    formula = "accepted_rate ~ perm_high_post + C(tag):time_index + C(tag) + C(month_id)"
    observed_key = tuple(sorted(HIGH_EXPOSURE_TAGS))
    for combo in combinations(remaining, len(HIGH_EXPOSURE_TAGS) - 1):
        high_set = tuple(sorted((anchor_tag,) + combo))
        data = monthly.copy()
        data["perm_high_tag"] = data["tag"].isin(high_set).astype(int)
        data["perm_high_post"] = data["perm_high_tag"] * data["post_chatgpt"]
        model = fit_weighted(formula, data, "n_questions")
        rows.append(
            {
                "high_tags": ";".join(high_set),
                "coef": float(model.params["perm_high_post"]),
                "se": float(model.bse["perm_high_post"]),
                "pval": float(model.pvalues["perm_high_post"]),
                "is_observed_split": int(high_set == observed_key),
            }
        )
    distribution = pd.DataFrame(rows)
    observed_row = distribution.loc[distribution["is_observed_split"] == 1].iloc[0]
    summary = {
        "anchor_tag": anchor_tag,
        "n_unique_partitions": int(len(distribution)),
        "observed_coef": float(observed_row["coef"]),
        "observed_pval": float(observed_row["pval"]),
        "coef_percentile_from_bottom": float(100.0 * (distribution["coef"] <= observed_row["coef"]).mean()),
        "pval_percentile_from_bottom": float(100.0 * (distribution["pval"] <= observed_row["pval"]).mean()),
        "share_negative_coefficients": float((distribution["coef"] < 0).mean()),
        "share_coefficients_at_least_as_negative_as_observed": float(
            (distribution["coef"] <= observed_row["coef"]).mean()
        ),
    }
    return distribution, summary


def exposure_tercile_summary(monthly):
    frame = monthly.copy()
    summary = (
        frame.groupby(["exposure_tercile", "post_chatgpt"], observed=False)
        .apply(
            lambda data: pd.Series(
                {
                    "weighted_accepted_rate": float(np.average(data["accepted_rate"], weights=data["n_questions"])),
                    "weighted_mean_log_answers": float(np.average(data["mean_log_answers"], weights=data["n_questions"])),
                    "weighted_log_questions": float(np.average(data["log_questions"], weights=data["n_questions"])),
                }
            )
        )
        .reset_index()
    )
    summary["period"] = np.where(summary["post_chatgpt"] == 1, "post", "pre")
    return summary


def save_outputs(
    questions,
    raw_flags,
    monthly,
    first_focal_monthly,
    fractional_monthly,
    exposure,
    exposure_tercile,
    exposure_diagnostics,
    placebo,
    wild_bootstrap,
    leave_one_out,
    partition_distribution,
    partition_summary,
    results,
    metadata,
):
    questions.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_question_level_enriched.csv"), index=False)
    raw_flags.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_raw_with_attrition_flags.csv"), index=False)
    monthly.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_tag_month_panel.csv"), index=False)
    first_focal_monthly.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_first_focal_tag_month_panel.csv"), index=False)
    fractional_monthly.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_fractional_tag_month_panel.csv"), index=False)
    exposure.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_exposure_index.csv"), index=False)
    exposure_tercile.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_exposure_tercile_summary.csv"), index=False)
    placebo.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_placebo_tests.csv"), index=False)
    wild_bootstrap.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_wild_cluster_bootstrap.csv"), index=False)
    leave_one_out.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_leave_one_out.csv"), index=False)
    partition_distribution.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_partition_permutations.csv"), index=False)
    with open(
        os.path.join(PROCESSED_DIR, "mainstream_design_partition_permutation_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(partition_summary, handle, indent=2)
    with open(
        os.path.join(PROCESSED_DIR, "mainstream_design_exposure_diagnostics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(exposure_diagnostics, handle, indent=2)

    tag_table = pd.DataFrame(
        [{"tag": tag, "group": "high"} for tag in sorted(HIGH_EXPOSURE_TAGS)]
        + [{"tag": tag, "group": "low"} for tag in sorted(LOW_EXPOSURE_TAGS)]
    )
    tag_table.to_csv(os.path.join(PROCESSED_DIR, "mainstream_design_tag_classification.csv"), index=False)

    serializable = {
        "metadata": metadata,
        "results": {key: value["summary"] for key, value in results.items()},
        "high_tags": sorted(HIGH_EXPOSURE_TAGS),
        "low_tags": sorted(LOW_EXPOSURE_TAGS),
        "partition_permutation_summary": partition_summary,
        "exposure_diagnostics": exposure_diagnostics,
    }
    with open(os.path.join(PROCESSED_DIR, "mainstream_design_regression_results.json"), "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    summary = []
    for key, value in results.items():
        payload = value["summary"]
        summary.append(
            {
                "model": key,
                "coef": round(payload["coef"], 4),
                "se": round(payload["se"], 4),
                "pval": round(payload["pval"], 4),
                "nobs": int(payload["nobs"]),
            }
        )
    with open(os.path.join(PROCESSED_DIR, "mainstream_design_summary_tables.md"), "w", encoding="utf-8") as handle:
        handle.write("# Mainstream Tag Design Summary Tables\n\n")
        handle.write(pd.DataFrame(summary).to_markdown(index=False))
        handle.write("\n\n## Placebo Tests\n\n")
        handle.write(placebo.to_markdown(index=False))
        handle.write("\n\n## Wild Cluster Bootstrap\n\n")
        handle.write(wild_bootstrap.to_markdown(index=False))
        handle.write("\n\n## Leave-One-Out Robustness\n\n")
        handle.write(leave_one_out.to_markdown(index=False))
        handle.write("\n\n## Partition Permutation Summary\n\n")
        handle.write(pd.DataFrame([partition_summary]).to_markdown(index=False))
        handle.write("\n\n## Exposure Tercile Summary\n\n")
        handle.write(exposure_tercile.to_markdown(index=False))


def plot_trends(monthly):
    accepted = (
        monthly.groupby(["month_id", "group"])
        .apply(lambda frame: np.average(frame["accepted_rate"], weights=frame["n_questions"]))
        .rename("accepted_rate")
        .reset_index()
        .pivot(index="month_id", columns="group", values="accepted_rate")
        .sort_index()
    )
    answers = (
        monthly.groupby(["month_id", "group"])
        .apply(lambda frame: np.average(frame["mean_log_answers"], weights=frame["n_questions"]))
        .rename("mean_log_answers")
        .reset_index()
        .pivot(index="month_id", columns="group", values="mean_log_answers")
        .sort_index()
    )
    counts = (
        monthly.groupby(["month_id", "group"], as_index=False)["n_questions"]
        .sum()
        .pivot(index="month_id", columns="group", values="n_questions")
        .sort_index()
    )
    palette = {"high": "#e45756", "low": "#4c78a8"}
    for frame, ylabel, title, filename in [
        (accepted, "Accepted-answer rate", "Accepted-Answer Rates Around the ChatGPT Shock", "mainstream_design_accepted_rate_trends.png"),
        (answers, "Mean log(1 + answers)", "Answer Depth Around the ChatGPT Shock", "mainstream_design_answer_depth_trends.png"),
        (counts, "Questions created", "Question Volume Around the ChatGPT Shock", "mainstream_design_question_volume_trends.png"),
    ]:
        plt.figure(figsize=(11, 5))
        for group in ["high", "low"]:
            plt.plot(frame.index, frame[group], marker="o", linewidth=2, label=group.title(), color=palette[group])
        plt.axvline(SHOCK_MONTH, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Question creation month")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=220)
        plt.close()


def build_stats(
    raw_flags,
    questions,
    monthly,
    results,
    placebo,
    wild_bootstrap,
    leave_one_out,
    partition_summary,
    exposure,
    exposure_diagnostics,
):
    stats = {
        "raw_rows": int(len(raw_flags)),
        "analysis_rows": int(len(questions)),
        "raw_unique_questions": int(raw_flags["question_id"].nunique()),
        "analysis_unique_questions": int(questions["question_id"].nunique()),
        "dropped_multi_selected_rows": int((raw_flags["selected_tag_overlap"] > 1).sum()),
        "unique_tags": int(questions["tag"].nunique()),
        "accepted_rate_primary": results["accepted_rate_primary"]["summary"],
        "mean_log_answers_primary": results["mean_log_answers_primary"]["summary"],
        "accepted_rate_continuous_primary": results["accepted_rate_continuous_primary"]["summary"],
        "mean_log_answers_continuous_primary": results["mean_log_answers_continuous_primary"]["summary"],
        "accepted_rate_first_focal": results["accepted_rate_first_focal"]["summary"],
        "mean_log_answers_first_focal": results["mean_log_answers_first_focal"]["summary"],
        "accepted_rate_fractional_all_tag": results["accepted_rate_fractional_all_tag"]["summary"],
        "mean_log_answers_fractional_all_tag": results["mean_log_answers_fractional_all_tag"]["summary"],
        "accepted_rate_continuous_fractional": results["accepted_rate_continuous_fractional"]["summary"],
        "mean_log_answers_continuous_fractional": results["mean_log_answers_continuous_fractional"]["summary"],
        "log_questions_secondary": results["log_questions_secondary"]["summary"],
        "placebo_tests": placebo.to_dict(orient="records"),
        "wild_cluster_bootstrap": wild_bootstrap.to_dict(orient="records"),
        "leave_one_out": leave_one_out.to_dict(orient="records"),
        "partition_permutation_summary": partition_summary,
        "exposure_index": exposure.to_dict(orient="records"),
        "exposure_diagnostics": exposure_diagnostics,
    }
    with open(os.path.join(PROCESSED_DIR, "mainstream_design_paper_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def main():
    ensure_dirs()
    raw = load_data()
    metadata = load_metadata()
    questions, raw_flags = prepare_question_data(raw)
    first_focal_questions = build_first_focal_assignment(raw_flags)
    fractional_questions = build_fractional_assignment(raw_flags)
    exposure, exposure_diagnostics = build_exposure_index(raw_flags)
    questions = attach_exposure(questions, exposure)
    first_focal_questions = attach_exposure(first_focal_questions, exposure)
    fractional_questions = attach_exposure(fractional_questions, exposure)
    monthly = attach_exposure(build_panels(questions), exposure)
    results = fit_models(monthly, questions)
    assignment_results, assignment_panels = fit_assignment_robustness(first_focal_questions, fractional_questions)
    results.update(assignment_results)
    assignment_panels["first_focal"] = attach_exposure(assignment_panels["first_focal"], exposure)
    assignment_panels["fractional"] = attach_exposure(assignment_panels["fractional"], exposure)
    results.update(fit_continuous_exposure_models(monthly, assignment_panels["fractional"]))
    exposure_tercile = exposure_tercile_summary(monthly)
    placebo = placebo_tests(monthly)
    wild_bootstrap = wild_cluster_bootstrap(
        [
            {
                "specification": "accepted_rate_primary",
                "outcome": "accepted_rate",
                "data": monthly,
                "full_formula": "accepted_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
                "restricted_formula": "accepted_rate ~ C(tag):time_index + C(tag) + C(month_id)",
            },
            {
                "specification": "mean_log_answers_primary",
                "outcome": "mean_log_answers",
                "data": monthly,
                "full_formula": "mean_log_answers ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
                "restricted_formula": "mean_log_answers ~ C(tag):time_index + C(tag) + C(month_id)",
            },
            {
                "specification": "accepted_rate_first_focal",
                "outcome": "accepted_rate",
                "data": assignment_panels["first_focal"],
                "full_formula": "accepted_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
                "restricted_formula": "accepted_rate ~ C(tag):time_index + C(tag) + C(month_id)",
            },
            {
                "specification": "accepted_rate_fractional_all_tag",
                "outcome": "accepted_rate",
                "data": assignment_panels["fractional"],
                "full_formula": "accepted_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)",
                "restricted_formula": "accepted_rate ~ C(tag):time_index + C(tag) + C(month_id)",
            },
            {
                "specification": "accepted_rate_continuous_primary",
                "outcome": "accepted_rate",
                "data": monthly,
                "full_formula": "accepted_rate ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
                "restricted_formula": "accepted_rate ~ C(tag):time_index + C(tag) + C(month_id)",
            },
            {
                "specification": "accepted_rate_continuous_fractional",
                "outcome": "accepted_rate",
                "data": assignment_panels["fractional"],
                "full_formula": "accepted_rate ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
                "restricted_formula": "accepted_rate ~ C(tag):time_index + C(tag) + C(month_id)",
            },
        ]
    )
    leave_one_out = leave_one_out_robustness(monthly)
    partition_distribution, partition_summary = partition_permutation(monthly)
    save_outputs(
        questions,
        raw_flags,
        monthly,
        assignment_panels["first_focal"],
        assignment_panels["fractional"],
        exposure,
        exposure_tercile,
        exposure_diagnostics,
        placebo,
        wild_bootstrap,
        leave_one_out,
        partition_distribution,
        partition_summary,
        results,
        metadata,
    )
    plot_trends(monthly)
    stats = build_stats(
        raw_flags,
        questions,
        monthly,
        results,
        placebo,
        wild_bootstrap,
        leave_one_out,
        partition_summary,
        exposure,
        exposure_diagnostics,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
