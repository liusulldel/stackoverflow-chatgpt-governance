import json
import os
import re
from itertools import product

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
TAG_METADATA_FILE = os.path.join(RAW_DIR, "stackoverflow_large_design_tag_metadata.json")

REFERENCE_MONTH = "2022-11"
SHOCK_MONTH = "2022-12"
PRE_EXPOSURE_END = "2022-10"
POLICY_START = "2022-12-01"
POLICY_SPLIT = "2022-12-05"
POLICY_END = "2022-12-07"
PLACEBO_MONTHS = ["2022-07", "2022-09", "2023-03"]
WILD_BOOTSTRAP_REPS = 999
WILD_BOOTSTRAP_SEED = 42
PERMUTATION_REPS = 5000
PERMUTATION_SEED = 42

ROUTINE_PATTERN = re.compile(
    r"\b(?:how|error|exception|convert|parse|replace|split|sort|join|merge|regex|"
    r"formula|string|array|list|column|row|csv|format|plot|numpy|data)\b",
    flags=re.IGNORECASE,
)


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_raw_data():
    return pd.read_csv(RAW_FILE)


def load_tag_metadata():
    with open(TAG_METADATA_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_text_column(df):
    if "body_text" in df.columns:
        text_source = df["body_text"]
    elif "excerpt" in df.columns:
        text_source = df["excerpt"]
    else:
        text_source = pd.Series([""] * len(df), index=df.index)
    return (
        df["title"].fillna("").astype(str).str.strip()
        + " "
        + text_source.fillna("").astype(str).str.strip()
    ).str.strip()


def prepare_question_data(raw, selected_tags):
    df = raw.copy()
    df["created_at"] = pd.to_datetime(df["created_at_iso"], utc=True)
    df["month_id"] = df["created_at"].dt.strftime("%Y-%m")
    df["day_id"] = df["created_at"].dt.strftime("%Y-%m-%d")
    df["text_blob"] = build_text_column(df)
    df["question_tags_list"] = (
        df["question_tags"].fillna("").astype(str).str.split(";").apply(lambda values: [v for v in values if v])
    )
    selected_set = set(selected_tags)
    df["selected_tags_on_question"] = df["question_tags_list"].apply(
        lambda tags: sorted(selected_set.intersection(tags))
    )
    df["selected_tag_overlap"] = df["selected_tags_on_question"].str.len()
    df["single_focal_tag"] = df["selected_tags_on_question"].apply(
        lambda tags: tags[0] if len(tags) == 1 else None
    )
    df["keep_single_focal"] = (df["selected_tag_overlap"] == 1).astype(int)

    filtered = df.loc[df["keep_single_focal"] == 1].copy()
    filtered = filtered.drop_duplicates(subset=["question_id"])
    filtered["tag"] = filtered["single_focal_tag"]
    filtered["post_chatgpt"] = (filtered["month_id"] >= SHOCK_MONTH).astype(int)
    filtered["log_views"] = np.log1p(filtered["view_count"].fillna(0))
    filtered["log_answers"] = np.log1p(filtered["answer_count"].fillna(0))
    filtered["post_policy"] = (
        (filtered["day_id"] >= POLICY_SPLIT) & (filtered["day_id"] <= POLICY_END)
    ).astype(int)
    filtered["in_policy_window"] = (
        (filtered["day_id"] >= POLICY_START) & (filtered["day_id"] <= POLICY_END)
    ).astype(int)
    filtered["routine_keyword_hit"] = filtered["text_blob"].str.contains(
        ROUTINE_PATTERN, regex=True, na=False
    ).astype(int)
    return filtered, df


def build_exposure_index(questions):
    pre = questions.loc[questions["month_id"] <= PRE_EXPOSURE_END].copy()
    tag_docs = (
        pre.groupby("tag", as_index=False)
        .agg(
            document=("text_blob", lambda series: " ".join(series.astype(str))),
            pre_questions=("question_id", "count"),
            routine_keyword_share=("routine_keyword_hit", "mean"),
        )
        .sort_values("tag")
        .reset_index(drop=True)
    )
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2500)
    matrix = vectorizer.fit_transform(tag_docs["document"])
    svd = TruncatedSVD(n_components=2, random_state=42)
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
    tag_docs["exposure_group"] = pd.qcut(
        tag_docs["exposure_index"],
        q=3,
        labels=["Low Exposure", "Mid Exposure", "High Exposure"],
    )

    feature_names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(component_weights)
    diagnostics = {
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
        "top_positive_terms": feature_names[order[-15:][::-1]].tolist(),
        "top_negative_terms": feature_names[order[:15]].tolist(),
        "orientation_correlation_with_routine_share": None if np.isnan(corr) else float(abs(corr)),
    }
    return tag_docs, diagnostics


def attach_exposure(questions, exposure):
    merged = questions.merge(
        exposure[
            [
                "tag",
                "exposure_index",
                "exposure_group",
                "exposure_rank",
                "pre_questions",
                "routine_keyword_share",
            ]
        ],
        on="tag",
        how="left",
    )
    merged["exposure_post"] = merged["exposure_index"] * merged["post_chatgpt"]
    return merged


def build_panels(questions):
    month_order = sorted(questions["month_id"].unique())
    month_map = {month: idx + 1 for idx, month in enumerate(month_order)}
    monthly = (
        questions.groupby(["tag", "month_id", "exposure_index", "exposure_group", "exposure_rank"], as_index=False)
        .agg(
            n_questions=("question_id", "count"),
            accepted_rate=("accepted", "mean"),
            mean_answer_count=("answer_count", "mean"),
            mean_score=("score", "mean"),
            mean_log_views=("log_views", "mean"),
            mean_log_answers=("log_answers", "mean"),
            routine_keyword_share=("routine_keyword_hit", "mean"),
        )
    )
    monthly["post_chatgpt"] = (monthly["month_id"] >= SHOCK_MONTH).astype(int)
    monthly["log_questions"] = np.log1p(monthly["n_questions"])
    monthly["time_index"] = monthly["month_id"].map(month_map)
    shock_index = month_map[SHOCK_MONTH]
    monthly["time_post"] = monthly["time_index"].sub(shock_index - 1).clip(lower=0)
    monthly["month_order"] = pd.Categorical(monthly["month_id"], categories=month_order, ordered=True)
    monthly["exposure_post"] = monthly["exposure_index"] * monthly["post_chatgpt"]

    policy = questions.loc[questions["in_policy_window"] == 1].copy()
    policy["policy_period"] = np.where(policy["post_policy"] == 1, "Dec 5-7", "Dec 1-4")
    return monthly, policy


def fit_model(formula, data, weight_col=None, cov_type="cluster", cluster_col="tag"):
    if weight_col:
        model = smf.wls(formula, data=data, weights=data[weight_col])
    else:
        model = smf.ols(formula, data=data)
    if cov_type == "cluster":
        return model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
    return model.fit(cov_type=cov_type)


def fit_plain_model(formula, data, weight_col=None):
    if weight_col:
        return smf.wls(formula, data=data, weights=data[weight_col]).fit()
    return smf.ols(formula, data=data).fit()


def extract_term(model, term):
    return {
        "coef": float(model.params.get(term, np.nan)),
        "se": float(model.bse.get(term, np.nan)),
        "pval": float(model.pvalues.get(term, np.nan)),
        "nobs": float(model.nobs),
        "r2": float(getattr(model, "rsquared", np.nan)),
    }


def fit_main_models(monthly, policy, questions):
    results = {}
    panel_specs = {
        "baseline_log_questions": {
            "formula": "log_questions ~ exposure_post + C(tag) + C(month_id)",
            "term": "exposure_post",
            "weight_col": None,
        },
        "baseline_mean_log_views": {
            "formula": "mean_log_views ~ exposure_post + C(tag) + C(month_id)",
            "term": "exposure_post",
            "weight_col": "n_questions",
        },
        "baseline_accepted_rate": {
            "formula": "accepted_rate ~ exposure_post + C(tag) + C(month_id)",
            "term": "exposure_post",
            "weight_col": "n_questions",
        },
        "baseline_mean_log_answers": {
            "formula": "mean_log_answers ~ exposure_post + C(tag) + C(month_id)",
            "term": "exposure_post",
            "weight_col": "n_questions",
        },
    }
    for label, spec in panel_specs.items():
        model = fit_model(spec["formula"], monthly, weight_col=spec["weight_col"])
        results[label] = {"term": spec["term"], "fit": model, "summary": extract_term(model, spec["term"])}

    segmented_specs = {
        "segmented_log_questions": {
            "formula": "log_questions ~ exposure_index:time_index + exposure_index:post_chatgpt + exposure_index:time_post + C(tag) + C(month_id)",
            "level_term": "exposure_index:post_chatgpt",
            "slope_term": "exposure_index:time_post",
            "pretrend_term": "exposure_index:time_index",
            "weight_col": None,
        },
        "segmented_mean_log_views": {
            "formula": "mean_log_views ~ exposure_index:time_index + exposure_index:post_chatgpt + exposure_index:time_post + C(tag) + C(month_id)",
            "level_term": "exposure_index:post_chatgpt",
            "slope_term": "exposure_index:time_post",
            "pretrend_term": "exposure_index:time_index",
            "weight_col": "n_questions",
        },
    }
    for label, spec in segmented_specs.items():
        model = fit_model(spec["formula"], monthly, weight_col=spec["weight_col"])
        results[label] = {
            "fit": model,
            "summary": {
                "level_break": extract_term(model, spec["level_term"]),
                "post_slope": extract_term(model, spec["slope_term"]),
                "pretrend_slope": extract_term(model, spec["pretrend_term"]),
            },
        }

    question_level_specs = {
        "question_log_views": "log_views ~ exposure_post + C(tag) + C(month_id)",
        "question_accepted": "accepted ~ exposure_post + C(tag) + C(month_id)",
    }
    for label, formula in question_level_specs.items():
        model = fit_model(formula, questions, cov_type="cluster")
        results[label] = {"term": "exposure_post", "fit": model, "summary": extract_term(model, "exposure_post")}

    if not policy.empty:
        policy_model = fit_model("accepted ~ exposure_index:post_policy + C(tag)", policy, cov_type="cluster")
        results["policy_question_accepted"] = {
            "term": "exposure_index:post_policy",
            "fit": policy_model,
            "summary": extract_term(policy_model, "exposure_index:post_policy"),
        }
    return results


def build_event_study(monthly, outcome, weight_col=None):
    formula = (
        f"{outcome} ~ C(month_order, Treatment(reference='{REFERENCE_MONTH}')):exposure_index "
        "+ C(tag) + C(month_order)"
    )
    model = fit_model(formula, monthly, weight_col=weight_col, cov_type="cluster")
    rows = []
    for month in monthly["month_order"].cat.categories:
        if month == REFERENCE_MONTH:
            coef = 0.0
            se = 0.0
            pval = np.nan
        else:
            term = f"C(month_order, Treatment(reference='{REFERENCE_MONTH}'))[T.{month}]:exposure_index"
            coef = float(model.params.get(term, np.nan))
            se = float(model.bse.get(term, np.nan))
            pval = float(model.pvalues.get(term, np.nan))
        rows.append(
            {
                "month": month,
                "coef": coef,
                "se": se,
                "pval": pval,
                "ci_low": coef - 1.96 * se,
                "ci_high": coef + 1.96 * se,
                "post_chatgpt": int(month >= SHOCK_MONTH),
                "outcome": outcome,
            }
        )
    param_names = list(model.params.index)
    pre_terms = [
        f"C(month_order, Treatment(reference='{REFERENCE_MONTH}'))[T.{month}]:exposure_index"
        for month in monthly["month_order"].cat.categories
        if month < REFERENCE_MONTH
    ]
    post_terms = [
        f"C(month_order, Treatment(reference='{REFERENCE_MONTH}'))[T.{month}]:exposure_index"
        for month in monthly["month_order"].cat.categories
        if month > REFERENCE_MONTH
    ]

    def joint_pvalue(term_names):
        if not term_names:
            return np.nan
        matrix = []
        for term_name in term_names:
            if term_name in param_names:
                row = np.zeros(len(param_names))
                row[param_names.index(term_name)] = 1.0
                matrix.append(row)
        if not matrix:
            return np.nan
        return float(model.f_test(np.vstack(matrix)).pvalue)

    tests = {
        "pre_joint_pval": joint_pvalue(pre_terms),
        "post_joint_pval": joint_pvalue(post_terms),
    }
    return pd.DataFrame(rows), tests


def placebo_break_tests(monthly):
    rows = []
    for placebo_month, outcome in product(PLACEBO_MONTHS, ["log_questions", "mean_log_views"]):
        frame = monthly.copy()
        frame["placebo_post"] = (frame["month_id"] >= placebo_month).astype(int)
        shock_index = int(frame.loc[frame["month_id"] == placebo_month, "time_index"].iloc[0])
        frame["placebo_time_post"] = frame["time_index"].sub(shock_index - 1).clip(lower=0)
        weight_col = "n_questions" if outcome == "mean_log_views" else None
        model = fit_model(
            f"{outcome} ~ exposure_index:time_index + exposure_index:placebo_post + exposure_index:placebo_time_post + C(tag) + C(month_id)",
            frame,
            weight_col=weight_col,
        )
        rows.append(
            {
                "outcome": outcome,
                "placebo_month": placebo_month,
                "level_coef": float(model.params.get("exposure_index:placebo_post", np.nan)),
                "level_pval": float(model.pvalues.get("exposure_index:placebo_post", np.nan)),
                "slope_coef": float(model.params.get("exposure_index:placebo_time_post", np.nan)),
                "slope_pval": float(model.pvalues.get("exposure_index:placebo_time_post", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def permutation_tests(monthly):
    rng = np.random.default_rng(PERMUTATION_SEED)
    tags = monthly[["tag", "exposure_index"]].drop_duplicates().sort_values("tag")
    tag_values = tags["tag"].tolist()
    exposure_values = tags["exposure_index"].to_numpy()
    observed_specs = {
        "baseline_log_questions": {
            "formula": "log_questions ~ exposure_post + C(tag) + C(month_id)",
            "term": "exposure_post",
            "weight_col": None,
        },
        "segmented_mean_log_views_level_break": {
            "formula": "mean_log_views ~ exposure_index:time_index + exposure_index:post_chatgpt + exposure_index:time_post + C(tag) + C(month_id)",
            "term": "exposure_index:post_chatgpt",
            "weight_col": "n_questions",
        },
    }
    rows = []
    for label, spec in observed_specs.items():
        observed_model = fit_model(spec["formula"], monthly, weight_col=spec["weight_col"], cov_type="nonrobust")
        observed_coef = float(observed_model.params.get(spec["term"], np.nan))
        permuted = []
        for _ in range(PERMUTATION_REPS):
            shuffled = rng.permutation(exposure_values)
            mapping = dict(zip(tag_values, shuffled))
            frame = monthly.copy()
            frame["perm_exposure"] = frame["tag"].map(mapping)
            frame["perm_exposure_post"] = frame["perm_exposure"] * frame["post_chatgpt"]
            if label == "baseline_log_questions":
                model = fit_model("log_questions ~ perm_exposure_post + C(tag) + C(month_id)", frame, cov_type="nonrobust")
                permuted.append(float(model.params.get("perm_exposure_post", np.nan)))
            else:
                model = fit_model(
                    "mean_log_views ~ perm_exposure:time_index + perm_exposure:post_chatgpt + perm_exposure:time_post + C(tag) + C(month_id)",
                    frame,
                    weight_col="n_questions",
                    cov_type="nonrobust",
                )
                permuted.append(float(model.params.get("perm_exposure:post_chatgpt", np.nan)))
        permuted = np.array(permuted)
        rows.append(
            {
                "specification": label,
                "observed_coef": observed_coef,
                "abs_permutation_pval": float(np.mean(np.abs(permuted) >= abs(observed_coef))),
                "one_sided_pval": float(np.mean(permuted <= observed_coef)),
                "permutation_reps": PERMUTATION_REPS,
            }
        )
    return pd.DataFrame(rows)


def wild_cluster_bootstrap(monthly):
    rng = np.random.default_rng(WILD_BOOTSTRAP_SEED)
    specs = [
        {
            "specification": "baseline_log_questions",
            "full_formula": "log_questions ~ exposure_post + C(tag) + C(month_id)",
            "restricted_formula": "log_questions ~ C(tag) + C(month_id)",
            "term": "exposure_post",
            "weight_col": None,
            "outcome": "log_questions",
        },
        {
            "specification": "segmented_mean_log_views_level_break",
            "full_formula": "mean_log_views ~ exposure_index:time_index + exposure_index:post_chatgpt + exposure_index:time_post + C(tag) + C(month_id)",
            "restricted_formula": "mean_log_views ~ exposure_index:time_index + exposure_index:time_post + C(tag) + C(month_id)",
            "term": "exposure_index:post_chatgpt",
            "weight_col": "n_questions",
            "outcome": "mean_log_views",
        },
        {
            "specification": "segmented_mean_log_views_post_slope",
            "full_formula": "mean_log_views ~ exposure_index:time_index + exposure_index:post_chatgpt + exposure_index:time_post + C(tag) + C(month_id)",
            "restricted_formula": "mean_log_views ~ exposure_index:time_index + exposure_index:post_chatgpt + C(tag) + C(month_id)",
            "term": "exposure_index:time_post",
            "weight_col": "n_questions",
            "outcome": "mean_log_views",
        },
    ]
    rows = []
    for spec in specs:
        full_model = fit_model(spec["full_formula"], monthly, weight_col=spec["weight_col"], cov_type="cluster")
        restricted = fit_plain_model(spec["restricted_formula"], monthly, weight_col=spec["weight_col"])
        observed_t = float(full_model.params[spec["term"]] / full_model.bse[spec["term"]])
        clusters = monthly["tag"].astype(str).to_numpy()
        unique_clusters = sorted(set(clusters))
        t_stars = []
        for _ in range(WILD_BOOTSTRAP_REPS):
            cluster_weights = {cluster: rng.choice([-1.0, 1.0]) for cluster in unique_clusters}
            bootstrap = monthly.copy()
            bootstrap[spec["outcome"]] = (
                restricted.fittedvalues.to_numpy()
                + restricted.resid.to_numpy() * np.array([cluster_weights[c] for c in clusters])
            )
            try:
                bootstrap_model = fit_model(spec["full_formula"], bootstrap, weight_col=spec["weight_col"], cov_type="cluster")
                t_value = float(bootstrap_model.params[spec["term"]] / bootstrap_model.bse[spec["term"]])
                if np.isfinite(t_value):
                    t_stars.append(t_value)
            except Exception:
                continue
        rows.append(
            {
                "specification": spec["specification"],
                "term": spec["term"],
                "coef": float(full_model.params[spec["term"]]),
                "cluster_se": float(full_model.bse[spec["term"]]),
                "cluster_pval": float(full_model.pvalues[spec["term"]]),
                "wild_cluster_pval": float(np.mean(np.abs(t_stars) >= abs(observed_t))),
                "successful_draws": int(len(t_stars)),
                "reps_requested": WILD_BOOTSTRAP_REPS,
            }
        )
    return pd.DataFrame(rows)


def save_processed_data(
    questions,
    raw_with_attrition,
    monthly,
    policy,
    exposure,
    event_views,
    event_questions,
    placebo,
    permutations,
    wild_bootstrap,
):
    raw_with_attrition.to_csv(os.path.join(PROCESSED_DIR, "large_design_raw_with_attrition_flags.csv"), index=False)
    questions.to_csv(os.path.join(PROCESSED_DIR, "large_design_question_level_enriched.csv"), index=False)
    monthly.to_csv(os.path.join(PROCESSED_DIR, "large_design_tag_month_panel.csv"), index=False)
    policy.to_csv(os.path.join(PROCESSED_DIR, "large_design_policy_window_questions.csv"), index=False)
    exposure.to_csv(os.path.join(PROCESSED_DIR, "large_design_exposure_index.csv"), index=False)
    event_views.to_csv(os.path.join(PROCESSED_DIR, "large_design_event_study_views.csv"), index=False)
    event_questions.to_csv(os.path.join(PROCESSED_DIR, "large_design_event_study_questions.csv"), index=False)
    placebo.to_csv(os.path.join(PROCESSED_DIR, "large_design_placebo_break_tests.csv"), index=False)
    permutations.to_csv(os.path.join(PROCESSED_DIR, "large_design_permutation_tests.csv"), index=False)
    wild_bootstrap.to_csv(os.path.join(PROCESSED_DIR, "large_design_wild_cluster_bootstrap.csv"), index=False)


def export_results(results, event_tests, exposure_diagnostics, attrition, placebo, permutations, wild_bootstrap):
    serializable = {key: payload["summary"] for key, payload in results.items()}
    serializable["event_study_tests"] = event_tests
    serializable["exposure_diagnostics"] = exposure_diagnostics
    serializable["attrition"] = attrition
    serializable["placebo_break_tests"] = placebo.to_dict(orient="records")
    serializable["permutation_tests"] = permutations.to_dict(orient="records")
    serializable["wild_cluster_bootstrap"] = wild_bootstrap.to_dict(orient="records")
    with open(os.path.join(PROCESSED_DIR, "large_design_regression_results.json"), "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
    return serializable


def plot_tercile_trends(monthly):
    grouped_views = (
        monthly.groupby(["month_id", "exposure_group"])
        .apply(lambda frame: np.average(frame["mean_log_views"], weights=frame["n_questions"]))
        .rename("weighted_mean_log_views")
        .reset_index()
    )
    pivot_views = grouped_views.pivot(index="month_id", columns="exposure_group", values="weighted_mean_log_views").sort_index()
    grouped_questions = monthly.groupby(["month_id", "exposure_group"], as_index=False)["n_questions"].sum()
    pivot_questions = grouped_questions.pivot(index="month_id", columns="exposure_group", values="n_questions").sort_index()
    palette = {"Low Exposure": "#4c78a8", "Mid Exposure": "#72b7b2", "High Exposure": "#e45756"}

    plt.figure(figsize=(12, 5))
    for label in ["Low Exposure", "Mid Exposure", "High Exposure"]:
        plt.plot(pivot_views.index, pivot_views[label], marker="o", linewidth=2, label=label, color=palette[label])
    plt.axvline(SHOCK_MONTH, color="black", linestyle="--", linewidth=1)
    plt.title("Later Cumulative Visibility by Pre-Shock Exposure Tercile")
    plt.ylabel("Weighted mean log(1 + views)")
    plt.xlabel("Question creation month")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "large_design_views_trends.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(12, 5))
    for label in ["Low Exposure", "Mid Exposure", "High Exposure"]:
        plt.plot(pivot_questions.index, pivot_questions[label], marker="o", linewidth=2, label=label, color=palette[label])
    plt.axvline(SHOCK_MONTH, color="black", linestyle="--", linewidth=1)
    plt.title("Question Volume by Pre-Shock Exposure Tercile")
    plt.ylabel("Questions created in month")
    plt.xlabel("Question creation month")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "large_design_question_volume_trends.png"), dpi=220)
    plt.close()


def plot_event_study(event_df, filename, title, ylabel):
    frame = event_df.copy()
    x = np.arange(len(frame))
    colors = frame["post_chatgpt"].map({0: "#4c78a8", 1: "#e45756"})
    plt.figure(figsize=(12, 5))
    plt.axhline(0, color="black", linewidth=1)
    plt.errorbar(x, frame["coef"], yerr=1.96 * frame["se"], fmt="o", color="#4c4c4c", ecolor="#9a9a9a", capsize=3)
    plt.scatter(x, frame["coef"], c=colors, s=48, zorder=3)
    ref_index = frame.index[frame["month"] == REFERENCE_MONTH][0]
    plt.axvline(ref_index, color="black", linestyle="--", linewidth=1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Question creation month (reference: 2022-11)")
    plt.xticks(x, frame["month"], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=220)
    plt.close()


def plot_exposure_rank(exposure):
    frame = exposure.sort_values("exposure_index", ascending=False).reset_index(drop=True)
    colors = frame["exposure_group"].map({"High Exposure": "#e45756", "Mid Exposure": "#72b7b2", "Low Exposure": "#4c78a8"})
    plt.figure(figsize=(10, 6))
    plt.barh(frame["tag"], frame["exposure_index"], color=colors)
    plt.gca().invert_yaxis()
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Pre-Shock Exposure Index by Tag")
    plt.xlabel("Standardized exposure index")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "large_design_exposure_rank.png"), dpi=220)
    plt.close()


def save_summary_tables(monthly, policy, exposure, results, event_tests, attrition, placebo, permutations, wild_bootstrap):
    baseline_rows = []
    for key in ["baseline_log_questions", "baseline_mean_log_views", "baseline_accepted_rate", "baseline_mean_log_answers", "question_log_views", "question_accepted"]:
        payload = results[key]["summary"]
        baseline_rows.append(
            {"model": key, "coef": round(payload["coef"], 4), "se": round(payload["se"], 4), "pval": round(payload["pval"], 4), "nobs": int(payload["nobs"])}
        )
    segmented_rows = []
    for key in ["segmented_log_questions", "segmented_mean_log_views"]:
        for term_name, payload in results[key]["summary"].items():
            segmented_rows.append(
                {"model": key, "term": term_name, "coef": round(payload["coef"], 4), "se": round(payload["se"], 4), "pval": round(payload["pval"], 4), "nobs": int(payload["nobs"])}
            )
    attrition_table = pd.DataFrame([attrition])
    exposure_table = exposure.sort_values("exposure_index", ascending=False)[["tag", "exposure_index", "exposure_group", "pre_questions", "routine_keyword_share"]]
    descriptive = (
        monthly.groupby(["exposure_group", "post_chatgpt"], as_index=False)
        .agg(questions=("n_questions", "sum"), accepted_rate=("accepted_rate", "mean"), mean_log_views=("mean_log_views", "mean"), mean_log_answers=("mean_log_answers", "mean"))
    )
    descriptive["period"] = descriptive["post_chatgpt"].map({0: "Pre-ChatGPT", 1: "Post-ChatGPT"})
    event_table = pd.DataFrame(
        [{"outcome": outcome, "pre_joint_pval": round(test["pre_joint_pval"], 4), "post_joint_pval": round(test["post_joint_pval"], 4)} for outcome, test in event_tests.items()]
    )

    with open(os.path.join(PROCESSED_DIR, "large_design_summary_tables.md"), "w", encoding="utf-8") as handle:
        handle.write("# Large-Design Summary Tables\n\n")
        handle.write("## Sample Attrition\n\n")
        handle.write(attrition_table.to_markdown(index=False))
        handle.write("\n\n## Exposure Ranking\n\n")
        handle.write(exposure_table.to_markdown(index=False))
        handle.write("\n\n## Descriptive Summary\n\n")
        handle.write(descriptive.to_markdown(index=False))
        if not policy.empty:
            policy_summary = (
                policy.groupby(["post_policy"], as_index=False)
                .agg(questions=("question_id", "count"), accepted_rate=("accepted", "mean"), mean_log_views=("log_views", "mean"))
            )
            policy_summary["period"] = policy_summary["post_policy"].map({0: "Dec 1-4", 1: "Dec 5-7"})
            handle.write("\n\n## Policy-Window Summary\n\n")
            handle.write(policy_summary.to_markdown(index=False))
        handle.write("\n\n## Baseline Models\n\n")
        handle.write(pd.DataFrame(baseline_rows).to_markdown(index=False))
        handle.write("\n\n## Segmented Models\n\n")
        handle.write(pd.DataFrame(segmented_rows).to_markdown(index=False))
        handle.write("\n\n## Event Study Joint Tests\n\n")
        handle.write(event_table.to_markdown(index=False))
        handle.write("\n\n## Placebo Break Tests\n\n")
        handle.write(placebo.to_markdown(index=False))
        handle.write("\n\n## Permutation Tests\n\n")
        handle.write(permutations.to_markdown(index=False))
        handle.write("\n\n## Wild-Cluster Bootstrap\n\n")
        handle.write(wild_bootstrap.to_markdown(index=False))


def save_paper_stats(questions, raw_with_attrition, monthly, exposure, results, event_tests, exposure_diagnostics, attrition, placebo, permutations, wild_bootstrap):
    stats = {
        "questions_raw_total": int(len(raw_with_attrition)),
        "questions_analysis_total": int(len(questions)),
        "unique_questions_analysis": int(questions["question_id"].nunique()),
        "unique_tags": int(questions["tag"].nunique()),
        "attrition": attrition,
        "baseline_log_questions": results["baseline_log_questions"]["summary"],
        "baseline_mean_log_views": results["baseline_mean_log_views"]["summary"],
        "segmented_log_questions": results["segmented_log_questions"]["summary"],
        "segmented_mean_log_views": results["segmented_mean_log_views"]["summary"],
        "question_log_views": results["question_log_views"]["summary"],
        "policy_question_accepted": results.get("policy_question_accepted", {}).get("summary"),
        "event_study_tests": event_tests,
        "exposure_diagnostics": exposure_diagnostics,
        "top_exposure_tags": exposure.sort_values("exposure_index", ascending=False)["tag"].head(5).tolist(),
        "bottom_exposure_tags": exposure.sort_values("exposure_index", ascending=True)["tag"].head(5).tolist(),
        "placebo_break_tests": placebo.to_dict(orient="records"),
        "permutation_tests": permutations.to_dict(orient="records"),
        "wild_cluster_bootstrap": wild_bootstrap.to_dict(orient="records"),
    }
    with open(os.path.join(PROCESSED_DIR, "large_design_paper_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def main():
    ensure_dirs()
    raw = load_raw_data()
    tag_metadata = load_tag_metadata()
    questions, raw_with_attrition = prepare_question_data(raw, tag_metadata["selected_tags"])
    exposure, exposure_diagnostics = build_exposure_index(questions)
    questions = attach_exposure(questions, exposure)
    monthly, policy = build_panels(questions)
    attrition = {
        "raw_rows": int(len(raw_with_attrition)),
        "analysis_rows": int(len(questions)),
        "raw_unique_questions": int(raw_with_attrition["question_id"].nunique()),
        "analysis_unique_questions": int(questions["question_id"].nunique()),
        "dropped_multi_selected_rows": int((raw_with_attrition["selected_tag_overlap"] > 1).sum()),
        "dropped_non_focal_rows": int((raw_with_attrition["keep_single_focal"] == 0).sum()),
    }
    results = fit_main_models(monthly, policy, questions)
    event_views, event_view_tests = build_event_study(monthly, "mean_log_views", weight_col="n_questions")
    event_questions, event_question_tests = build_event_study(monthly, "log_questions")
    placebo = placebo_break_tests(monthly)
    permutations = permutation_tests(monthly)
    wild_bootstrap = wild_cluster_bootstrap(monthly)
    event_tests = {"mean_log_views": event_view_tests, "log_questions": event_question_tests}

    save_processed_data(questions, raw_with_attrition, monthly, policy, exposure, event_views, event_questions, placebo, permutations, wild_bootstrap)
    export_results(results, event_tests, exposure_diagnostics, attrition, placebo, permutations, wild_bootstrap)
    plot_tercile_trends(monthly)
    plot_event_study(event_views, "large_design_event_study_views.png", "Event Study: Exposure-Sorted Later Visibility", "Differential slope relative to 2022-11")
    plot_event_study(event_questions, "large_design_event_study_questions.png", "Event Study: Exposure-Sorted Question Creation", "Differential slope relative to 2022-11")
    plot_exposure_rank(exposure)
    save_summary_tables(monthly, policy, exposure, results, event_tests, attrition, placebo, permutations, wild_bootstrap)
    stats = save_paper_stats(questions, raw_with_attrition, monthly, exposure, results, event_tests, exposure_diagnostics, attrition, placebo, permutations, wild_bootstrap)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
