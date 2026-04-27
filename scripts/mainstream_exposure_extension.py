import json
import os
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

QUESTIONS_FILE = os.path.join(PROCESSED_DIR, "mainstream_design_question_level_enriched.csv")
PRIMARY_PANEL_FILE = os.path.join(PROCESSED_DIR, "mainstream_design_tag_month_panel.csv")
FRACTIONAL_PANEL_FILE = os.path.join(PROCESSED_DIR, "mainstream_design_fractional_tag_month_panel.csv")

PRE_EXPOSURE_END = "2022-11"
WILD_BOOTSTRAP_REPS = 999
WILD_BOOTSTRAP_SEED = 42
TITLE_SAMPLE_PER_TAG = 4000

ROUTINE_PATTERN = re.compile(
    r"\b(?:how|error|exception|convert|parse|replace|split|sort|join|merge|regex|"
    r"formula|string|array|list|column|row|csv|format|plot|numpy|data)\b",
    flags=re.IGNORECASE,
)


def load_inputs():
    primary = pd.read_csv(PRIMARY_PANEL_FILE)
    fractional = pd.read_csv(FRACTIONAL_PANEL_FILE)
    return primary, fractional


def build_exposure_index():
    tag_samples = {}
    tag_counts = {}
    tag_routine = {}
    tag_group = {}

    for chunk in pd.read_csv(
        QUESTIONS_FILE,
        usecols=["question_id", "title", "tag", "month_id", "group"],
        chunksize=50000,
    ):
        pre = chunk.loc[chunk["month_id"] <= PRE_EXPOSURE_END].copy()
        if pre.empty:
            continue
        pre["text_blob"] = pre["title"].fillna("").astype(str).str.strip()
        pre["routine_keyword_hit"] = pre["text_blob"].str.contains(ROUTINE_PATTERN, regex=True, na=False).astype(int)
        for tag, data in pre.groupby("tag"):
            tag_counts[tag] = tag_counts.get(tag, 0) + int(len(data))
            tag_routine[tag] = tag_routine.get(tag, 0) + int(data["routine_keyword_hit"].sum())
            if tag not in tag_group and not data["group"].empty:
                tag_group[tag] = data["group"].iloc[0]
            if tag not in tag_samples:
                tag_samples[tag] = []
            remaining = TITLE_SAMPLE_PER_TAG - len(tag_samples[tag])
            if remaining > 0:
                tag_samples[tag].extend(data["text_blob"].head(remaining).tolist())

    tag_docs = pd.DataFrame(
        [
            {
                "tag": tag,
                "document": " ".join(tag_samples.get(tag, [])),
                "pre_questions": tag_counts.get(tag, 0),
                "routine_keyword_share": (
                    tag_routine.get(tag, 0) / tag_counts.get(tag, 1)
                ),
                "current_group": tag_group.get(tag, ""),
            }
            for tag in sorted(tag_counts)
        ]
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
    exposure_index = (exposure_raw - exposure_raw.mean()) / exposure_raw.std(ddof=0)
    tag_docs["exposure_raw"] = exposure_raw
    tag_docs["exposure_index"] = exposure_index
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


def attach_exposure(panel, exposure):
    merged = panel.merge(
        exposure[["tag", "exposure_index", "exposure_rank", "exposure_tercile", "pre_questions", "routine_keyword_share"]],
        on="tag",
        how="left",
    )
    merged["exposure_post"] = merged["exposure_index"] * merged["post_chatgpt"]
    return merged


def fit_weighted(formula, data):
    return smf.wls(formula, data=data, weights=data["n_questions"]).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["tag"]},
    )


def fit_plain(formula, data):
    return smf.wls(formula, data=data, weights=data["n_questions"]).fit()


def extract_term(model, term):
    return {
        "coef": float(model.params.get(term, np.nan)),
        "se": float(model.bse.get(term, np.nan)),
        "pval": float(model.pvalues.get(term, np.nan)),
        "nobs": float(model.nobs),
        "r2": float(getattr(model, "rsquared", np.nan)),
    }


def fit_models(primary, fractional):
    specs = {
        "accepted_rate_continuous_primary": {
            "data": primary,
            "formula": "accepted_rate ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
            "term": "exposure_post",
        },
        "mean_log_answers_continuous_primary": {
            "data": primary,
            "formula": "mean_log_answers ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
            "term": "exposure_post",
        },
        "accepted_rate_continuous_fractional": {
            "data": fractional,
            "formula": "accepted_rate ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
            "term": "exposure_post",
        },
        "mean_log_answers_continuous_fractional": {
            "data": fractional,
            "formula": "mean_log_answers ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
            "term": "exposure_post",
        },
    }
    results = {}
    for label, spec in specs.items():
        model = fit_weighted(spec["formula"], spec["data"])
        results[label] = {"summary": extract_term(model, spec["term"]), "fit": model, "term": spec["term"], "formula": spec["formula"]}
    return results


def wild_cluster_bootstrap(specs):
    rng = np.random.default_rng(WILD_BOOTSTRAP_SEED)
    rows = []
    for spec in specs:
        data = spec["data"]
        term = spec["term"]
        full = fit_weighted(spec["full_formula"], data)
        restricted = fit_plain(spec["restricted_formula"], data)
        observed_t = float(full.params[term] / full.bse[term])
        clusters = data["tag"].astype(str).to_numpy()
        unique_clusters = sorted(set(clusters))
        t_stars = []
        for _ in range(WILD_BOOTSTRAP_REPS):
            weights = {cluster: rng.choice([-1.0, 1.0]) for cluster in unique_clusters}
            boot = data.copy()
            boot[spec["outcome"]] = restricted.fittedvalues.to_numpy() + restricted.resid.to_numpy() * np.array([weights[c] for c in clusters])
            try:
                boot_model = fit_weighted(spec["full_formula"], boot)
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


def exposure_tercile_summary(primary):
    summary = (
        primary.groupby(["exposure_tercile", "post_chatgpt"], observed=False)
        .apply(
            lambda data: pd.Series(
                {
                    "weighted_accepted_rate": float(np.average(data["accepted_rate"], weights=data["n_questions"])),
                    "weighted_mean_log_answers": float(np.average(data["mean_log_answers"], weights=data["n_questions"])),
                }
            )
        )
        .reset_index()
    )
    summary["period"] = np.where(summary["post_chatgpt"] == 1, "post", "pre")
    return summary


def save_outputs(exposure, diagnostics, results, wild_bootstrap, terciles):
    exposure.to_csv(os.path.join(PROCESSED_DIR, "mainstream_exposure_extension_tag_index.csv"), index=False)
    terciles.to_csv(os.path.join(PROCESSED_DIR, "mainstream_exposure_extension_terciles.csv"), index=False)
    with open(os.path.join(PROCESSED_DIR, "mainstream_exposure_extension_diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    payload = {
        "results": {key: value["summary"] for key, value in results.items()},
        "wild_cluster_bootstrap": wild_bootstrap.to_dict(orient="records"),
        "diagnostics": diagnostics,
    }
    with open(os.path.join(PROCESSED_DIR, "mainstream_exposure_extension_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    rows = []
    for key, value in results.items():
        summary = value["summary"]
        rows.append(
            {
                "model": key,
                "coef": round(summary["coef"], 4),
                "se": round(summary["se"], 4),
                "pval": round(summary["pval"], 4),
                "nobs": int(summary["nobs"]),
            }
        )
    with open(os.path.join(PROCESSED_DIR, "mainstream_exposure_extension_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("# Mainstream Exposure Extension Summary\n\n")
        handle.write(pd.DataFrame(rows).to_markdown(index=False))
        handle.write("\n\n## Wild Cluster Bootstrap\n\n")
        handle.write(wild_bootstrap.to_markdown(index=False))
        handle.write("\n\n## Exposure Tercile Summary\n\n")
        handle.write(terciles.to_markdown(index=False))


def main():
    primary_panel, fractional_panel = load_inputs()
    exposure, diagnostics = build_exposure_index()
    primary_panel = attach_exposure(primary_panel, exposure)
    fractional_panel = attach_exposure(fractional_panel, exposure)
    results = fit_models(primary_panel, fractional_panel)
    wild = wild_cluster_bootstrap(
        [
            {
                "specification": "accepted_rate_continuous_primary",
                "data": primary_panel,
                "full_formula": "accepted_rate ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
                "restricted_formula": "accepted_rate ~ C(tag):time_index + C(tag) + C(month_id)",
                "term": "exposure_post",
                "outcome": "accepted_rate",
            },
            {
                "specification": "accepted_rate_continuous_fractional",
                "data": fractional_panel,
                "full_formula": "accepted_rate ~ exposure_post + C(tag):time_index + C(tag) + C(month_id)",
                "restricted_formula": "accepted_rate ~ C(tag):time_index + C(tag) + C(month_id)",
                "term": "exposure_post",
                "outcome": "accepted_rate",
            },
        ]
    )
    terciles = exposure_tercile_summary(primary_panel)
    save_outputs(exposure, diagnostics, results, wild, terciles)
    print(json.dumps({"results": {k: v["summary"] for k, v in results.items()}, "wild_cluster_bootstrap": wild.to_dict(orient="records"), "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
