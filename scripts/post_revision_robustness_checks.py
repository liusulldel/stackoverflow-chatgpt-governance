import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(r"D:\AI alignment\projects\stackoverflow_chatgpt_governance")
PROCESSED = ROOT / "processed"
MONTHLY_PATH = PROCESSED / "mainstream_design_tag_month_panel.csv"

FORMULA = "accepted_rate ~ high_post + C(tag):time_index + C(tag) + C(month_id)"
PLACEBO_FORMULA = "accepted_rate ~ high_tag:placebo_post + C(tag):time_index + C(tag) + C(month_id)"


def fit_weighted(formula: str, data: pd.DataFrame):
    return smf.wls(formula, data=data, weights=data["n_questions"]).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["tag"]},
    )


def leave_two_out(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tags = sorted(monthly["tag"].unique())
    for tag_a, tag_b in combinations(tags, 2):
        subset = monthly.loc[~monthly["tag"].isin([tag_a, tag_b])].copy()
        model = fit_weighted(FORMULA, subset)
        rows.append(
            {
                "dropped_tag_a": tag_a,
                "dropped_tag_b": tag_b,
                "coef": float(model.params.get("high_post", np.nan)),
                "se": float(model.bse.get("high_post", np.nan)),
                "pval": float(model.pvalues.get("high_post", np.nan)),
                "n_clusters": int(subset["tag"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def placebo_grid(monthly: pd.DataFrame) -> pd.DataFrame:
    month_order = sorted(monthly["month_id"].unique())
    rows = []
    for placebo_month in month_order[2:-2]:
        frame = monthly.copy()
        frame["placebo_post"] = (frame["month_id"] >= placebo_month).astype(int)
        model = fit_weighted(PLACEBO_FORMULA, frame)
        rows.append(
            {
                "placebo_month": placebo_month,
                "coef": float(model.params.get("high_tag:placebo_post", np.nan)),
                "se": float(model.bse.get("high_tag:placebo_post", np.nan)),
                "pval": float(model.pvalues.get("high_tag:placebo_post", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def summarize(leave_two: pd.DataFrame, placebo: pd.DataFrame) -> dict:
    pre_placebo = placebo.loc[placebo["placebo_month"] < "2022-12"].copy()
    return {
        "leave_two_out": {
            "n_models": int(len(leave_two)),
            "negative_share": float((leave_two["coef"] < 0).mean()),
            "significant_005_share": float((leave_two["pval"] < 0.05).mean()),
            "coef_min": float(leave_two["coef"].min()),
            "coef_median": float(leave_two["coef"].median()),
            "coef_max": float(leave_two["coef"].max()),
            "worst_case_row": leave_two.sort_values("coef", ascending=False).iloc[0].to_dict(),
            "most_negative_row": leave_two.sort_values("coef", ascending=True).iloc[0].to_dict(),
        },
        "placebo_grid": {
            "n_months": int(len(placebo)),
            "pre_shock_n": int(len(pre_placebo)),
            "pre_shock_significant_months": pre_placebo.loc[pre_placebo["pval"] < 0.05, "placebo_month"].tolist(),
            "pre_shock_min_pval": float(pre_placebo["pval"].min()) if not pre_placebo.empty else None,
            "closest_pre_shock_row": pre_placebo.sort_values("pval").iloc[0].to_dict() if not pre_placebo.empty else None,
        },
    }


def write_markdown(summary: dict, leave_two: pd.DataFrame, placebo: pd.DataFrame) -> str:
    worst = summary["leave_two_out"]["worst_case_row"]
    most_negative = summary["leave_two_out"]["most_negative_row"]
    pre_sig = summary["placebo_grid"]["pre_shock_significant_months"]
    lines = [
        "# Post-Revision Robustness Checks",
        "",
        "## Leave-Two-Tags-Out Jackknife",
        "",
        f"- Models estimated: `{summary['leave_two_out']['n_models']}`",
        f"- Negative coefficient share: `{summary['leave_two_out']['negative_share']:.3f}`",
        f"- p < 0.05 share: `{summary['leave_two_out']['significant_005_share']:.3f}`",
        f"- Coefficient range: `{summary['leave_two_out']['coef_min']:.4f}` to `{summary['leave_two_out']['coef_max']:.4f}`",
        f"- Least negative case: drop `{worst['dropped_tag_a']}` and `{worst['dropped_tag_b']}` -> coef `{worst['coef']:.4f}`, p `{worst['pval']:.4f}`",
        f"- Most negative case: drop `{most_negative['dropped_tag_a']}` and `{most_negative['dropped_tag_b']}` -> coef `{most_negative['coef']:.4f}`, p `{most_negative['pval']:.4f}`",
        "",
        "## Dense Placebo Grid",
        "",
        f"- Placebo months evaluated: `{summary['placebo_grid']['n_months']}`",
        f"- Pre-shock placebo months evaluated: `{summary['placebo_grid']['pre_shock_n']}`",
        f"- Significant pre-shock placebos (p < 0.05): `{', '.join(pre_sig) if pre_sig else 'none'}`",
        f"- Lowest pre-shock placebo p-value: `{summary['placebo_grid']['pre_shock_min_pval']:.4f}`",
        "",
        "## Head Rows",
        "",
        leave_two.head(10).to_markdown(index=False),
        "",
        placebo.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def main():
    monthly = pd.read_csv(MONTHLY_PATH)
    leave_two = leave_two_out(monthly)
    placebo = placebo_grid(monthly)
    summary = summarize(leave_two, placebo)

    leave_two_path = PROCESSED / "post_revision_leave_two_out.csv"
    placebo_path = PROCESSED / "post_revision_placebo_grid.csv"
    summary_json_path = PROCESSED / "post_revision_robustness_summary.json"
    summary_md_path = PROCESSED / "post_revision_robustness_summary.md"

    leave_two.to_csv(leave_two_path, index=False)
    placebo.to_csv(placebo_path, index=False)
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary_md_path.write_text(write_markdown(summary, leave_two, placebo), encoding="utf-8")

    print(summary_json_path)
    print(summary_md_path)


if __name__ == "__main__":
    main()
