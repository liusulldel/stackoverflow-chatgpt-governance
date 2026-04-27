"""
Build main-text figure assets for the P1 JMIS special-issue package.

Outputs:
  - figures/p1_jmis_figure1_governance_map.png
  - figures/p1_jmis_figure2_complexity_asymmetry.png
  - figures/p1_jmis_figure3_family_heterogeneity.png
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
PROC_DIR = ROOT / "processed"


def _setup_plot():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def build_figure1_governance_map():
    fig = plt.figure(figsize=(10, 4.8), dpi=180)
    ax = fig.add_subplot(111)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.35", facecolor="#f2f4f8", edgecolor="#3a3a3a")
    header_style = dict(color="#1b1b1b", fontsize=11, fontweight="bold")
    body_style = dict(color="#1b1b1b", fontsize=9)

    ax.text(0.05, 0.82, "Private GenAI substitution", bbox=box_style, **header_style)
    ax.text(0.05, 0.74, "Routine work moves off-platform", **body_style)

    ax.text(0.32, 0.82, "Residual public queue change", bbox=box_style, **header_style)
    ax.text(0.32, 0.74, "More context-heavy, less templatable", **body_style)

    ax.text(0.60, 0.82, "Entrant-type re-sorting", bbox=box_style, **header_style)
    ax.text(0.60, 0.74, "Brand-new up; low-tenure down", **body_style)

    ax.text(
        0.05,
        0.38,
        "Governance split",
        bbox=box_style,
        **header_style,
    )
    ax.text(
        0.05,
        0.30,
        "Response scales faster than settlement",
        **body_style,
    )

    ax.text(0.60, 0.38, "Bounded claim discipline", bbox=box_style, **header_style)
    ax.text(
        0.60,
        0.30,
        "Visible by the ChatGPT period; few-cluster design",
        **body_style,
    )

    arrow_kwargs = dict(arrowstyle="->", color="#1b1b1b", linewidth=1.2)
    ax.annotate("", xy=(0.30, 0.80), xytext=(0.22, 0.80), arrowprops=arrow_kwargs)
    ax.annotate("", xy=(0.58, 0.80), xytext=(0.50, 0.80), arrowprops=arrow_kwargs)
    ax.annotate("", xy=(0.28, 0.52), xytext=(0.18, 0.72), arrowprops=arrow_kwargs)
    ax.annotate("", xy=(0.55, 0.52), xytext=(0.70, 0.72), arrowprops=arrow_kwargs)

    ax.text(
        0.05,
        0.08,
        "Figure 1. Governance architecture: residual queue change and entrant re-sorting\n"
        "make some provisional response surfaces easier to scale than settlement signals.",
        fontsize=9,
        color="#333333",
    )

    out_path = FIG_DIR / "p1_jmis_figure1_governance_map.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_figure2_complexity_asymmetry():
    df = pd.read_csv(PROC_DIR / "p1_brand_new_complexity_gaps.csv")

    order = ["low", "mid", "high"]
    response = (
        df[df["outcome"] == "first_answer_1d"]
        .set_index("complexity_tercile")
        .reindex(order)
    )
    settlement = (
        df[df["outcome"] == "accepted_30d"]
        .set_index("complexity_tercile")
        .reindex(order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=180, sharey=False)

    axes[0].bar(
        order,
        response["coef_brand_new_vs_established"],
        color="#4c78a8",
    )
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Response: first_answer_1d gap")
    axes[0].set_ylabel("Brand-new minus established")
    axes[0].set_xlabel("Complexity tercile")

    axes[1].bar(
        order,
        settlement["coef_brand_new_vs_established"],
        color="#f58518",
    )
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Settlement: accepted_30d gap")
    axes[1].set_xlabel("Complexity tercile")

    fig.suptitle(
        "Figure 2. Complexity-tercile asymmetry (brand-new vs established)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    out_path = FIG_DIR / "p1_jmis_figure2_complexity_asymmetry.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_figure3_family_heterogeneity():
    df = pd.read_csv(PROC_DIR / "p1_family_consequence_results.csv")

    df = df[(df["family"] == "binary") & (df["term"] == "high_post")]
    df = df[df["outcome"].isin(["first_answer_1d_rate_closure", "accepted_vote_30d_rate"])]

    fam_order = ["data_scripting", "application_framework", "systems_infra"]
    df["tag_family"] = pd.Categorical(df["tag_family"], fam_order, ordered=True)

    pivot = df.pivot_table(
        index="tag_family",
        columns="outcome",
        values="coef",
        aggfunc="first",
    ).reindex(fam_order)

    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=180)
    x = range(len(fam_order))
    width = 0.36

    ax.bar(
        [i - width / 2 for i in x],
        pivot["first_answer_1d_rate_closure"],
        width=width,
        color="#4c78a8",
        label="Response (first_answer_1d)",
    )
    ax.bar(
        [i + width / 2 for i in x],
        pivot["accepted_vote_30d_rate"],
        width=width,
        color="#f58518",
        label="Settlement (accepted_vote_30d)",
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        ["data_scripting", "application_framework", "systems_infra"],
        rotation=0,
    )
    ax.set_ylabel("High_post coefficient")
    ax.set_title("Figure 3. Family heterogeneity in response vs settlement")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    out_path = FIG_DIR / "p1_jmis_figure3_family_heterogeneity.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    _setup_plot()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    build_figure1_governance_map()
    build_figure2_complexity_asymmetry()
    build_figure3_family_heterogeneity()


if __name__ == "__main__":
    main()
