from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = BASE_DIR / "figures"
OUTFILE = FIGURES_DIR / "staged_public_resolution_ladder.png"


def add_box(ax, xy, width, height, text, facecolor):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#1f3d5a",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color="#17324d",
        wrap=True,
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.2,
            color="#1f3d5a",
        )
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.34, 0.83), 0.32, 0.10, "Private AI outside option", "#e7f0fa")
    add_box(ax, (0.28, 0.65), 0.44, 0.11, "Residual public queue becomes more selective", "#eef5db")

    add_box(ax, (0.05, 0.34), 0.26, 0.18, "Stage 1\nRapid-response coordination\n• first_answer_1d\n• any_answer_7d", "#fdf1d6")
    add_box(ax, (0.37, 0.34), 0.26, 0.18, "Stage 2\nAnswer endorsement\n• any_positively_scored_answer_7d\n• first_positive_answer_latency", "#fde2e4")
    add_box(ax, (0.69, 0.34), 0.26, 0.18, "Stage 3\nFormalized closure\n• accepted_cond_any_answer_30d\n• accepted_vote_30d", "#e5ecf6")

    arrow(ax, (0.50, 0.83), (0.50, 0.76))
    arrow(ax, (0.50, 0.65), (0.18, 0.52))
    arrow(ax, (0.50, 0.65), (0.50, 0.52))
    arrow(ax, (0.50, 0.65), (0.82, 0.52))

    ax.text(
        0.5,
        0.14,
        "Staged public resolution treats platform-native closure as a coordination ladder rather than a single interchangeable metric.",
        ha="center",
        va="center",
        fontsize=11,
        color="#17324d",
        wrap=True,
    )

    fig.tight_layout()
    fig.savefig(OUTFILE, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(str(OUTFILE))


if __name__ == "__main__":
    main()
