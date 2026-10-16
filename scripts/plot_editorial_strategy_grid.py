"""Create an editorial 3x5 strategy graphic from saved results; never runs the model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PAPER = "#F5F0E8"
PANEL = "#FCF9F3"
INK = "#242421"
MUTED = "#7C7B73"
GOLD = "#A96808"
COLORS = {"win": "#D3DECF", "front": "#F1E6C8", "middle": "#E2DEDA", "back": "#ECD1CB"}
SCENARIOS = ("none", "1", "5", "20", "35")


def load_results(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    results = {}
    for row in rows:
        key = row["strategy"], row["sc_lap"]
        if key in results:
            raise ValueError(f"Duplicate result: {key}")
        results[key] = {
            "position": int(row["position"]),
            "gap": float(row["gap_to_winner_s"]),
            "time": float(row["race_time_s"]),
        }
    expected = {(strategy, sc) for strategy in "ABC" for sc in SCENARIOS}
    if set(results) != expected:
        raise ValueError("Expected exactly three strategies and five Safety Car scenarios.")
    for result in results.values():
        if not 1 <= result["position"] <= 22 or result["gap"] < 0:
            raise ValueError(f"Invalid result: {result}")
    return results


def band(position: int) -> str:
    return "win" if position == 1 else "front" if position <= 6 else "middle" if position <= 8 else "back"


def make_figure(results: dict, out: Path) -> list[Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(14.56, 10.40), facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def text(x, y, value, **kwargs):
        return ax.text(x, y, value, transform=ax.transAxes, **kwargs)

    text(0.05, 0.965, "MONZA 2026  ·  ANTONELLI FROM THE BACK  ·  THREE PLANS, FIVE RACES",
         fontsize=10.8, fontweight="bold", color=GOLD, va="center")
    text(0.96, 0.965, "MODEL RESULTS", fontsize=10.2, fontweight="bold", color=GOLD,
         ha="right", va="center")
    advantage = results["A", "none"]["time"] - results["B", "none"]["time"]
    headline = f"A quiet-race tie on position. The hard start is {advantage:.1f} seconds quicker."
    text(0.05, 0.918, headline, fontsize=21.0, fontfamily="DejaVu Serif",
         fontweight="bold", va="center")
    subtitle = (
        "Simulated finishing positions from a pit-lane start: 53 laps, 22 cars; 22s pit loss, 11s under the Safety Car. Hard degradation: 0.015s/lap.\n"
        "A and B both finish P3 without a Safety Car. Early or mid-race neutralisation hurts the lap-one stop; the conventional plan finishes P8 throughout."
    )
    text(0.05, 0.880, subtitle, fontsize=10.6, va="top", linespacing=1.35)

    # A deliberately direct counterpart to the reference: same plan-by-world reading order.
    ax.add_patch(Rectangle((0.259, 0.138), 0.701, 0.661, facecolor=PANEL, edgecolor="none"))
    x0, step, width, height = 0.265, 0.141, 0.129, 0.166
    bottoms = (0.605, 0.385, 0.165)
    labels = ("Box lap 1\nhard to the flag", "Start hard, go long\nmedium at lap 36",
              "Conventional\nmedium, stop lap 20, hard")
    headers = ("no safety car", "SC lap 1", "SC lap 5", "SC lap 20", "SC lap 35")
    for col, header in enumerate(headers):
        cx = x0 + col * step + width / 2
        text(cx, 0.817, header, fontsize=11.0, ha="center", va="center")
        ax.plot([cx, cx], [0.799, 0.803], color=INK, lw=0.7)
    for row, strategy in enumerate("ABC"):
        bottom = bottoms[row]
        text(0.243, bottom + height / 2, labels[row], fontsize=11.4,
             fontweight="bold" if row == 0 else "normal", ha="right", va="center", linespacing=1.05)
        for col, sc in enumerate(SCENARIOS):
            item = results[strategy, sc]
            left = x0 + col * step
            ax.add_patch(FancyBboxPatch((left, bottom), width, height,
                         boxstyle="round,pad=0,rounding_size=0.0025", linewidth=0,
                         facecolor=COLORS[band(item["position"])]))
            text(left + width / 2, bottom + 0.104, f"P{item['position']}",
                 fontsize=23.0, fontweight="bold", ha="center", va="center")
            text(left + width / 2, bottom + 0.031, f"+{item['gap']:.1f}s to the winner",
                 fontsize=9.3, ha="center", va="center")

    # Spell out the palette instead of implying unexplained strategy rankings.
    text(0.05, 0.097, "FINISHING-POSITION BANDS", fontsize=8.1, fontweight="bold", color=MUTED, va="center")
    for i, (label, color) in enumerate((("P1", "win"), ("P2–P6", "front"), ("P7–P8", "middle"), ("P9+", "back"))):
        lx = 0.251 + i * 0.104
        ax.add_patch(Rectangle((lx, 0.091), 0.012, 0.012, facecolor=COLORS[color], edgecolor="none"))
        text(lx + 0.018, 0.097, label, fontsize=8.7, color=MUTED, va="center")
    text(0.96, 0.097, "Independent reconstruction · Not a validated forecast", fontsize=8.7,
         color=MUTED, ha="right", va="center")
    footer = (
        "Baseline: hard 0.015s/lap, medium 0.05, soft 0.10; medium and soft are 0.20s and 0.35s quicker than a new hard. Fuel effect: 0.07s/lap.\n"
        "Dirty air costs 0.25s/lap within 2s; passing requires a 0.35s/lap advantage and time crossing. Safety Car: one shared, fixed-duration window.\n"
        "Source: saved project baseline results (antonelli_15_scenarios.csv). Gaps are to each scenario's own winner, rounded to 0.1s. Layout inspired by Ruth Buscombe's briefing."
    )
    text(0.05, 0.065, footer, fontsize=8.2, color=MUTED, fontstyle="italic", va="top", linespacing=1.35)

    out.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg", "pdf"):
        path = out.with_suffix(f".{extension}")
        fig.savefig(path, dpi=200, facecolor=PAPER)
        paths.append(path)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/tables/antonelli_15_scenarios.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures/antonelli_strategy_grid_editorial")
    args = parser.parse_args()
    for output in make_figure(load_results(args.input), args.output):
        print(output)
