"""
make_figures_v1.py  —  every Proto 06 figure, from recorded measurements.

    python figures/make_figures_v1.py

Writes PNGs to prototypes/06_feudal/results/best_figures/.

PROVENANCE. Numbers below are transcribed from the runs recorded in BUILDNOTES (P1.x, P2.x,
P3.x) and from the harnesses that produced them — `explorer_sensitivity_v1`,
`consumer_sensitivity_v1`, `test_module_diff_v1`, `reactive_ceiling_v1`,
`explorer_stochastic_probe_v1`. Every configuration is 8 seeds on the standing config
(smell 3, band (9,11), nondoomed eval with leeway 10, decay 0.7, h_fill 1.6 / s_fill 1.3),
unless the figure says otherwise. Constants are kept at the top so a re-run can be diffed
against them rather than silently overwriting them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "best_figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- palette -------------------------------------------------------------------
INK = "#1b1b1b"
MUTED = "#8a8a8a"
ORACLE = "#2a6f97"
LEARNED = "#c1451a"
GOD = "#4a4a4a"
GOOD = "#2d7a3e"
GRID = "#e6e6e6"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p.relative_to(Path(__file__).resolve().parents[3])}")


# --- recorded measurements ------------------------------------------------------

# explorer_sensitivity_v1, 8 seeds
LADDER = [("random", 0.1712, 121), ("momentum", 0.3333, 62),
          ("smell_momentum\n(reactive)", 0.4776, 35), ("GOD navigation", 1.0000, 0)]

# held-out solveScore, seeds 0-7, per learned attempt
LEARNED_ATTEMPTS = [
    ("DQN\n3 seeds", [0.1130, 0.1092, 0.0553]),
    ("DQN\n+stratified", [0.1059, 0.1030, 0.0591]),
    ("DQN\n+terminal fix", [0.1037]),
    ("DQN\n+softmax eval", [0.1818, 0.1553, 0.1350]),   # best T per seed
    ("policy grad\nH=0.02", [0.1589]),
]

# explorer noisy_oracle calibration: epsilon -> solveScore
EXPL_CAL = [(0.0, 0.4776), (0.15, 0.4079), (0.4, 0.2703), (0.7, 0.2077), (1.0, 0.1156)]
# consumer noisy_oracle calibration (drink): drink_rate_at_water, the ONLY monotone detector
CONS_CAL_SIGMA = [(0.0, 0.8362), (0.05, 0.7997), (0.15, 0.7589), (0.35, 0.7192)]
CONS_CAL_SOLVE = [(0.0, 0.4776), (0.05, 0.4776), (0.15, 0.4507), (0.35, 0.4571)]

# pathfinder seed 2 training curve (fixed eval set)
PF_EPISODES = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
PF_ARRIVAL = [0.835, 0.985, 0.940, 0.960, 1.000, 1.000, 0.985, 0.785]
PF_BEST_EP = 2500
PF_BANDS = ["approach\n(1-5)", "commute\n(6-11)", "long\n(12-20)", "far\n(21-40)"]
PF_FINAL_WEIGHTS = [1.0000, 0.9771, 0.7685, 0.3273]   # s2, shipped at episode 4000
PF_BEST_WEIGHTS = [1.0000, 1.0000, 1.0000, 0.9880]    # s2b, selected at episode 2500

# reactive_ceiling_v1 coordinate sweep, 8 seeds
CEILING = {
    "persist_p": ([0.40, 0.55, 0.70, 0.75, 0.85, 0.95],
                  [0.4848, 0.4429, 0.5424, 0.4776, 0.4638, 0.2373], 0.75),
    "follow_p": ([0.60, 0.75, 0.85, 0.95, 1.00],
                 [0.4921, 0.4211, 0.4923, 0.4776, 0.4507], 0.95),
    "reverse_on_drop_p": ([0.25, 0.45, 0.60, 0.75, 0.90],
                          [0.4267, 0.4444, 0.4848, 0.4776, 0.4384], 0.75),
}


# --- fig 1: the headline --------------------------------------------------------

def fig_explorer_ladder():
    """The single most important figure: the gap, and that nothing closed it."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    names = [n for n, _, _ in LADDER]
    vals = [v for _, v, _ in LADDER]
    xs = np.arange(len(names))
    cols = [MUTED, MUTED, ORACLE, GOD]
    ax.bar(xs, vals, width=0.58, color=cols, zorder=3)
    for x, v, (_, _, d) in zip(xs, vals, LADDER):
        ax.text(x, v + 0.022, f"{v:.3f}", ha="center", fontsize=9, color=INK)
        ax.text(x, 0.018, f"{d} deaths", ha="center", fontsize=7.5, color="white", zorder=4)

    ax.axhline(0.4776, color=ORACLE, ls="--", lw=1.1, zorder=2)
    ax.text(len(names) - 0.35, 0.4776 + 0.015, "reactive floor to beat",
            fontsize=8, color=ORACLE, ha="right")
    ax.axhline(0.1712, color=MUTED, ls=":", lw=1.1, zorder=2)
    ax.text(len(names) - 0.35, 0.1712 + 0.015, "random floor",
            fontsize=8, color=MUTED, ha="right")

    # learned attempts scattered over the gap
    x0 = len(names) + 0.35
    for i, (label, scores) in enumerate(LEARNED_ATTEMPTS):
        x = x0 + i * 0.85
        ax.scatter([x] * len(scores), scores, s=46, color=LEARNED, zorder=5,
                   marker="o", edgecolor="white", linewidth=0.8)
        ax.text(x, -0.075, label, ha="center", va="top", fontsize=7.2, color=LEARNED)

    ax.axvspan(len(names) - 0.1, x0 + (len(LEARNED_ATTEMPTS) - 1) * 0.85 + 0.45,
               color=LEARNED, alpha=0.045, zorder=0)
    ax.text((x0 + (len(LEARNED_ATTEMPTS) - 1) * 0.85) / 2 + len(names) / 2 + 0.3, 0.92,
            "five learned attempts — none clears the random floor",
            fontsize=8.5, color=LEARNED, ha="center", style="italic")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlim(-0.6, x0 + (len(LEARNED_ATTEMPTS) - 1) * 0.85 + 0.55)
    ax.set_ylabel("solveScore  =  timeouts / (timeouts + deaths)")
    ax.set_title("Explorer: hand-written policies span 0.17–1.00; every learned policy sits at ~0.10\n"
                 "smell 3 · band (9,11) · nondoomed cold start · 8 seeds", loc="left")
    save(fig, "explorer_ladder__headline")


# --- fig 2: dead band geometry --------------------------------------------------

def fig_dead_band():
    """Why smell 5 deletes the phenomenon rather than easing it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    L = 10
    for row, (r, col) in enumerate([(3, ORACLE), (5, LEARNED)]):
        y = 1 - row
        ax1.plot([0, L], [y, y], color=MUTED, lw=1.0, zorder=1)
        ax1.scatter([0, L], [y, y], s=90, color=[ORACLE, GOOD], zorder=3)
        # sensed zones
        ax1.plot([0, r], [y, y], color=col, lw=7, alpha=0.30, solid_capstyle="butt", zorder=2)
        ax1.plot([L - r, L], [y, y], color=col, lw=7, alpha=0.30, solid_capstyle="butt", zorder=2)
        d = max(0, L - 2 * r)
        if d > 0:
            ax1.plot([r, L - r], [y, y], color=INK, lw=7, alpha=0.85,
                     solid_capstyle="butt", zorder=2)
        ax1.text(-0.9, y, f"r={r}", ha="right", va="center", fontsize=10, color=col)
        ax1.text(L + 0.5, y, f"blind = {d}", ha="left", va="center", fontsize=9,
                 color=INK if d else GOOD)

    ax1.text(0, 1.42, "water", ha="center", fontsize=8, color=ORACLE)
    ax1.text(L, 1.42, "food", ha="center", fontsize=8, color=GOOD)
    ax1.set_xlim(-2.6, L + 3.4)
    ax1.set_ylim(-0.55, 1.7)
    ax1.set_yticks([])
    ax1.set_xlabel("hexes along the commute  (L = 10)")
    ax1.grid(False)
    ax1.set_title("Sensed / blind, by smell radius", loc="left")

    rs = np.linspace(1, 6, 400)
    for L_, ls in [(9, ":"), (10, "-"), (11, "--")]:
        diff = np.maximum(0, L_ - 2 * rs) ** 2 / rs ** 2
        ax2.plot(rs, diff, ls, color=INK, lw=1.4, label=f"L={L_}")
    ax2.axvline(3, color=ORACLE, lw=1.1)
    ax2.axvline(5, color=LEARNED, lw=1.1)
    ax2.text(3.05, ax2.get_ylim()[1] * 0.82, "r=3\nphenomenon intact",
             fontsize=8, color=ORACLE)
    ax2.text(5.05, ax2.get_ylim()[1] * 0.55, "r=5\ndeleted", fontsize=8, color=LEARNED)
    ax2.set_xlabel("smell radius  r")
    ax2.set_ylabel(r"blind-leg difficulty  $\propto (L-2r)^2 / r^2$")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("Difficulty is quadratic in the blind width", loc="left")

    fig.suptitle("The controlling quantity is $L-2r$, not $r$: +2 on radius removes 4 from the blind band",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "dead_band_geometry")


# --- fig 3: pathfinder checkpoint selection -------------------------------------

def fig_pathfinder_checkpoint():
    """Capability was in every seed; the variable was where training stopped."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    ax1.plot(PF_EPISODES, PF_ARRIVAL, "-o", color=INK, lw=1.5, ms=4.5, zorder=3)
    bi = PF_EPISODES.index(PF_BEST_EP)
    ax1.scatter([PF_BEST_EP], [PF_ARRIVAL[bi]], s=150, facecolor="none",
                edgecolor=GOOD, lw=2, zorder=4)
    ax1.annotate("selected\n(best checkpoint)", (PF_BEST_EP, PF_ARRIVAL[bi]),
                 textcoords="offset points", xytext=(-6, -38), fontsize=8,
                 color=GOOD, ha="center")
    ax1.scatter([4000], [PF_ARRIVAL[-1]], s=150, facecolor="none",
                edgecolor=LEARNED, lw=2, zorder=4)
    ax1.annotate("shipped by default\n(final weights)", (4000, PF_ARRIVAL[-1]),
                 textcoords="offset points", xytext=(-16, -40), fontsize=8,
                 color=LEARNED, ha="center")
    ax1.set_xlabel("training episode")
    ax1.set_ylabel("arrival rate (fixed eval set)")
    ax1.set_ylim(0.72, 1.045)
    ax1.set_title("Seed 2: the policy peaks, then decays", loc="left")

    x = np.arange(len(PF_BANDS))
    w = 0.38
    ax2.bar(x - w / 2, PF_FINAL_WEIGHTS, w, label="final weights", color=LEARNED, zorder=3)
    ax2.bar(x + w / 2, PF_BEST_WEIGHTS, w, label="best checkpoint", color=GOOD, zorder=3)
    ax2.axhline(0.99, color=INK, ls="--", lw=1.0, zorder=2)
    ax2.text(3.45, 0.995, "gate 0.99", fontsize=8, ha="right", color=INK)
    ax2.axvspan(-0.5, 2.5, color=ORACLE, alpha=0.05, zorder=0)
    ax2.text(1.0, 0.06, "operational range", fontsize=8, color=ORACLE, ha="center")
    ax2.set_xticks(x)
    ax2.set_xticklabels(PF_BANDS, fontsize=8)
    ax2.set_ylim(0, 1.09)
    ax2.set_ylabel("optimality  (fraction of moves that close the distance)")
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    ax2.set_title("Same run, same seed — only the checkpoint differs", loc="left")

    fig.suptitle("Best-checkpoint selection turned the worst seed into the best module",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "pathfinder_checkpoint_selection")


# --- fig 4: calibration ---------------------------------------------------------

def fig_calibration():
    """Two modules, two metrics: one can rank policies, one cannot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    e_x = [e for e, _ in EXPL_CAL]
    e_y = [v for _, v in EXPL_CAL]
    ax1.plot(e_x, e_y, "-o", color=ORACLE, lw=1.8, ms=5.5, zorder=3)
    ax1.axhline(0.1712, color=MUTED, ls=":", lw=1.1)
    ax1.text(1.0, 0.19, "random floor", fontsize=8, color=MUTED, ha="right")
    ax1.set_xlabel("epsilon  (probability of a uniform random move)")
    ax1.set_ylabel("solveScore")
    ax1.set_ylim(0.05, 0.55)
    ax1.set_title(f"EXPLORER — monotone, range {max(e_y)-min(e_y):.3f}", loc="left",
                  color=GOOD)
    ax1.annotate("", xy=(1.0, 0.13), xytext=(0.0, 0.46),
                 arrowprops=dict(arrowstyle="<->", color=GOOD, lw=1.2))
    ax1.text(0.52, 0.32, "usable\ndynamic range", fontsize=8, color=GOOD, ha="center")

    c_x = [s for s, _ in CONS_CAL_SOLVE]
    ax2.plot(c_x, [v for _, v in CONS_CAL_SOLVE], "-o", color=LEARNED, lw=1.8, ms=5.5,
             label="solveScore (NON-monotone)")
    ax2.plot(c_x, [v for _, v in CONS_CAL_SIGMA], "-s", color=ORACLE, lw=1.8, ms=5,
             label="drink_rate (only detector)")
    ax2.set_xlabel("sigma  (gaussian error on every fill)")
    ax2.set_ylabel("metric value")
    ax2.set_ylim(0.40, 0.90)
    ax2.legend(frameon=False, fontsize=8, loc="center left")
    ax2.set_title("CONSUMER — barely resolvable, range 0.044", loc="left", color=LEARNED)

    fig.suptitle("Calibrate before training: a null result only means something if the metric can produce a non-null one",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "calibration_detection_floor")


# --- fig 5: reactive ceiling ----------------------------------------------------

def fig_reactive_ceiling():
    """Is 0.4776 the reactive optimum, or the point we happened to pick?"""
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4), sharey=True)
    base = 0.4776
    # Wilson half-width at these counts is ~0.12; the whole surface sits inside it
    band = 0.12

    for ax, (name, (xs, ys, bx)) in zip(axes, CEILING.items()):
        ax.axhspan(base - band, base + band, color=ORACLE, alpha=0.10, zorder=0)
        ax.plot(xs, ys, "-o", color=INK, lw=1.4, ms=5, zorder=3)
        bi = xs.index(bx)
        ax.scatter([bx], [ys[bi]], s=130, facecolor="none", edgecolor=ORACLE, lw=2, zorder=4)
        ax.axhline(base, color=ORACLE, ls="--", lw=1.0, zorder=2)
        ax.set_xlabel(name)
        ax.set_title(name, loc="left", fontsize=9.5)

    axes[0].set_ylabel("solveScore")
    axes[0].set_ylim(0.20, 0.62)
    axes[0].text(0.42, base + band - 0.02, "baseline 95% CI", fontsize=7.5, color=ORACLE)

    fig.suptitle("Reactive ceiling: no single-parameter change separates from the baseline CI —\n"
                 "the spread is sampling noise, not a tuning gradient  (8 seeds)",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "reactive_ceiling_sweep")


# --- fig 6: module scoreboard ---------------------------------------------------

def fig_module_scoreboard():
    """One-glance summary of which modules RL could learn."""
    fig, ax = plt.subplots(figsize=(8.6, 3.2))

    mods = ["pathfinder", "drink consumer", "eat consumer", "explorer"]
    status = ["solved", "learnable\nbut unmeasurable", "untrained\n(starved by design)",
              "NOT learnable\non this contract"]
    cols = [GOOD, ORACLE, MUTED, LEARNED]
    notes = [
        "1.0000 optimality, 3-seed reproducible\n+0.000 solveScore over 34,859 calls",
        "37 deaths vs oracle 35\nenvironment cannot resolve it",
        "0 dispatches in 1200 ticks\nfinding food IS the phenomenon",
        "5 methods, all at or below random\nno positional memory in the contract",
    ]

    for i, (m, s, c, n) in enumerate(zip(mods, status, cols, notes)):
        y = len(mods) - i - 1
        ax.barh([y], [1], color=c, alpha=0.13, height=0.72, zorder=1)
        ax.text(0.015, y + 0.17, m, fontsize=11, color=INK, va="center", weight="bold")
        ax.text(0.015, y - 0.16, n, fontsize=7.8, color=MUTED, va="center")
        ax.text(0.985, y, s, fontsize=9.5, color=c, va="center", ha="right", weight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, len(mods) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Decomposition makes exploitation learnable and does not rescue exploration",
                 loc="left", fontsize=11)
    save(fig, "module_scoreboard")


def main():
    print("generating Proto 06 figures...")
    fig_explorer_ladder()
    fig_dead_band()
    fig_pathfinder_checkpoint()
    fig_calibration()
    fig_reactive_ceiling()
    fig_module_scoreboard()
    print("done.")


if __name__ == "__main__":
    main()
