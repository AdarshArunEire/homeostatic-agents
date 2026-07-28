"""
make_figures_v1.py  —  every Proto 06 figure, from recorded measurements.

    python figures/make_figures_v1.py

Writes PNGs to prototypes/06_feudal/results/best_figures/.

STYLE. Matches the existing repo figures (plot_fn_v5): matplotlib defaults, short lowercase
titles, viridis/inferno/magma where a colormap helps. Explanation belongs in the caption next
to the figure, not inside it — a plot with a paragraph printed on it is harder to read, not
easier.

PROVENANCE. Numbers are transcribed from the runs recorded in BUILDNOTES (P1.x, P2.x, P3.x,
P4.x) and from the harnesses that produced them. Every configuration is 8 seeds on the standing
config (smell 3, band (9,11), nondoomed eval with leeway 10, decay 0.7, h_fill 1.6/s_fill 1.3)
unless the figure says otherwise. Constants sit at the top so a re-run can be diffed against
them rather than silently overwriting them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "best_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight"})

ORACLE = "#2166ac"
LEARNED = "#b2182b"
NEUTRAL = "#777777"


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}")


# --- recorded measurements ------------------------------------------------------

LADDER = [("random", 0.1712), ("momentum", 0.3333),
          ("smell_momentum", 0.4776), ("god", 1.0000)]

LEARNED_EXPL = {
    "DQN": [0.1130, 0.1092, 0.0553],
    "+stratified": [0.1059, 0.1030, 0.0591],
    "+terminal": [0.1037],
    "+softmax": [0.1818, 0.1553, 0.1350],
    "policy grad": [0.1589],
}

EXPL_CAL = [(0.0, 0.4776), (0.15, 0.4079), (0.4, 0.2703), (0.7, 0.2077), (1.0, 0.1156)]
CONS_CAL_SIGMA = [(0.0, 0.8362), (0.05, 0.7997), (0.15, 0.7589), (0.35, 0.7192)]
CONS_CAL_SOLVE = [(0.0, 0.4776), (0.05, 0.4776), (0.15, 0.4507), (0.35, 0.4571)]
ORCH_CAL_EPS = [(0.0, 0.4776), (0.05, 0.4366), (0.15, 0.4267), (0.35, 0.2066), (0.7, 0.0000)]

PF_EPISODES = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
PF_ARRIVAL = [0.835, 0.985, 0.940, 0.960, 1.000, 1.000, 0.985, 0.785]
PF_BANDS = ["approach", "commute", "long", "far"]
PF_FINAL = [1.0000, 0.9771, 0.7685, 0.3273]
PF_BEST = [1.0000, 1.0000, 1.0000, 0.9880]

CEILING = {
    "persist_p": ([0.40, 0.55, 0.70, 0.75, 0.85, 0.95],
                  [0.4848, 0.4429, 0.5424, 0.4776, 0.4638, 0.2373], 0.75),
    "follow_p": ([0.60, 0.75, 0.85, 0.95, 1.00],
                 [0.4921, 0.4211, 0.4923, 0.4776, 0.4507], 0.95),
    "reverse_on_drop_p": ([0.25, 0.45, 0.60, 0.75, 0.90],
                          [0.4267, 0.4444, 0.4848, 0.4776, 0.4384], 0.75),
}

# Orchestrator action shares, EVAL SEGMENT, 8 seeds — measured by tests/action_mix_v1.py.
#
# The oracle's mix had never been recorded before that script existed; an earlier draft of
# this file carried invented values for it. Measuring took 30 seconds and the real contrast is
# sharper than the guess: the oracle arbitrates at exactly 1.0:1 between the two drives.
#
# Note the learned shares here differ slightly from the training log (0.537/0.061/0.206/0.196)
# because that figure pooled the whole run; these are eval-segment only, matching every other
# number in the ledger.
ORCH_MIX = [("GO_WATER", 0.571), ("GO_FOOD", 0.075), ("CONSUME", 0.235), ("EXPLORE", 0.119)]
ORACLE_MIX = [("GO_WATER", 0.342), ("GO_FOOD", 0.342), ("CONSUME", 0.192), ("EXPLORE", 0.123)]
ORCH_CAUSES = [("hydration", 62), ("satiation", 31)]
ORACLE_CAUSES = [("hydration", 29), ("satiation", 6)]

# CORRECTED IN P4.2 — the raw 7.6:1 read off ORCH_MIX above is not an arbitration ratio.
# GO_FOOD is masked illegal unless food_known, and the learned module holds food_known on
# only 32.1% of eval ticks against the oracle's 92.8%. Conditioning progressively:
#
#   7.63 : 1   raw, as originally published        <- overstates by 2.3x
#   3.37 : 1   restricted to ticks where BOTH GO_* were legal
#  24.22 : 1   ...and standing on NEITHER resource (the only genuine travel decisions)
#
# The last is the honest arbitration number and it is far WORSE than the headline, because
# most GO_FOOD is emitted while still standing on water and abandoned one tick later.
# Source: tests/orchestrator_legality_v1.py, 8 seeds.
ORCH_RATIOS = [("raw\n(as published)", 7.63), ("both GO_*\nlegal", 3.37),
               ("...and off\nany resource", 24.22)]

# P4.3 — enforced commitment horizon at INFERENCE on the existing o0 weights. 24 seeds.
# tests/orchestrator_hold_sweep_v1.py. Wilson intervals as printed.
ORACLE_24 = (0.5053, (0.43, 0.58))
HOLD_SWEEP = [(1, 0.1683, (0.13, 0.21)), (15, 0.2689, (0.22, 0.33)),
              (20, 0.2624, (0.21, 0.32)), (25, 0.2259, (0.18, 0.28)),
              (40, 0.2194, (0.17, 0.27))]
# Control A: uniform-random legal option, then held. Zero at both horizons — persistence is
# not intrinsically valuable, it helps THIS policy and harms an arbitrary one.
HOLD_RANDOM = [(1, 0.0000), (15, 0.0000)]

# P4.4 — deliberation cost, learned rather than enforced. Monotone toward degeneracy:
# every arm collapses onto CONSUME and dies of thirst standing on food.
# (delib_cost, CONSUME share, CONSUME dwell in ticks)
DELIB = [(0.000, 0.235, 1.0), (0.002, 0.573, 30.4), (0.020, 0.789, 307.2)]

# Learned performance as a fraction of the oracle it replaced.
#   drink        0.4638 / 0.4776   (consumer_sensitivity)
#   orchestrator 0.1622 / 0.4776   (orchestrator_sensitivity)
#   explorer     0.1059 / 0.4776   (best held-out DQN seed)
# The pathfinder is on a DIFFERENT basis: it has no solveScore of its own, and 1.000 records
# a +0.000 in-sim delta against the oracle over 34,859 invocations. Same conclusion, different
# measurement — noted rather than blended silently.
MODULE_GAP = [
    ("pathfinder", 1.000),
    ("drink consumer", 0.971),
    ("orchestrator", 0.340),
    ("explorer", 0.222),
]


# --- figures --------------------------------------------------------------------

def fig_explorer_ladder():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    names = [n for n, _ in LADDER]
    vals = [v for _, v in LADDER]
    ax.bar(range(len(names)), vals, width=0.6,
           color=[NEUTRAL, NEUTRAL, ORACLE, "#444444"])

    x0 = len(names) + 0.4
    for i, (label, scores) in enumerate(LEARNED_EXPL.items()):
        x = x0 + i * 0.8
        ax.scatter([x] * len(scores), scores, s=40, color=LEARNED, zorder=4)

    ax.axhline(0.4776, color=ORACLE, ls="--", lw=1)
    ax.axhline(0.1712, color=NEUTRAL, ls=":", lw=1)

    ticks = list(range(len(names))) + [x0 + i * 0.8 for i in range(len(LEARNED_EXPL))]
    labels = names + list(LEARNED_EXPL)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("solveScore")
    ax.set_title("explorer: hand-written policies vs every learned attempt", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "explorer_ladder__headline")


def fig_dead_band():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    L = 10
    for row, r in enumerate([3, 5]):
        y = 1 - row
        ax1.plot([0, L], [y, y], color="0.75", lw=1)
        ax1.plot([0, r], [y, y], color=ORACLE, lw=8, solid_capstyle="butt", alpha=0.5)
        ax1.plot([L - r, L], [y, y], color=ORACLE, lw=8, solid_capstyle="butt", alpha=0.5)
        d = max(0, L - 2 * r)
        if d:
            ax1.plot([r, L - r], [y, y], color="0.15", lw=8, solid_capstyle="butt")
        ax1.scatter([0, L], [y, y], s=70, color="0.2", zorder=3)
        ax1.text(-1.2, y, f"r={r}", ha="right", va="center", fontsize=9)

    ax1.set_xlim(-2.5, L + 1)
    ax1.set_ylim(-0.6, 1.6)
    ax1.set_yticks([])
    ax1.set_xlabel("hexes along the commute")
    ax1.set_title("sensed (blue) vs blind (black), L = 10", fontsize=10)

    rs = np.linspace(1, 6, 400)
    for L_, ls in [(9, ":"), (10, "-"), (11, "--")]:
        ax2.plot(rs, np.maximum(0, L_ - 2 * rs) ** 2 / rs ** 2, ls, color="0.2", lw=1.3,
                 label=f"L={L_}")
    ax2.axvline(3, color=ORACLE, lw=1)
    ax2.axvline(5, color=LEARNED, lw=1)
    ax2.set_xlabel("smell radius r")
    ax2.set_ylabel(r"$(L-2r)^2/r^2$")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("blind-leg difficulty", fontsize=10)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    save(fig, "dead_band_geometry")


def fig_pathfinder_checkpoint():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    ax1.plot(PF_EPISODES, PF_ARRIVAL, "-o", color="0.2", lw=1.4, ms=4)
    ax1.scatter([2500], [1.000], s=120, facecolor="none", edgecolor="#1a9850", lw=2, zorder=4)
    ax1.scatter([4000], [0.785], s=120, facecolor="none", edgecolor=LEARNED, lw=2, zorder=4)
    ax1.set_xlabel("training episode")
    ax1.set_ylabel("arrival rate")
    ax1.set_title("seed 2: peaks, then decays", fontsize=10)
    ax1.grid(alpha=0.25)

    x = np.arange(len(PF_BANDS))
    ax2.bar(x - 0.19, PF_FINAL, 0.38, color=LEARNED, label="final weights")
    ax2.bar(x + 0.19, PF_BEST, 0.38, color="#1a9850", label="best checkpoint")
    ax2.axhline(0.99, color="0.3", ls="--", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(PF_BANDS, fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("optimality")
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    ax2.set_title("same run, different checkpoint", fontsize=10)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save(fig, "pathfinder_checkpoint_selection")


def fig_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    for ax, data, title, ylab in [
        (axes[0], ORCH_CAL_EPS, "orchestrator", "solveScore"),
        (axes[1], EXPL_CAL, "explorer", "solveScore"),
    ]:
        ax.plot([e for e, _ in data], [v for _, v in data], "-o", color=ORACLE, lw=1.6, ms=5)
        ax.set_xlabel("epsilon")
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.03, 0.55)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot([s for s, _ in CONS_CAL_SOLVE], [v for _, v in CONS_CAL_SOLVE], "-o",
            color=LEARNED, lw=1.6, ms=5, label="solveScore")
    ax.plot([s for s, _ in CONS_CAL_SIGMA], [v for _, v in CONS_CAL_SIGMA], "-s",
            color=ORACLE, lw=1.6, ms=4, label="drink_rate")
    ax.set_xlabel("sigma")
    ax.set_ylim(0.40, 0.90)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("consumer", fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save(fig, "calibration_detection_floor")


def fig_reactive_ceiling():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    base, band = 0.4776, 0.12

    for ax, (name, (xs, ys, bx)) in zip(axes, CEILING.items()):
        ax.axhspan(base - band, base + band, color=ORACLE, alpha=0.12)
        ax.plot(xs, ys, "-o", color="0.2", lw=1.4, ms=5)
        ax.scatter([bx], [ys[xs.index(bx)]], s=110, facecolor="none",
                   edgecolor=ORACLE, lw=2, zorder=4)
        ax.set_xlabel(name, fontsize=9)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("solveScore")
    axes[0].set_ylim(0.20, 0.62)
    fig.suptitle("reactive parameter sweep, shaded = baseline 95% CI", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "reactive_ceiling_sweep")


def fig_orchestrator_mix():
    """The water-cult attractor: oracle arbitrates 1:1, the learned policy 7.6:1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.3),
                                   gridspec_kw={"width_ratios": [1.7, 1]})

    labels = [n for n, _ in ORACLE_MIX]
    x = np.arange(len(labels))
    ax1.bar(x - 0.19, [v for _, v in ORACLE_MIX], 0.38, color=ORACLE, label="oracle")
    ax1.bar(x + 0.19, [v for _, v in ORCH_MIX], 0.38, color=LEARNED, label="learned")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5, rotation=15)
    ax1.set_ylabel("share of decisions")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title("what it chose", fontsize=10)
    ax1.grid(axis="y", alpha=0.25)

    cnames = [n for n, _ in ORCH_CAUSES]
    cx = np.arange(len(cnames))
    ax2.bar(cx - 0.19, [v for _, v in ORACLE_CAUSES], 0.38, color=ORACLE)
    ax2.bar(cx + 0.19, [v for _, v in ORCH_CAUSES], 0.38, color=LEARNED)
    ax2.set_xticks(cx)
    ax2.set_xticklabels(cnames, fontsize=9)
    ax2.set_ylabel("eval deaths")
    ax2.set_title("what killed it", fontsize=10)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save(fig, "orchestrator_action_mix")


def fig_module_gap():
    """Learned performance as a fraction of the oracle it replaced."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))

    names = [n for n, _ in MODULE_GAP]
    vals = [v for _, v in MODULE_GAP]
    y = np.arange(len(names))[::-1]
    cmap = plt.cm.viridis
    ax.barh(y, vals, height=0.55, color=[cmap(v * 0.85) for v in vals])
    ax.axvline(1.0, color="0.3", ls="--", lw=1)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("learned / oracle")
    ax.set_title("how close each learned module came to the policy it replaced", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "module_gap")


def fig_arbitration_denominator():
    """The published ratio, and what conditioning on legality does to it (P4.2)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    names = [n for n, _ in ORCH_RATIOS]
    vals = [v for _, v in ORCH_RATIOS]
    cols = [NEUTRAL, ORACLE, LEARNED]
    ax.bar(range(len(names)), vals, width=0.55, color=cols)
    ax.axhline(1.0, color="0.3", ls="--", lw=1)
    ax.text(2.42, 1.15, "oracle 1.0:1", fontsize=8, color="0.3", ha="right")

    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.2f}:1", ha="center", fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0, 27)
    ax.set_ylabel("GO_WATER : GO_FOOD")
    ax.set_title("the same policy, three denominators", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "arbitration_denominator")


def fig_commitment():
    """P4.3 enforced horizon (works) beside P4.4 deliberation cost (degenerates)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))

    xs = [h for h, _, _ in HOLD_SWEEP]
    ys = [v for _, v, _ in HOLD_SWEEP]
    lo = [v - c[0] for _, v, c in HOLD_SWEEP]
    hi = [c[1] - v for _, v, c in HOLD_SWEEP]
    ax1.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=LEARNED, capsize=3,
                 lw=1.6, ms=5, label="learned o0, held")

    rx = [h for h, _ in HOLD_RANDOM]
    ry = [v for _, v in HOLD_RANDOM]
    ax1.plot(rx, ry, "s--", color=NEUTRAL, ms=5, lw=1.2,
             label="random option, held (control)")

    ax1.axhline(ORACLE_24[0], color=ORACLE, ls="--", lw=1.2)
    ax1.text(40, ORACLE_24[0] + 0.012, "oracle", fontsize=8, color=ORACLE, ha="right")
    ax1.set_xlabel("commitment horizon (ticks)")
    ax1.set_ylabel("solveScore")
    ax1.set_ylim(-0.03, 0.60)
    ax1.set_title("enforced: interior optimum near commute length", fontsize=10)
    ax1.legend(fontsize=8, loc="center right")
    ax1.grid(alpha=0.25)

    dx = [d for d, _, _ in DELIB]
    dc = [c for _, c, _ in DELIB]
    dw = [w for _, _, w in DELIB]
    ax2.plot(dx, dc, "o-", color=LEARNED, lw=1.6, ms=5)
    for x, c, w in zip(dx, dc, dw):
        ax2.annotate(f"dwell {w:.0f}", (x, c), textcoords="offset points",
                     xytext=(6, -11), fontsize=8, color="0.35")
    ax2.axhline(ORACLE_MIX[2][1], color=ORACLE, ls="--", lw=1.2)
    ax2.text(0.019, ORACLE_MIX[2][1] + 0.02, "oracle CONSUME share",
             fontsize=8, color=ORACLE, ha="right")
    ax2.set_xlabel("deliberation cost per switch")
    ax2.set_ylabel("CONSUME share of eval ticks")
    ax2.set_ylim(0, 0.9)
    ax2.set_title("learned: monotone collapse into standing still", fontsize=10)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    save(fig, "commitment")


def main():
    print("figures ->", OUT)
    fig_explorer_ladder()
    fig_dead_band()
    fig_pathfinder_checkpoint()
    fig_calibration()
    fig_reactive_ceiling()
    fig_orchestrator_mix()
    fig_module_gap()
    fig_arbitration_denominator()
    fig_commitment()


if __name__ == "__main__":
    main()
