"""
make_agent_gif_v1.py  —  the picture this repo is missing.

Every other figure here is a bar chart. This one shows the actual thing: a hex world with
water and food in it, an agent walking between them, drinking and eating, and two drives
draining underneath while it does.

Scent is drawn as a falloff, not a flat disc, so the DEAD BAND is visible as the bare
stretch in the middle of the commute where neither halo reaches.

Both resources are scented (see `update_smell_history` in sim_instance_v3), which is what
makes the blind width `L - 2r` rather than `L - r`. Proto 04's food-only sensorium is a
different, earlier thing.

THIS IS A DEMO, NOT A RESULT. The seed is picked to look good. Caption it that way.

    python figures/make_agent_gif_v1.py --scan 16
    python figures/make_agent_gif_v1.py --seed 1

    # P4.3, side by side: same weights, only the re-decision frequency differs
    python figures/make_agent_gif_v1.py --scan 24 --left orch --vs orch-held
    python figures/make_agent_gif_v1.py --seed N --left orch --vs orch-held \
        --out results/best_figures/commitment_alive.gif

PAIRED RUNS SHARE A MAP ONLY FOR THE FIRST EVAL LIFE. `MapCurriculum.build(eval_mode=True)`
draws a fresh map seed on every eval respawn, so two stacks that die at different ticks
diverge into different worlds. Paired mode therefore locks both panels to the segment
starting at the eval boundary, where `eval_rng` has been drawn exactly once for both. Spawn
*coordinates* still differ (the global RNG has seen different histories by then) — same
world, independent starting points.

Reads only what `sim_instance` already returns, so it adds no coupling to the sim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROTO_DIR = Path(__file__).resolve().parents[1]
if str(_PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(_PROTO_DIR))

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import RegularPolygon  # noqa: E402

OUT_DEFAULT = _PROTO_DIR / "results" / "best_figures" / "agent_alive.gif"

SQ3 = np.sqrt(3.0)
HMAX = 3.0
DEATH_THRESH = 0.05
IDEAL = 1.0

# Scent falloff. 1.0 = full scent colour, 0.0 = bare ground. The outer ring stays well
# clear of zero on purpose: a linear ramp fades the edge into the background and the dead
# band stops being legible, which is the one thing this figure exists to show.
W_NEAR, W_FAR, W_GAMMA = 0.95, 0.40, 0.75

DARK = dict(
    fig="#12151a", base="#1c2026", edge="#262b33",
    water="#4bb3e6", food="#63b85f",
    scent_w="#1b3d55", scent_f="#1d3d24",
    trail="#7f8792", text="#d8dce2", dim="#8b929c",
    ideal="#5f6672", death="#ff5f56", ring="#12151a",
)
LIGHT = dict(
    fig="#ffffff", base="#f6f6f4", edge="#e6e6e3",
    water="#3d7ea6", food="#4f8f4a",
    scent_w="#bcd9ee", scent_f="#c6e4c1",
    trail="#8a8a86", text="#222222", dim="#666666",
    ideal="#999999", death="#c0392b", ring="#ffffff",
)

# standing Proto 06 config — matches tests/probe_infra_v1.py
STANDING_PARAMS = {
    "orchestrator": {"h_fill": 1.6, "s_fill": 1.3, "h_crit": 0.7, "s_crit": 0.7},
    "pathfinder": {},
    "eat": {"fill_target": 1.3},
    "drink": {"fill_target": 1.6},
    "explorer": {"persist_p": 0.75, "avoid_reverse": True, "trend_eps": 0.01,
                 "follow_p": 0.95, "reverse_on_drop_p": 0.75},
}

# Named stacks. Tags are the ones recorded in results/weights/MANIFEST.json — the .pt
# blobs are gitignored, so anything here needs its trainer run first.
#
# `eat` is absent by necessity, not oversight: no eat module was ever trained. A
# 1200-tick oracle run produced 279 drink dispatches and 0 eat dispatches, so the head
# would have been fitted on nothing. Every stack below runs the oracle eat consumer.
STACKS = {
    "oracle":   ("oracle stack  ·  0.478", {}),
    "random":   ("random explorer  ·  0.171", {"explorer": "random"}),
    "momentum": ("momentum, no scent  ·  0.333", {"explorer": "momentum"}),
    "god":      ("GOD navigation  ·  1.000", {"god": True}),
    "orch":     ("learned o0, re-decides every tick  ·  0.168",
                 {"orchestrator": "o0"}),
    # Same weights, same argmax — only the re-decision frequency differs. This is P4.3 made
    # visible: the left panel abandons GO_FOOD after one tick, the right one walks the
    # commute. Nothing was retrained between them.
    "orch-held": ("learned o0, holds intent 15 ticks  ·  0.269",
                  {"orchestrator": "o0", "orch_kw": {"hold": 15}}),
    "path":     ("learned pathfinder s2b  ·  0.478", {"pathfinder": "s2b"}),
    "drink":    ("learned drink v1  ·  0.464", {"drink": "v1"}),
    "learned":  ("learned orch + path + drink", {"orchestrator": "o0",
                                                 "pathfinder": "s2b", "drink": "v1"}),
    "all":      ("all learned (incl. explorer)", {"orchestrator": "o0", "pathfinder": "s2b",
                                                  "drink": "v1", "explorer_w": "g0"}),
}


def action_colour(name: str, P: dict) -> str:
    return {"GO_WATER": P["water"], "GO_FOOD": P["food"],
            "CONSUME": "#e8b84b", "EXPLORE": "#a882d5"}.get(name, P["dim"])


# --- hex geometry ---------------------------------------------------------------

def hex_dist(a, b) -> int:
    aq, ar = a
    bq, br = b
    return (abs(aq - bq) + abs(aq + ar - bq - br) + abs(ar - br)) // 2


def disc(radius: int):
    return [(q, r)
            for q in range(-radius, radius + 1)
            for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1)]


def to_xy(q, r):
    """Axial -> cartesian, pointy-top, circumradius 1."""
    return SQ3 * (q + r / 2.0), 1.5 * r


def lerp(c0, c1, t: float):
    a = np.array(mcolors.to_rgb(c0))
    b = np.array(mcolors.to_rgb(c1))
    return tuple(a + (b - a) * float(np.clip(t, 0.0, 1.0)))


def scent_weight(d: int, r: int) -> float:
    return W_FAR + (W_NEAR - W_FAR) * (1.0 - d / r) ** W_GAMMA


# --- running the sim ------------------------------------------------------------

def run(seed: int, orchestrator=None, pathfinder=None, drink=None, eat=None,
        explorer: str = "smell_momentum", explorer_w=None, god: bool = False,
        orch_kw: dict | None = None, sim_len: int = 7000, eval_len: int = 5000):
    import sim_instance_v3 as S
    from model_modules.contract_v1 import ModuleSpec

    params = {k: dict(v) for k, v in STANDING_PARAMS.items()}
    slots = dict(orchestrator="oracle", pathfinder="oracle", eat="oracle", drink="oracle")

    if orchestrator:
        slots["orchestrator"] = "learned"
        params["orchestrator"] = {"weights": orchestrator, **(orch_kw or {})}
    if pathfinder:
        slots["pathfinder"] = "learned"
        params["pathfinder"] = {"weights": pathfinder, "scale": 20}
    if drink:
        slots["drink"] = "learned"
        params["drink"] = {"weights": drink}
    if eat:
        slots["eat"] = "learned"
        params["eat"] = {"weights": eat}

    if explorer_w:
        explorer = "learned"
        params["explorer"] = {"weights": explorer_w}

    return S.sim_instance(
        seed=seed,
        sim_len=sim_len,
        eval_len=eval_len,
        env_kwargs=dict(radius=20, band=(9, 11), start_coord=(0, 0)),
        decay_mult=0.7,
        smell_radius=3,
        curriculum_mode="band",
        c_min=2, c_max=9, band_width=2,
        life_cap=1000,
        oracle_params=params,
        module_spec=ModuleSpec(**slots, explorer=explorer),
        eval_spawn_nondoomed=True,
        spawn_leeway=10,
        eval_god_memory=god,
        log_every=10 ** 9,
    )


def eval_segments(res):
    """(t_start, t_end, segment_record) for each eval life."""
    segs = res["eval_segments"]
    n = len(res["death_T"])
    out = []
    for i, s in enumerate(segs):
        t0 = int(s["t_start"])
        t1 = int(segs[i + 1]["t_start"]) if i + 1 < len(segs) else n
        out.append((t0, t1, s))
    return out


def first_eval_segment(res):
    """The one life two different stacks are guaranteed to share a map on."""
    EB = int(res["eval_boundary"])
    for t0, t1, seg in eval_segments(res):
        if t0 == EB:
            return t0, t1, seg
    return eval_segments(res)[0]


def count_trips(res, a: int, b: int, seg) -> tuple[int, int]:
    """Completed water->food and food->water crossings inside [a, b), sim's own definition."""
    water = {tuple(c) for c in seg["water_coords"]}
    food = {tuple(c) for c in seg["food_coords"]}
    wf = fw = 0
    last = None
    for p in (tuple(c) for c in res["coordinates_T"][a:b]):
        if p in water:
            if last == "food":
                fw += 1
            last = "water"
        elif p in food:
            if last == "water":
                wf += 1
            last = "food"
    return wf, fw


def pick_window(res, frames: int, require_eat: bool = True):
    """
    Single-panel: the prettiest death-free life containing a drink and an eat, anchored so
    the water->food commute is on screen. Explicitly a cherry-pick — that is the point.
    """
    df = np.asarray(res["drink_frac_T"])
    ef = np.asarray(res["eat_frac_T"])

    best = None
    for t0, t1, seg in eval_segments(res):
        if t1 - t0 < 80:
            continue
        eats = np.flatnonzero(ef[t0:t1] > 0)
        drinks = np.flatnonzero(df[t0:t1] > 0)
        if require_eat and (len(eats) == 0 or len(drinks) == 0):
            continue
        anchor = t0 + int(eats[0]) if len(eats) else t0
        a = max(t0, anchor - frames // 3)
        b = min(t1, a + frames)
        a = max(t0, b - frames)
        # completed crossings first: a window where the agent actually commutes reads far
        # better than one where it drinks a lot in one place
        wf, fw = count_trips(res, a, b, seg)
        score = (wf + fw, len(eats), b - a)
        if best is None or score > best[0]:
            best = (score, a, b, seg)
    return None if best is None else (best[1], best[2], best[3])


def paired_window(results, frames: int):
    """Shared window across stacks: the first eval life, truncated to the shortest."""
    firsts = [first_eval_segment(r) for r in results]
    a = firsts[0][0]
    b = min(min(t1 for _, t1, _ in firsts), a + frames)
    return [(a, b, seg) for _, _, seg in firsts]


def action_name(code: int) -> str:
    if code == 0:
        return "GO_WATER"
    if code == 1:
        return "GO_FOOD"
    if code == 8:
        return "CONSUME"
    if 2 <= code <= 7:
        return "EXPLORE"
    return "—"


# --- drawing --------------------------------------------------------------------

def build_panel(ax, P, res, a, b, seg, smell_radius, crop, crop_margin, label):
    water = [tuple(c) for c in seg["water_coords"]]
    food = [tuple(c) for c in seg["food_coords"]]
    traj = np.array([list(c) for c in res["coordinates_T"][a:b]], dtype=float)
    cells = disc(20)

    # Every cell is drawn; the view box does the cropping. Filtering the patch list was
    # what sheared the left edge off the first render.
    patches, colours = [], []
    for c in cells:
        x, y = to_xy(*c)
        patches.append(RegularPolygon((x, y), numVertices=6, radius=1.0, orientation=0))
        if c in water:
            colours.append(P["water"])
            continue
        if c in food:
            colours.append(P["food"])
            continue
        col = P["base"]
        dw = min((hex_dist(c, w) for w in water), default=99)
        df = min((hex_dist(c, f) for f in food), default=99)
        if dw <= smell_radius:
            col = lerp(col, P["scent_w"], scent_weight(dw, smell_radius))
        if df <= smell_radius:
            col = lerp(col, P["scent_f"], scent_weight(df, smell_radius))
        colours.append(col)

    ax.add_collection(PatchCollection(patches, facecolors=colours,
                                      edgecolors=P["edge"], linewidths=0.25))

    if crop:
        focus = [to_xy(q, r) for q, r in water + food] + [to_xy(q, r) for q, r in traj]
        pad = crop_margin * SQ3
    else:
        focus = [to_xy(*c) for c in cells]
        pad = 1.5
    fx = [p[0] for p in focus]
    fy = [p[1] for p in focus]
    ax.set_xlim(min(fx) - pad, max(fx) + pad)
    ax.set_ylim(min(fy) - pad, max(fy) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("none")

    trail, = ax.plot([], [], "-", color=P["trail"], lw=1.5, alpha=0.55, zorder=3)
    agent, = ax.plot([], [], "o", ms=11, mec=P["ring"], mew=1.6, zorder=5)
    flash, = ax.plot([], [], "o", ms=26, mfc="none", mew=2.2, zorder=4)

    mono = dict(family="monospace", transform=ax.transAxes, va="top", ha="left")
    ax.text(0.02, 0.985, label, fontsize=9, color=P["dim"], **mono)
    t_txt = ax.text(0.02, 0.938, "", fontsize=9, color=P["dim"], **mono)
    a_txt = ax.text(0.02, 0.898, "", fontsize=12, fontweight="bold", **mono)

    return dict(traj=traj, trail=trail, agent=agent, flash=flash,
                t_txt=t_txt, a_txt=a_txt)


def build_bars(ax, P):
    bars = ax.barh([1, 0], [0, 0], height=0.55, color=[P["water"], P["food"]], zorder=3)
    ax.axvline(IDEAL, color=P["ideal"], lw=1.0, ls="--", zorder=2)
    ax.axvline(DEATH_THRESH, color=P["death"], lw=1.4, zorder=2)
    ax.set_xlim(0, HMAX)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["hydration", "satiation"], fontsize=8.5)
    ax.set_xticks([0, IDEAL, 2, 3])
    ax.tick_params(labelsize=7, colors=P["dim"])
    ax.set_facecolor("none")

    for tick, col in zip(ax.get_yticklabels(), (P["water"], P["food"])):
        tick.set_color(col)
        tick.set_fontweight("bold")

    ax.spines["bottom"].set_color(P["edge"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    vals = [ax.text(0, y, "", va="center", ha="left", fontsize=8.5, family="monospace",
                    fontweight="bold", color=c, zorder=4)
            for y, c in ((1, P["water"]), (0, P["food"]))]
    return bars, vals


def render(panels, out_path: Path, P, fps, smell_radius, crop, crop_margin):
    k = len(panels)
    fig = plt.figure(figsize=(6.2 * k, 6.8))
    fig.patch.set_facecolor(P["fig"])
    gs = fig.add_gridspec(2, k, height_ratios=[6, 1], hspace=0.08, wspace=0.05)

    drawn = []
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        d = build_panel(ax, P, p["res"], p["a"], p["b"], p["seg"],
                        smell_radius, crop, crop_margin, p["label"])
        bax = fig.add_subplot(gs[1, i])
        d["bars"], d["vals"] = build_bars(bax, P)
        d.update(res=p["res"], a=p["a"])
        drawn.append(d)

    n_frames = min(p["b"] - p["a"] for p in panels)

    def update(i):
        for d in drawn:
            res, a = d["res"], d["a"]
            t = a + i
            traj = d["traj"]

            lo = max(0, i - 28)
            d["trail"].set_data(*zip(*[to_xy(q, r) for q, r in traj[lo:i + 1]]))

            name = action_name(int(res["abstract_action_T"][t]))
            x, y = to_xy(*traj[i])
            dead = bool(res["death_T"][t])

            d["agent"].set_data([x], [y])
            d["agent"].set_color(P["death"] if dead else action_colour(name, P))

            drank = float(res["drink_frac_T"][t]) > 0
            ate = float(res["eat_frac_T"][t]) > 0
            if drank or ate:
                d["flash"].set_data([x], [y])
                d["flash"].set_color(P["water"] if drank else P["food"])
            else:
                d["flash"].set_data([], [])

            d["t_txt"].set_text(f"t ={i:>4}")
            d["a_txt"].set_text("DEAD" if dead else name)
            d["a_txt"].set_color(P["death"] if dead else action_colour(name, P))

            h = float(res["hydration_T"][t])
            s = float(res["satiation_T"][t])
            d["bars"][0].set_width(h)
            d["bars"][1].set_width(s)
            d["vals"][0].set_position((h + 0.05, 1))
            d["vals"][0].set_text(f"{h:.2f}")
            d["vals"][1].set_position((s + 0.05, 0))
            d["vals"][1].set_text(f"{s:.2f}")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps),
              savefig_kwargs=dict(facecolor=P["fig"]))
    plt.close(fig)
    print(f"wrote {out_path}  ({n_frames} frames @ {fps}fps)")


# --- seed scan ------------------------------------------------------------------

def scan(n_seeds: int, frames: int, stack_names: list[str]):
    """
    Rank candidate seeds by COMPLETED CROSSINGS in the animated window.

    Drink and eat counts reward standing on a resource, which is exactly the behaviour the
    figure is not trying to show. A commute is the thing worth watching, so w->f + f->w
    inside the window is the sort key. With two panels the ranking follows the RIGHT one —
    the stack being showcased — and the left is printed so the contrast is visible.
    """
    paired = len(stack_names) > 1
    head = f"{'seed':>5}  {'window':>7}"
    for nm in stack_names:
        head += f"  | {nm[:16]:^16} {'w->f':>5} {'f->w':>5} {'eat':>4}"
    print(head)
    print("-" * len(head))

    rows = []
    for seed in range(n_seeds):
        try:
            results = [run(seed, **STACKS[nm][1]) for nm in stack_names]
        except FileNotFoundError as e:
            raise SystemExit(f"missing weights: {e}")

        if paired:
            wins = paired_window(results, frames)
        else:
            w = pick_window(results[0], frames) or pick_window(results[0], frames, False)
            if w is None:
                print(f"{seed:>5}  {'--':>7}")
                continue
            wins = [w]

        n = wins[0][1] - wins[0][0]
        line = f"{seed:>5}  {n:>7}"
        trips = []
        for res, (a, b, seg) in zip(results, wins):
            wf, fw = count_trips(res, a, b, seg)
            ne = int((np.asarray(res["eat_frac_T"])[a:b] > 0).sum())
            line += f"  | {'':^16} {wf:>5} {fw:>5} {ne:>4}"
            trips.append(wf + fw)
        print(line)
        # showcase panel is the last one; window length breaks ties
        rows.append((trips[-1], min(n, frames), seed))

    if rows:
        rows.sort(reverse=True)
        best = rows[0]
        print(f"\nmost crossings: --seed {best[2]}  ({best[0]} trips over {best[1]} ticks)")
        if paired:
            print("(paired runs are capped by the shorter first eval life — a short window "
                  "means one side died early, which caps the other side's trips too)")


# --- cli ------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--frames", type=int, default=360, help="ticks to animate (1 tick = 1 move)")
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--scan", type=int, metavar="N",
                   help="run N seeds, report which make a good demo, exit")
    p.add_argument("--left", default="oracle", choices=sorted(STACKS),
                   help="left panel stack (default: oracle)")
    p.add_argument("--vs", dest="right", default=None, choices=sorted(STACKS),
                   help="right panel stack; omit for a single panel")
    p.add_argument("--light", action="store_true", help="light theme (default is dark)")
    p.add_argument("--crop", action="store_true",
                   help="tighten the view onto the resources and the path taken")
    p.add_argument("--crop-margin", type=int, default=4)
    p.add_argument("--smell-radius", type=int, default=3)
    a = p.parse_args()

    names = [a.left] + ([a.right] if a.right else [])

    if a.scan:
        scan(a.scan, a.frames, names)
        return

    P = LIGHT if a.light else DARK

    results = []
    for nm in names:
        try:
            results.append(run(a.seed, **STACKS[nm][1]))
        except FileNotFoundError as e:
            raise SystemExit(
                f"stack '{nm}' needs weights that aren't on disk.\n{e}\n"
                f"The .pt blobs are gitignored — see results/weights/MANIFEST.json for "
                f"the trainer and config that produced each tag."
            )

    if len(results) > 1:
        wins = paired_window(results, a.frames)
        n = wins[0][1] - wins[0][0]
        if n < 40:
            print(f"warning: shared window is only {n} ticks — one stack died almost "
                  f"immediately on seed {a.seed}. Try --scan to find a longer one.")
        w0 = {tuple(c) for c in wins[0][2]["water_coords"]}
        w1 = {tuple(c) for c in wins[1][2]["water_coords"]}
        if w0 != w1:
            print("warning: panels are on DIFFERENT maps — do not present as a "
                  "like-for-like comparison.")
    else:
        w = pick_window(results[0], a.frames) or pick_window(results[0], a.frames, False)
        if w is None:
            raise SystemExit(f"no usable window on seed {a.seed} — try --scan 16")
        wins = [w]

    panels = [dict(label=STACKS[nm][0], res=res, a=w[0], b=w[1], seg=w[2])
              for nm, res, w in zip(names, results, wins)]

    render(panels, a.out, P, a.fps, a.smell_radius, a.crop, a.crop_margin)


if __name__ == "__main__":
    main()
