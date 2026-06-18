import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from pathlib import Path


# -----------------------------
# save / labels / run picking
# -----------------------------

def _save_fig(fig, name):
    cwd = Path.cwd()
    root = next((p for p in [cwd, *cwd.parents] if (p / ".git").exists()), cwd)
    out_dir = root / "results" / "best_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")


def get_run_env(run):
    env_kwargs = run.get("env_kwargs", {})
    radius = env_kwargs.get("radius", 1)

    water_coord = env_kwargs.get("water_coord", None)
    food_coord = env_kwargs.get("food_coord", None)

    if water_coord is None:
        water_coord = (-radius, 0)

    if food_coord is None:
        food_coord = (0, radius)

    return int(radius), tuple(water_coord), tuple(food_coord)


def run_label(run, name="run"):
    agent = run.get("agent_kwargs", {})
    sim = run.get("sim_kwargs", {})
    radius, water_coord, food_coord = get_run_env(run)

    return (
        f"{name}\n"
        f"comfort={run.get('mean_comfort', np.nan):.3f}, "
        f"deaths={run.get('death_count_eval', np.nan)}\n"
        f"len={sim.get('sim_len')}, r={radius}, "
        f"water={water_coord}, food={food_coord}\n"
        f"γ={agent.get('gamma')}, ε_0={agent.get('epsilon_start')}, "
        f"b={agent.get('batch_size')}, buf={agent.get('replay_archive_len')}, "
        f"wu={agent.get('replay_warmup')}, u={agent.get('update_ticks')}"
    )


def pick_run(config_name, pick="upper_median", res_obj=None):
    if res_obj is None:
        res_obj = res

    ranked = res_obj[config_name]["ranked"]

    if pick in ["worst", "min"]:
        return ranked[0]

    if pick in ["best", "max"]:
        return ranked[-1]

    if pick in ["median", "upper_median"]:
        return ranked[len(ranked) // 2]

    if pick == "lower_median":
        return ranked[(len(ranked) - 1) // 2]

    if isinstance(pick, int):
        return ranked[pick]

    raise ValueError(f"Unknown pick: {pick}")


def comparison(config_names=None, pick="upper_median", res_obj=None):
    if res_obj is None:
        res_obj = res

    if config_names is None:
        config_names = list(res_obj.keys())

    return [(name, pick_run(name, pick=pick, res_obj=res_obj)) for name in config_names]


# -----------------------------
# eval slicing helpers
# -----------------------------

def eb_of(run):
    return int(run.get("eval_boundary", len(run["comfort_train"])))


def eval_slice(run, key_T):
    eb = eb_of(run)
    return np.asarray(run[key_T])[eb:]


def train_slice(run, key_T):
    eb = eb_of(run)
    return np.asarray(run[key_T])[:eb]


# -----------------------------
# core diagnostic plots
# -----------------------------
def plot_phase_heatmap(
    run,
    name="run",
    bins=70,
    hs_max=3,
    save=False,
    ax=None,
    ideal_h=1.0,
    ideal_s=1.0,
    lam_over=0.3,
    k=3.0,
):
    h = np.asarray(run["hydration_eval"])
    s = np.asarray(run["satiation_eval"])

    # -----------------------------
    # comfort surface background
    # -----------------------------
    n_surface = 320
    h_vals = np.linspace(0, hs_max, n_surface)
    s_vals = np.linspace(0, hs_max, n_surface)
    HH, SS = np.meshgrid(h_vals, s_vals)

    dh = HH - ideal_h
    ds = SS - ideal_s

    h_over = np.maximum(0, dh)
    h_under = np.minimum(0, dh)

    s_over = np.maximum(0, ds)
    s_under = np.minimum(0, ds)

    d2 = h_under**2 + lam_over * h_over**2 + s_under**2 + lam_over * s_over**2
    Z = 2 * np.exp(-k * d2) - 1
    Z = np.clip(Z, -1, 1)

    # -----------------------------
    # phase density overlay
    # -----------------------------
    D, xe, ye = np.histogram2d(
        h,
        s,
        bins=bins,
        range=[[0, hs_max], [0, hs_max]],
    )

    D_masked = np.ma.masked_where(D <= 0, D)

    own = ax is None
    fig = plt.subplots(figsize=(6.8, 6.2))[1].figure if own else ax.figure
    if own:
        ax = fig.axes[0]

    fig.patch.set_facecolor("#f2f2f2")
    ax.set_facecolor("#d9d9d9")

    # Comfort surface underneath
    surface = ax.contourf(
        HH,
        SS,
        Z,
        levels=np.linspace(-1, 1, 240),
        cmap="viridis",
        norm=mcolors.Normalize(vmin=-1, vmax=1),
        zorder=0,
    )

    # Soft contour lines from the comfort surface
    ax.contour(
        HH,
        SS,
        Z,
        levels=np.linspace(-1, 1, 11),
        colors="black",
        linewidths=0.45,
        alpha=0.32,
        zorder=1,
    )

    # Phase density on top
    density = None   
    if D.max() > 0:
        density = ax.pcolormesh(
            xe,
            ye,
            D_masked.T,
            cmap="inferno",
            norm=mcolors.LogNorm(vmin=1, vmax=max(1, D.max())),
            alpha=0.68,
            shading="auto",
            zorder=3,
        )

    # Ideal point
    ax.scatter(
        [ideal_h],
        [ideal_s],
        c="white",
        s=60,
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="ideal",
    )

    ax.axvline(ideal_h, ls="--", c="white", lw=0.9, alpha=0.75, zorder=4)
    ax.axhline(ideal_s, ls="--", c="white", lw=0.9, alpha=0.75, zorder=4)

    ring = plt.Circle(
        (ideal_h, ideal_s),
        0.13,
        fill=False,
        color="white",
        linewidth=1.3,
        alpha=0.95,
        zorder=5,
    )
    ax.add_patch(ring)

    ax.set_xlim(0, hs_max)
    ax.set_ylim(0, hs_max)
    ax.set_xlabel("hydration")
    ax.set_ylabel("satiation")
    ax.set_title(f"eval phase density over comfort surface\n{run_label(run, name)}", fontsize=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")

    if own:
        cbar = fig.colorbar(surface, ax=ax, shrink=0.82, pad=0.03)
        cbar.set_label("comfort surface")
        cbar.set_ticks(np.linspace(-1, 1, 9))
        plt.tight_layout()
        if save:
            _save_fig(fig, f"phase_density_surface_{name}")
        plt.show()

    return surface, density   # ← density may be None if D.max()==0

def plot_death_gaps(run, name="run", save=False):
    eb = eb_of(run)
    death_T = np.asarray(run["death_T"]).astype(bool)
    dt = np.flatnonzero(death_T)

    train_deaths = dt[dt < eb]
    eval_deaths = dt[dt >= eb] - eb

    gaps_train = np.diff(train_deaths)
    gaps_eval = np.diff(eval_deaths)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    panels = [
        (axes[0], gaps_train, "train ε-greedy", len(train_deaths), train_deaths),
        (axes[1], gaps_eval, "eval greedy", len(eval_deaths), eval_deaths),
    ]

    for ax, gaps, label, n_d, death_idx in panels:
        if n_d == 0:
            ax.text(0.5, 0.5, "no deaths", ha="center", va="center", transform=ax.transAxes)
        elif n_d < 5:
            ax.text(
                0.5,
                0.5,
                f"{n_d} deaths\nindices: {death_idx[:10]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            ax.hist(gaps, bins=40, color="crimson", edgecolor="black")
            ax.axvline(
                np.median(gaps),
                ls="--",
                c="black",
                label=f"median gap {np.median(gaps):.0f}",
            )
            ax.legend(fontsize=8)

        ax.set_title(f"{label} — {n_d} deaths\n{run_label(run, name)}", fontsize=9)
        ax.set_xlabel("ticks survived between deaths")
        ax.set_ylabel("frequency")

    plt.tight_layout()

    if save:
        _save_fig(fig, f"death_gaps_{name}")

    plt.show()


def plot_rolling_comfort_deaths(run, name="run", w=500, save=False):
    label = run_label(run, name)
    eb = eb_of(run)
    eb_roll = max(0, eb - w + 1)

    comfort_T = np.asarray(run["comfort_T"], dtype=float)
    death_T = np.asarray(run["death_T"]).astype(float)

    roll_c = np.convolve(comfort_T, np.ones(w) / w, mode="valid")
    roll_d = np.convolve(death_T, np.ones(w) / w, mode="valid")

    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_c.plot(roll_c, color="green", lw=0.7)
    ax_c.axvline(eb_roll, ls=":", c="gray")
    ax_c.axhline(0, ls="-", c="black", lw=0.5, alpha=0.5)
    ax_c.set_title(f"rolling comfort, w={w}\n{label}", fontsize=9)
    ax_c.set_xlabel("tick")
    ax_c.set_ylabel("rolling comfort")

    ax_d.plot(roll_d, color="crimson", lw=1.2)
    ax_d.axvline(eb_roll, ls=":", c="gray")
    ax_d.set_title(f"rolling death rate, w={w}\n{label}", fontsize=9)
    ax_d.set_xlabel("tick")
    ax_d.set_ylabel("death rate")

    plt.tight_layout()

    if save:
        _save_fig(fig, f"comfort_deaths_{name}")

    plt.show()


def plot_eval_hs(run, name="run", save=False):
    h = np.asarray(run["hydration_eval"])
    s = np.asarray(run["satiation_eval"])
    te = np.arange(len(h))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(te, h, lw=0.5, alpha=0.7, label="hydration", color="tab:blue")
    ax.plot(te, s, lw=0.5, alpha=0.7, label="satiation", color="tab:orange")
    ax.axhline(1, ls="--", c="pink")

    ax.set_title(f"eval hydration & satiation\n{run_label(run, name)}", fontsize=9)
    ax.set_xlabel("eval tick")
    ax.set_ylabel("h & s")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()

    if save:
        _save_fig(fig, f"eval_hs_{name}")

    plt.show()


# -----------------------------
# hex plotting
# -----------------------------

def axial_to_xy(q, r, size=1.0):
    x = np.sqrt(3) * size * (q + r / 2)
    y = 1.5 * size * r
    return x, y


def hex_vertices(x, y, size=1.0, scale=1.0):
    angles = np.radians([90, 150, 210, 270, 330, 30])
    return np.column_stack([
        x + scale * size * np.cos(angles),
        y + scale * size * np.sin(angles),
    ])


def make_hex_coords(radius):
    return [
        (q, r)
        for q in range(-radius, radius + 1)
        for r in range(-radius, radius + 1)
        if max(abs(q), abs(r), abs(-q - r)) <= radius
    ]

# ---- hex eval occupancy, side by side, shared log norm ----

def _draw_hex_panel(ax, cw, all_coords, xs, ys, norm, cmap, water_coord, food_coord, title):
    if len(cw) > 0:
        uniq, counts = np.unique(cw, axis=0, return_counts=True)
        lookup = {tuple(c): int(n) for c, n in zip(uniq, counts)}
    else:
        lookup = {}
    for q, r in all_coords:
        x, y = axial_to_xy(q, r)
        count = lookup.get((q, r), 0)
        face = cmap(0.0) if count == 0 else cmap(norm(count))
        ax.add_patch(Polygon(hex_vertices(x, y, scale=1.003), closed=True,
                             facecolor=face, edgecolor=face, linewidth=0.0, antialiased=False))
    xw, yw = axial_to_xy(*water_coord)
    ax.add_patch(Polygon(hex_vertices(xw, yw, scale=1.02), closed=True,
                         facecolor="none", edgecolor="deepskyblue", linewidth=3.0))
    xf, yf = axial_to_xy(*food_coord)
    ax.add_patch(Polygon(hex_vertices(xf, yf, scale=1.02), closed=True,
                         facecolor="none", edgecolor="darkorange", linewidth=3.0))
    ax.set_xlim(min(xs) - 1.4, max(xs) + 1.4)
    ax.set_ylim(min(ys) - 1.4, max(ys) + 1.4)
    ax.set_title(title, fontsize=12); ax.set_aspect("equal"); ax.axis("off")


def plot_hex_eval_compare(runs, eval_window=None, save=False):
    cmap = plt.cm.magma
    panels, vmax_count = [], 1
    for nm, run in runs:
        coords_T = np.asarray(run["coordinates_T"], dtype=int)
        eb = eb_of(run)
        ev = coords_T[eb:] if eval_window is None else coords_T[eb:eb + eval_window]
        radius, water_coord, food_coord = get_run_env(run)
        all_coords = make_hex_coords(radius)
        xs, ys = zip(*[axial_to_xy(q, r) for q, r in all_coords])
        if len(ev) > 0:
            _, counts = np.unique(ev, axis=0, return_counts=True)
            vmax_count = max(vmax_count, int(counts.max()))
        panels.append((nm, ev, all_coords, xs, ys, water_coord, food_coord))

    norm = mcolors.LogNorm(vmin=1, vmax=max(1, vmax_count))
    n = len(panels)
    fig = plt.figure(figsize=(6 * n + 0.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.045], wspace=0.05)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax = fig.add_subplot(gs[0, n])

    for ax, (nm, ev, all_coords, xs, ys, wc, fc) in zip(axes, panels):
        _draw_hex_panel(ax, ev, all_coords, xs, ys, norm, cmap, wc, fc, f"eval — {nm}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label("eval ticks in tile (log, shared)")
    fig.suptitle("eval hex occupancy — " + " vs ".join(nm for nm, _ in runs) +
                 "\nblue outline = water, orange outline = food", fontsize=13)
    if save:
        _save_fig(fig, "cmp_hex_eval_" + "_vs_".join(nm for nm, _ in runs))
    plt.show()


# ---- eval death gaps, side by side, shared bins ----

def _eval_death_gaps(run):
    eb = eb_of(run)
    dt = np.flatnonzero(np.asarray(run["death_T"]).astype(bool))
    eval_deaths = dt[dt >= eb] - eb
    return np.diff(eval_deaths), len(eval_deaths), eval_deaths


def _draw_gaps_panel(ax, gaps, n_d, idx, title, bins):
    if n_d == 0:
        ax.text(0.5, 0.5, "no deaths", ha="center", va="center", transform=ax.transAxes)
    elif n_d < 5:
        ax.text(0.5, 0.5, f"{n_d} deaths\nindices: {idx[:10]}",
                ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(gaps, bins=bins if bins is not None else 40, color="crimson", edgecolor="black")
        ax.axvline(np.median(gaps), ls="--", c="black", label=f"median gap {np.median(gaps):.0f}")
        ax.legend(fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ticks survived between deaths"); ax.set_ylabel("frequency")


def plot_death_gaps_eval_compare(runs, shared_bins=True, save=False):
    data = [(nm, *_eval_death_gaps(run)) for nm, run in runs]
    bins = None
    if shared_bins:
        big = [g for _, g, n_d, _ in data if n_d >= 5]
        if big:
            allg = np.concatenate(big)
            bins = np.linspace(allg.min(), allg.max(), 41)

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5),
                             sharex=shared_bins, sharey=shared_bins)
    axes = np.atleast_1d(axes)
    for ax, (nm, gaps, n_d, idx) in zip(axes, data):
        _draw_gaps_panel(ax, gaps, n_d, idx, f"eval greedy — {nm} ({n_d} deaths)", bins)
    plt.tight_layout()
    if save:
        _save_fig(fig, "cmp_death_gaps_eval_" + "_vs_".join(nm for nm, *_ in data))
    plt.show()

def plot_hex_occupancy_log(
    run,
    name="run",
    radius=None,
    water_coord=None,
    food_coord=None,
    train_window=10_000,
    eval_window=None,
    save=False,
):
    coords_T = np.asarray(run["coordinates_T"], dtype=int)
    eb = eb_of(run)

    if radius is None or water_coord is None or food_coord is None:
        r0, w0, f0 = get_run_env(run)

        if radius is None:
            radius = r0
        if water_coord is None:
            water_coord = w0
        if food_coord is None:
            food_coord = f0

    water_coord = tuple(water_coord)
    food_coord = tuple(food_coord)

    train_coords = coords_T[:eb]
    eval_coords = coords_T[eb:] if eval_window is None else coords_T[eb:eb + eval_window]

    windows = [
        ("first train", train_coords[:train_window]),
        ("last train", train_coords[max(0, eb - train_window):eb]),
        ("eval", eval_coords),
    ]

    all_coords = make_hex_coords(radius)

    vmax_count = 1
    for _, cw in windows:
        if len(cw) == 0:
            continue
        _, counts = np.unique(cw, axis=0, return_counts=True)
        vmax_count = max(vmax_count, int(counts.max()))

    cmap = plt.cm.magma
    norm = mcolors.LogNorm(vmin=1, vmax=max(1, vmax_count))

    fig = plt.figure(figsize=(18, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.05)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    xs, ys = zip(*[axial_to_xy(q, r) for q, r in all_coords])

    for ax, (label, cw) in zip(axes, windows):
        if len(cw) > 0:
            uniq, counts = np.unique(cw, axis=0, return_counts=True)
            count_lookup = {tuple(c): int(n) for c, n in zip(uniq, counts)}
        else:
            count_lookup = {}

        for q, r in all_coords:
            x, y = axial_to_xy(q, r)
            count = count_lookup.get((q, r), 0)

            face = cmap(0.0) if count == 0 else cmap(norm(count))

            ax.add_patch(
                Polygon(
                    hex_vertices(x, y, scale=1.003),
                    closed=True,
                    facecolor=face,
                    edgecolor=face,
                    linewidth=0.0,
                    antialiased=False,
                )
            )

        # resource outlines
        xw, yw = axial_to_xy(*water_coord)
        ax.add_patch(
            Polygon(
                hex_vertices(xw, yw, scale=1.02),
                closed=True,
                facecolor="none",
                edgecolor="deepskyblue",
                linewidth=3.0,
            )
        )

        xf, yf = axial_to_xy(*food_coord)
        ax.add_patch(
            Polygon(
                hex_vertices(xf, yf, scale=1.02),
                closed=True,
                facecolor="none",
                edgecolor="darkorange",
                linewidth=3.0,
            )
        )

        ax.set_xlim(min(xs) - 1.4, max(xs) + 1.4)
        ax.set_ylim(min(ys) - 1.4, max(ys) + 1.4)
        ax.set_title(label, fontsize=12)
        ax.set_aspect("equal")
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("ticks in window (log)")

    fig.suptitle(
        f"hex occupancy over training — {name}\n"
        f"blue outline = water, orange outline = food",
        fontsize=14,
    )

    if save:
        _save_fig(fig, f"hex_occupancy_log_{name}")

    plt.show()


# -----------------------------
# route diagnostics
# -----------------------------

def plot_trip_lengths(run, name="run", max_trip_ticks=300, save=False):
    radius, water_coord, food_coord = get_run_env(run)

    coords_eval = eval_slice(run, "coordinates_T").astype(int)
    death_eval = eval_slice(run, "death_T").astype(bool)

    trips = extract_water_to_food_trips(
        coords_eval=coords_eval,
        death_eval=death_eval,
        water_coord=water_coord,
        food_coord=food_coord,
        max_trip_ticks=max_trip_ticks,
    )

    success = [tr for tr in trips if tr["success"]]
    success_moves = np.array([tr["moves"] for tr in success], dtype=float)
    shortest = hex_dist(water_coord, food_coord)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    if len(success_moves) == 0:
        ax.text(0.5, 0.5, "no successful water → food trips", ha="center", va="center", transform=ax.transAxes)
    else:
        bins = np.arange(success_moves.min(), success_moves.max() + 2) - 0.5
        ax.hist(success_moves, bins=bins, edgecolor="black")
        ax.axvline(shortest, ls="--", c="black", label=f"shortest = {shortest}")
        ax.axvline(np.median(success_moves), ls=":", c="gray", label=f"median = {np.median(success_moves):.1f}")
        ax.legend(fontsize=8)

    ax.set_title(
        f"successful water → food trip lengths\n"
        f"{run_label(run, name)}\n"
        f"trips={len(trips)}, success={len(success)}",
        fontsize=9,
    )
    ax.set_xlabel("moves from water departure to food arrival")
    ax.set_ylabel("frequency")

    plt.tight_layout()

    if save:
        _save_fig(fig, f"trip_lengths_{name}")

    plt.show()


def print_run_metrics(run):
    m = run.get("metrics", {})

    keys = [
        "eval_len",
        "eval_deaths",
        "non_neighbor_jumps",
        "water_visit_pct",
        "food_visit_pct",
        "drink_rate_at_water",
        "eat_rate_at_food",
        "full_eat_rate_at_food",
        "water_to_food_trip_count",
        "water_to_food_success_count",
        "water_to_food_success_rate",
        "water_to_food_shortest_dist",
        "median_success_trip_moves",
        "path_efficiency",
        "perfectish_trip_rate",
    ]

    for k in keys:
        print(f"{k}: {m.get(k, np.nan)}")


def plot_run_diagnostics(run, name="run", w=500, save=False):
    plot_death_gaps(run, name=name, save=save)
    plot_phase_heatmap(run, name=name, save=save)
    plot_rolling_comfort_deaths(run, name=name, w=w, save=save)
    plot_eval_hs(run, name=name, save=save)
    plot_hex_occupancy_log(run, name=name, save=save)
    plot_trip_lengths(run, name=name, save=save)