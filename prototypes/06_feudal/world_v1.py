"""
world.py  —  single source of physical truth for Proto 06.

The env runner AND the oracles import this; that shared import is what stops decay
constants getting copy-pasted into an oracle and drifting. It is also the greppable
privilege boundary: an oracle is *allowed* to know the true physics (that's the legit
privilege that evaporates on swap), but **a learned module importing `world` is a leak**
— grep for it.

Ported faithfully from sim_instance (the monolith). Parameterised only where the sim
baked in `env.radius`. Nothing here is policy: fill targets, hysteresis margins, explore
thresholds all live on the oracles, not in this file.
"""

import numpy as np


# --- drives ---------------------------------------------------------------------
IDEAL_H = 1.0
IDEAL_S = 1.0
HMAX = 3.0
SMAX = 3.0
DEATH_THRESH = 0.05                     # h <= this OR s <= this  =>  dead


# --- comfort surface (tolerance band above ideal, under-ideal dear) -------------
# OVER_TOL: free buffer above ideal. Drives in [ideal, ideal+OVER_TOL] cost NOTHING,
# so survival reserve is truly untaxed -- comfort is honestly indifferent across the
# safe zone (which is what we want: safe is safe, no lean-running pull). Beyond the
# band OVER_W bites, so the graded consumer still avoids gross overfill. The only
# blemish is the kink at ideal+OVER_TOL; if that ever trips the optimizer during
# training, smooth just that corner (softplus hinge) rather than reverting to a curve
# that re-taxes the buffer. Set OVER_TOL=0 for the old asymmetric-peak quadratic.
OVER_W   = 0.02
UNDER_W  = 0.5
OVER_TOL = 1.0

def comfort_d2(h, s):
    h_over  = max(0.0, h - IDEAL_H - OVER_TOL); h_under = min(0.0, h - IDEAL_H)
    s_over  = max(0.0, s - IDEAL_S - OVER_TOL); s_under = min(0.0, s - IDEAL_S)
    return (UNDER_W * h_under**2 + OVER_W * h_over**2 +
            UNDER_W * s_under**2 + OVER_W * s_over**2)

def comfort(h, s, surface="exponential"):
    d2 = comfort_d2(h, s)
    return 2 * np.exp(-3 * d2) - 1 if surface == "exponential" else 1 - d2


# --- consume model (the eat/drink oracles invert this) --------------------------
DRINK_AMOUNT = 0.15
EAT_AMOUNT   = 0.30

# absorption profiles: drink front-loaded over 7 ticks, eat back-loaded over 10.
# both sum to 1, so per-action *total* gain is amount * fraction * (coupling).
_p1 = (11 - np.arange(1, 8)) ** 2;  DRINK_PROFILE = _p1 / _p1.sum()
_p2 = np.arange(1, 11).astype(float); EAT_PROFILE  = _p2 / _p2.sum()

def drink_gain(frac, s):
    """Total hydration a drink of `frac` eventually adds. Coupling: water uptake
    scales with satiation (0.8 + 0.3 s). s taken ~constant over the absorption window."""
    return DRINK_AMOUNT * frac * (0.8 + 0.3 * s)

def eat_gain(frac):
    return EAT_AMOUNT * frac

def drink_frac_for(h, s, target):
    """Inverse of drink_gain: the fraction that lands h at `target`, clipped [0,1].
    Deficit larger than one max drink -> returns 1.0 and tapers as h nears target
    across successive ticks (drink_amount is small, so refills take several ticks)."""
    denom = DRINK_AMOUNT * (0.8 + 0.3 * s)
    return float(np.clip((target - h) / denom, 0.0, 1.0)) if denom > 0 else 0.0

def eat_frac_for(s, target):
    return float(np.clip((target - s) / EAT_AMOUNT, 0.0, 1.0))


# --- metabolism (decay). scaling is radius-derived, fixed per run. ---------------
def decay_scaling(radius):
    eff = radius / 2 + 1
    h_scale = 0.05 + 1.45 / ((1 + 1.0426 * (eff - 1)) ** 0.7122)
    return h_scale, 0.8 * h_scale               # (hydration, satiation)

def decay_hydration(h, s, brightness, h_scale):
    d = max(0.0, 0.15 * brightness - 0.03 * s) + np.random.normal(0.05, 0.03)
    return h - d * h_scale

def decay_satiation(s, h, brightness, s_scale):
    d = max(0.0, (0.05 - 0.05 * brightness) + (0.04 - 0.04 * h) + 0.1 * (IDEAL_H - h))
    d += np.random.normal(0.01, 0.005)
    return s - d * s_scale

# deterministic variants for oracle planning (noise replaced by its mean).
def expected_decay_hydration(h, s, brightness, h_scale):
    d = max(0.0, 0.15 * brightness - 0.03 * s) + 0.05
    return h - d * h_scale

def expected_decay_satiation(s, h, brightness, s_scale):
    d = max(0.0, (0.05 - 0.05 * brightness) + (0.04 - 0.04 * h) + 0.1 * (IDEAL_H - h)) + 0.01
    return s - d * s_scale


# --- brightness (day cycle) -----------------------------------------------------
def brightness(t, day_len=50, noise=True):
    val = 0.5 + 0.3 * np.sin((2 * np.pi / day_len) * t)
    if noise:
        val += np.random.normal(0, 0.05)
    return min(1.0, max(0.0, val))


# --- respawn: the drive half only (annulus around ideal). ----------------------
# coord sampling stays in the sim — it needs the env's spawn pool.
def sample_spawn_drives():
    while True:
        ang = np.random.uniform(0, 2 * np.pi)
        rad = np.sqrt(np.random.uniform(0.1 ** 2, 2.9 ** 2))
        nh = IDEAL_H + rad * np.cos(ang)
        ns = IDEAL_S + rad * np.sin(ang)
        if (0.05 < nh < HMAX) and (0.05 < ns < SMAX) and not (nh < 0.35 and ns < 0.35):
            return float(nh), float(ns)