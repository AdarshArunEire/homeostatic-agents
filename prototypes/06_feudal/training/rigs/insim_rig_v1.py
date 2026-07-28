"""
insim_rig_v1.py  —  training rig for modules whose Obs depends on sim dynamics.

Covers consumer, orchestrator and explorer. The pathfinder does not belong here: its Obs is
`to_goal` and nothing else, so it trains standalone (pathfinder_rig_v1) and that fact is
itself the contract test.

Each module trains in an OTHERWISE-ALL-ORACLE stack, so its training distribution is not
polluted by another module's incompetence. The all-learned rung is then a pure composition
test rather than a tangle of confounded failures.

TWO DESIGN DECISIONS WORTH KNOWING BEFORE USING THIS.

1. ITERATED BATCH OFF-POLICY, not online RL.
   `sim_instance` has no hook to interleave gradient steps into its main loop, and adding
   one means surgery on a 950-line function that currently works. So the loop is: run a
   rollout with frozen weights -> harvest transitions -> train -> repeat. This is DQN with
   a very large collect-to-update ratio. It is legitimate but sample-inefficient in
   wall-clock terms, and the policy is stale within a rollout by construction. Budget
   rollouts accordingly.

2. ZERO EDITS TO THE SIM LOOP.
   The module records `(obs, action)` in call order into a buffer it owns. Rewards are
   computed AFTERWARDS by joining those records onto the timeseries the sim already
   returns (comfort_T, hydration_T, satiation_T, death_T). Nothing inside the tick loop
   changes.

   The join is the dangerous part. Records carry no tick index -- the module cannot see the
   clock -- so alignment is by construction: the i-th call corresponds to the i-th tick on
   which the module was dispatched. `align_records` recovers those ticks from
   `abstract_action_T` and ASSERTS the counts match. A silent off-by-one here would train
   every transition against the wrong tick's reward and produce plausible-looking garbage
   with no error, so the assertion is load-bearing, not defensive decoration.
"""

from __future__ import annotations

import numpy as np

from model_modules.contract_v1 import Action
from model_modules.learned_modules.recorder_v1 import Recorder  # noqa: F401  (re-export)

# which abstract action indicates each module was dispatched this tick
DISPATCH_ACTION = {
    "eat": (int(Action.CONSUME),),
    "drink": (int(Action.CONSUME),),
    "orchestrator": None,   # called every tick
    "explorer": tuple(int(a) for a in range(Action.EXPLORE_0, Action.EXPLORE_5 + 1)),
}


def align_records(run: dict, records: list, slot: str, eval_only: bool = False):
    """
    Recover the tick index for each recorded call.

    Returns an int array `ticks` with len(ticks) == len(records).

    Raises on mismatch rather than truncating. A quiet truncation is the failure mode that
    produces a trained module which is subtly wrong everywhere.
    """
    abstract = np.asarray(run["abstract_action_T"])
    lo = int(run["eval_boundary"]) if eval_only else 0
    hi = len(abstract)

    if slot in ("eat", "drink"):
        # Exact split via consumer_slot_T (0 none / 1 drink / 2 eat), written by the
        # composer at dispatch time. Inferring this from the fracs does not work: both a
        # learned consumer and the oracle can return 0.0 when the drive is already at
        # target, which leaves a zero-frac CONSUME tick unattributable between the slots.
        if "consumer_slot_T" not in run:
            raise KeyError(
                "run has no 'consumer_slot_T' -- this sim_instance predates the dispatch "
                "record. Re-run against the current sim_instance_v3; do not fall back to "
                "inferring the slot from fracs."
            )
        want = 2 if slot == "eat" else 1
        cs = np.asarray(run["consumer_slot_T"])[lo:hi]
        ticks = np.flatnonzero(cs == want) + lo
    else:
        wanted = DISPATCH_ACTION[slot]
        if wanted is None:
            ticks = np.arange(lo, hi, dtype=int)
        else:
            ticks = np.flatnonzero(np.isin(abstract[lo:hi], wanted)) + lo

    if len(ticks) != len(records):
        raise AssertionError(
            f"record/tick misalignment for slot '{slot}': {len(records)} records vs "
            f"{len(ticks)} dispatch ticks. Do NOT train on this -- every transition would "
            f"be joined to the wrong tick. Likely causes: the module was constructed twice "
            f"(check build_modules), recording was not reset between rollouts, or the "
            f"dispatch mask for this slot is wrong."
        )
    return ticks


def life_bounds(run: dict) -> list[tuple[int, int]]:
    """
    (start, end_exclusive) per life. A life ends on a death tick; the last one runs to the
    end of the sim and is a truncation, not a terminal -- callers should bootstrap through
    it rather than treating it as death.
    """
    dead = np.asarray(run["death_T"]).astype(bool)
    ends = list(np.flatnonzero(dead) + 1)
    bounds, start = [], 0
    for e in ends:
        bounds.append((start, int(e)))
        start = int(e)
    if start < len(dead):
        bounds.append((start, len(dead)))
    return bounds


def build_transitions(
    run: dict,
    records: list,
    ticks: np.ndarray,
    reward_fn,
    *,
    n_act: int,
    gamma: float = 0.99,
    n_step: int = 1,
    terminal_fn=None,
):
    """
    Join records to rewards and roll into the 7-tuple format rl_core expects.

    `reward_fn(run, tick, next_tick)` returns the scalar reward for the call made at
    `tick`, given the sim's timeseries. Keeping it a callback is what lets one rig serve
    the consumer (comfort over the absorption window), the orchestrator (survival) and the
    explorer (discovery) without three copies of this function.

    Successive DISPATCHES are the transition boundary, not successive ticks: from the
    module's point of view nothing happened in between, because it was not consulted.

    `terminal_fn(run, t, t_next) -> bool` marks a transition as TERMINAL beyond death, and
    for the explorer it is load-bearing rather than optional.

    Without it the explorer's discovery transition bootstraps into garbage. The module is
    called only on EXPLORE_*, so after it finds water the orchestrator runs GO_WATER ->
    CONSUME and only then explores again: `next_state` for the +10 transition is recorded
    after a full travel-and-drink sequence, in a different region, causally unrelated to the
    action being credited. Those are precisely the transitions forced into 25% of every
    batch, so the highest-reward samples carried the most corrupted targets — which is what
    an anti-informative Q-function looks like.

    Discovery ends the explorer's task. Marking it terminal makes +10 a clean terminal
    reward with no bootstrap, which is what it always should have been.
    """
    from training.rl_core_v1 import NStepAccumulator

    dead = np.asarray(run["death_T"]).astype(bool)
    lives = life_bounds(run)
    acc = NStepAccumulator(n_step=n_step, gamma=gamma)
    mask = np.ones(n_act, dtype=bool)  # slots with real legality masks override this
    out = []

    for lo, hi in lives:
        idx = np.flatnonzero((ticks >= lo) & (ticks < hi))
        if len(idx) < 2:
            continue
        acc.clear()

        for j in range(len(idx) - 1):
            i, i_next = int(idx[j]), int(idx[j + 1])
            t, t_next = int(ticks[i]), int(ticks[i_next])
            obs, act = records[i]
            obs_next, _ = records[i_next]
            r = float(reward_fn(run, t, t_next))
            died = bool(dead[t_next - 1]) if t_next > 0 else False
            done = died or (terminal_fn(run, t, t_next) if terminal_fn else False)
            out.extend(acc.push(obs, act, r, obs_next, done, mask))

        # tail of the life: truncation if it ran to the end of the sim, terminal if it died
        last = int(idx[-1])
        t_last = int(ticks[last])
        if hi <= len(dead) and dead[hi - 1]:
            obs, act = records[last]
            r = float(reward_fn(run, t_last, hi))
            out.extend(acc.push(obs, act, r, obs, True, mask))
        else:
            out.extend(acc.flush())

    return out


# --- reward builders ------------------------------------------------------------

def consumer_reward(death_penalty: float = 20.0, tick_scale: float = 50.0,
                    comfort_weight: float = 1.0, window: int = 25, gamma: float = 0.99):
    """
    Consumer reward: TIME BOUGHT, plus a gross-overfill check, minus death.

    Replaces comfort_window_reward for this slot, which failed for a reason the comfort
    surface makes structural. OVER_TOL=1.0 means comfort is FLAT from ideal to ideal+1.0 --
    filling to 1.0, 1.6 or 2.0 scores identically. That flatness is deliberate ("safe is
    safe, no lean-running pull") and it is exactly what makes comfort useless as a local
    consumer signal: the reward cannot distinguish the decision the module is making. The
    cost of underfilling only lands far outside any absorption-length window, when decay
    eats the buffer and the agent dies. Trained on comfort, the drink head underfilled to
    meanFrac 0.64 and took 48 eval deaths against the oracle's 35.

    Three terms, each doing a job comfort cannot:

      ticks-to-next-consume   the consumer's actual causal effect. A bigger fill buys more
                              time before the next one is needed, and that IS the decision.
      comfort over `window`   catches GROSS overfill only, beyond ideal+OVER_TOL where the
                              surface stops being flat. Not the main signal, a guard rail.
      death penalty           explicit, not just the truncated bootstrap. Carried from 03b:
                              scale it against the discounted return it interrupts, or
                              underfilling stays locally cheap.

    Overfilling is bounded from both ends: past ideal+OVER_TOL comfort bites, and drives
    cap at HMAX/SMAX so extra fill is simply wasted rather than rewarded.
    """
    def fn(run, t, t_next):
        comfort = np.asarray(run["comfort_T"], dtype=float)
        dead = np.asarray(run["death_T"]).astype(bool)

        bought = (t_next - t) / tick_scale

        hi = min(len(comfort), t + window)
        seg = comfort[t:hi]
        comf = float(seg.mean()) if len(seg) else 0.0

        died = bool(dead[t:min(len(dead), t_next)].any())
        return bought + comfort_weight * comf - (death_penalty if died else 0.0)

    return fn


def comfort_window_reward(window: int = 11, gamma: float = 0.99):
    """
    SUPERSEDED for the consumer by `consumer_reward` -- kept for reference and for
    reproducing the v1 result.

    Discounted comfort over the absorption window. It correctly avoids the instantaneous-
    comfort trap (which rewards slamming 1.0), but fails for a subtler reason: the
    tolerance band makes comfort flat across the whole safe zone, so the signal is nearly
    constant in the action being chosen. Underfilling looks free locally and is punished
    only much later. See P2 in BUILDNOTES.
    """
    disc = None

    def fn(run, t, t_next):
        nonlocal disc
        comfort = np.asarray(run["comfort_T"], dtype=float)
        hi = min(len(comfort), t + window)
        seg = comfort[t:hi]
        if disc is None or len(disc) < len(seg):
            disc = gamma ** np.arange(window + 1, dtype=float)
        return float((seg * disc[: len(seg)]).sum() / max(len(seg), 1))

    return fn


def explorer_reward(discover_reward: float = 10.0, step_cost: float = 0.02,
                    death_penalty: float = 10.0):
    """
    Explorer reward: sparse, on DISCOVERY. The honest version of the hard problem.

    Detecting discovery needs no change to the sim. `GO_WATER` / `GO_FOOD` are masked
    illegal until the corresponding memory slot is filled (see Orchestrator.legal_mask), so
    the first appearance of a GO_* in `abstract_action_T` after a run of EXPLORE_* IS the
    moment a resource was found. Reading the dispatch record recovers the event exactly.

    Deliberately NOT shaped on smell. Rewarding smell increase would pay the agent for
    chemotaxis — precisely the reactive behaviour the 0.48 floor already achieves — and
    would encode the answer rather than let it be found. It would also be the
    proximity-as-reward violation the project has refused since Proto 04: scent is an
    OBSERVATION, never a reward. The whole question is whether directed coverage of the
    scentless band can be learned from the arrival signal alone.

    Consequence to expect: this reward is sparse and long-horizon, so a null result is
    ambiguous between "memory does not help" and "the signal never reached the policy".
    The training-time discovery counter is what separates those.
    """
    from model_modules.contract_v1 import Action

    GO = (int(Action.GO_WATER), int(Action.GO_FOOD))

    def fn(run, t, t_next):
        ab = np.asarray(run["abstract_action_T"])
        dead = np.asarray(run["death_T"]).astype(bool)

        hi = min(len(ab), max(t + 1, t_next + 1))
        discovered = bool(np.isin(ab[t + 1:hi], GO).any())
        died = bool(dead[t:min(len(dead), max(t + 1, t_next))].any())

        return (discover_reward if discovered else 0.0) \
            - step_cost * max(1, t_next - t) \
            - (death_penalty if died else 0.0)

    return fn


def discovery_terminal():
    """
    `terminal_fn` for the explorer: a discovery ends its task.

    Pair with `explorer_reward`, which pays out on the same event. Reward and terminality
    must agree on what "done" means, or the +10 is credited to a transition that then also
    bootstraps a value from somewhere else.
    """
    from model_modules.contract_v1 import Action

    GO = (int(Action.GO_WATER), int(Action.GO_FOOD))

    def fn(run, t, t_next):
        ab = np.asarray(run["abstract_action_T"])
        hi = min(len(ab), max(t + 1, t_next + 1))
        return bool(np.isin(ab[t + 1:hi], GO).any())

    return fn


def count_discoveries(run) -> int:
    """
    Training-time diagnostic: how many times did an EXPLORE run actually end in a discovery?

    Proto 04's Go-Explore probe used the same idea — a training crossing counter that
    distinguished "memory did not help" from "exploration never sampled the loop". If this
    stays flat at zero, the explorer is not failing to learn; it is failing to ever receive
    a signal, and that is a different diagnosis with a different fix.

    CONFOUNDED BY DEATHS — do not read the raw count as quality. Death clears memory, so a
    worse explorer rediscovers more often. Measured on the calibration ladder: random logged
    136 discoveries against smell_momentum's 97, while dying 121 times against 35. Per life
    the ordering inverts and is meaningful (1.12 vs 2.77). Use `discoveries_per_life`.
    """
    from model_modules.contract_v1 import Action

    ab = np.asarray(run["abstract_action_T"])
    is_go = np.isin(ab, (int(Action.GO_WATER), int(Action.GO_FOOD)))
    is_explore = (ab >= int(Action.EXPLORE_0)) & (ab <= int(Action.EXPLORE_5))
    # a discovery is an explore->go transition
    return int((is_explore[:-1] & is_go[1:]).sum())


def _abstract4(a):
    """Action enum -> the 4-way head's categories. EXPLORE_0..5 collapse to one option."""
    a = np.asarray(a)
    return np.where(a == 0, 0, np.where(a == 1, 1, np.where(a == 8, 2, 3)))


def orchestrator_reward(step_reward: float = 0.01, death_penalty: float = 1.0,
                        delib_cost: float = 0.0, delib_exempt_consume: bool = True):
    """
    Orchestrator reward: pure survival. +step_reward per tick alive, -death_penalty on death.

    NOT comfort, and that is a decision paid for in P2. `OVER_TOL = 1.0` makes comfort flat
    from ideal to ideal+1.0, and the orchestrator's interesting decisions — when to abandon a
    hunt, when to stop consuming — happen almost entirely INSIDE that flat region. Comfort is
    constant across exactly the choices being made, so it cannot grade them.

    Survival can. With `step_reward = 0.01` and gamma 0.99 the value of an immortal agent is
    0.01/(1-0.99) = 1.0, so `death_penalty = 1.0` is "you lose a full lifetime" — commensurate
    with the discounted return it interrupts, which is the k/(1-gamma) scaling carried since
    03b rather than an arbitrary constant.

    The orchestrator is dispatched every tick, so transitions are tick-to-tick and n-step
    returns propagate over real time.

    DELIBERATION COST (`delib_cost`, default 0.0 so nothing already trained changes).

    A penalty charged whenever the emitted abstract action differs from the previous tick's.
    This is Harb et al. 2018's lever, and it exists because P4.2/P4.3 measured the textbook
    option-collapse failure: with no termination function the module re-decides every tick,
    median GO_FOOD run length is 1.0 against the oracle's 10.0, and forcing persistence at
    inference recovers ~30% of the gap. A switching cost asks whether the module will
    ACQUIRE persistence when it is merely cheaper, rather than having it imposed.

    SIZE IT AGAINST THE SWITCH RATE, NOT THE STEP REWARD. This is the trap, and d0 fell in
    it. The per-tick cost is delib_cost x switches-per-tick, and the uncommitted policy
    switches ~0.5-0.7 times per tick (dwell ~1.4). At delib_cost=0.02 that is ~0.012/tick
    against step_reward=0.01 — a penalty LARGER than the entire survival signal. Target
    ~10% of step_reward once multiplied out, which puts the usable range near 0.001-0.005.

    CONSUME IS EXEMPT BY DEFAULT, and this is not a tuning choice. `d0` (delib 0.02, no
    exemption) collapsed to 78.9% CONSUME held for a mean of 307 ticks: never switching
    costs nothing, so parking on a resource tile is the trivial optimum. It died of thirst
    while standing on food. Transitions into and out of CONSUME are forced by which tile
    the agent occupies rather than deliberated, so charging them prices the wrong thing.
    With the exemption the cost falls exactly where the pathology is — mid-commute flips
    between GO_WATER and GO_FOOD — while a proper commute (CONSUME -> GO_FOOD -> hold ->
    CONSUME -> GO_WATER) pays almost nothing.

    A respawn is not a switch. Deaths break the comparison so a fresh life's first action
    is never charged against the previous life's last.
    """
    cache = {"id": None, "sw": None}

    def _switches(run):
        if cache["id"] == id(run):
            return cache["sw"]
        k = _abstract4(run["abstract_action_T"])
        dead = np.asarray(run["death_T"]).astype(bool)
        sw = np.zeros(len(k), dtype=bool)
        sw[1:] = (k[1:] != k[:-1]) & ~dead[:-1]
        if delib_exempt_consume:
            # CONSUME is index 2; a transition touching it is tile-forced, not deliberated
            sw[1:] &= ~((k[1:] == 2) | (k[:-1] == 2))
        cache.update(id=id(run), sw=sw)
        return sw

    def fn(run, t, t_next):
        dead = np.asarray(run["death_T"]).astype(bool)
        span = max(1, t_next - t)
        died = bool(dead[t:min(len(dead), t + span)].any())
        r = step_reward * span - (death_penalty if died else 0.0)
        if delib_cost:
            sw = _switches(run)
            r -= delib_cost * float(sw[t:min(len(sw), t + span)].sum())
        return r

    return fn


def dwell_stats(actions, deaths=None) -> dict:
    """
    How long does the module hold an intention?

    Mean and median run length of each abstract option, plus the overall switch rate.
    Takes the Action-enum timeseries (EXPLORE_0..5 collapse to one option) and optionally
    `death_T`, so a respawn does not read as a long run spanning two lives.

    This is the readout for the deliberation-cost sweep. It is only meaningful when the
    horizon is NOT enforced: under `hold=k` these values are set by construction.
    """
    k = _abstract4(actions)
    if deaths is not None:
        d = np.asarray(deaths).astype(bool)
        k = k.copy()
        k[np.flatnonzero(d)] = -1          # break runs at a death

    names = {0: "GO_WATER", 1: "GO_FOOD", 2: "CONSUME", 3: "EXPLORE"}
    runs = {v: [] for v in names.values()}
    cur, n = None, 0
    for v in k:
        if v == cur:
            n += 1
            continue
        if cur in names and n:
            runs[names[cur]].append(n)
        cur, n = int(v), 1
    if cur in names and n:
        runs[names[cur]].append(n)

    out = {}
    for name, xs in runs.items():
        out[f"dwell_mean_{name}"] = float(np.mean(xs)) if xs else 0.0
        out[f"dwell_med_{name}"] = float(np.median(xs)) if xs else 0.0
        out[f"dwell_n_{name}"] = len(xs)
    total = sum(len(x) for x in runs.values())
    out["dwell_mean_all"] = float(np.mean([n for xs in runs.values() for n in xs])) if total else 0.0
    out["switch_rate"] = total / max(1, len(k))
    return out


def survival_reward(run_key: str = "comfort_T"):
    """
    Orchestrator reward: comfort accrued between decisions, with death heavily penalised.

    Scaled by k/(1-gamma) in the trainer, matching the death-penalty scaling carried since
    03b, so the terminal cost stays commensurate with the discounted return it interrupts.
    """
    def fn(run, t, t_next):
        series = np.asarray(run[run_key], dtype=float)
        seg = series[t:max(t + 1, t_next)]
        return float(seg.mean()) if len(seg) else 0.0

    return fn


# --- rollout --------------------------------------------------------------------

def rollout(module_spec, oracle_params, *, seed: int, sim_len: int = 7000,
            eval_len: int = 5000, radius: int = 20, **sim_overrides):
    """
    One frozen-weights sim run. Returns the raw run dict.

    Defaults mirror god_vs_legal_check so numbers stay on the standing config; override
    per call rather than editing them here.
    """
    import sim_instance_v3 as S

    kwargs = dict(
        seed=seed, sim_len=sim_len, eval_len=eval_len,
        env_kwargs=dict(radius=radius, band=(9, 11), start_coord=(0, 0)),
        decay_mult=0.7, smell_radius=3,
        curriculum_mode="band", c_min=2, c_max=9, band_width=2, life_cap=1000,
        oracle_params=oracle_params, module_spec=module_spec, log_every=10**9,
        eval_spawn_nondoomed=True, spawn_leeway=10, eval_god_memory=False,
    )
    kwargs.update(sim_overrides)
    return S.sim_instance(**kwargs)


def harvest(run, module, slot: str, reward_fn, *, n_act: int,
            gamma: float = 0.99, n_step: int = 1, terminal_fn=None):
    """rollout + module records -> transitions. Raises if the join does not line up."""
    ticks = align_records(run, module.records, slot)
    return build_transitions(run, module.records, ticks, reward_fn,
                             n_act=n_act, gamma=gamma, n_step=n_step,
                             terminal_fn=terminal_fn)


class use_module:
    """
    Context manager: make the sim use THIS module instance for a slot.

    `build_modules` constructs modules from the registry, so a trainer holding a net with
    live gradients has no way to hand it in — `oracle_params` carries a string tag, which
    would load a stale copy from disk. Overriding the factory to return the existing
    instance is the seam that needs no change to build_modules or sim_instance.

    Restores on exit, including on exception: a leaked override would silently poison every
    later run in the process with a half-trained module.

        with use_module(eat=my_consumer):
            run = rollout(spec, params, seed=0)
    """

    def __init__(self, **slots):
        self.slots = slots
        self._saved = {}

    def __enter__(self):
        import sim_instance_v3 as S

        for slot, module in self.slots.items():
            if slot not in S.MODULE_REGISTRY:
                raise KeyError(f"unknown module slot {slot!r}")
            self._saved[slot] = S.MODULE_REGISTRY[slot].get("learned")
            S.MODULE_REGISTRY[slot]["learned"] = (lambda m: (lambda **kw: m))(module)
        return self

    def __exit__(self, *exc):
        import sim_instance_v3 as S

        for slot, prev in self._saved.items():
            if prev is None:
                S.MODULE_REGISTRY[slot].pop("learned", None)
            else:
                S.MODULE_REGISTRY[slot]["learned"] = prev
        self._saved.clear()
        return False
