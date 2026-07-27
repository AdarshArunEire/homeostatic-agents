# Reproducing the results

Every number in [`RESULTS.md`](RESULTS.md) is regenerable from this repository. Trained weights
are gitignored — they are binaries, and the trainers plus recorded seeds reproduce them — but
`results/weights/MANIFEST.json` is committed and records the architecture, full training config,
seed, git SHA and achieved metric for every tag. A regenerated checkpoint can therefore be checked
against its record rather than trusted.

```
pip install -r requirements.txt      # numpy, torch
cd prototypes/06_feudal              # all commands below run from here
```

Runtimes are from a laptop CPU; nothing here needs a GPU.

---

## 1. Verify the machinery before trusting any number

```
python tests/probe_infra_v1.py --verbose
```

Nine probes, ~1 min. Each guards an invariant whose violation is **silent** — the run completes,
numbers come out, and they are wrong. Expect 9/9.

The two that matter most: the **privilege boundary** probe greps `learned_modules/` for any import
of `world_v1` (a learned module reading the true physics would train beautifully and prove
nothing), and the **rig alignment** probe checks that recorded module decisions join to the ticks
they actually came from.

---

## 2. Calibrate the metrics — before training anything

```
python tests/explorer_sensitivity_v1.py --calibrate-only      # ~1 min
python tests/reactive_ceiling_v1.py                           # ~2 min
```

A null result is uninterpretable unless the metric has been shown capable of producing a non-null
one. The first sweeps a deliberately degraded explorer and reports whether the metric is monotone
in the degradation, where its detection floor sits, and its dynamic range. Expect the ladder to
reproduce: random 0.1712, momentum 0.3333, smell_momentum 0.4776.

The second asks whether 0.4776 is the reactive *optimum* or just the point that was picked. Expect
no configuration to separate from the baseline's Wilson interval.

---

## 3. Pathfinder — the module that worked

```
python training/train_pathfinder_v1.py --tag s2b --seed 2     # ~90 s
python tests/test_module_diff_v1.py --tag s2b --in-sim        # ~2 min
```

Trains out-of-sim on bare hex geometry, then drops the result into the full agent.

Expect: 1.0000 optimality on approach / commute / long bands, `excess_steps_per_move` 0.0000,
gate PASS, and an in-sim solveScore delta of **+0.000** over ~34,000 invocations. Oracle agreement
will be ~0.56 — the module is independently optimal, not a copy.

For the reproducibility claim, run seeds 0 and 1 too (`--tag s0b --seed 0`, `--tag s1b --seed 1`);
2 of 3 clear the pre-registered 0.99 gate.

---

## 4. Consumer

```
python training/train_consumer_v1.py --slot drink --tag v1    # ~2 min
python tests/consumer_sensitivity_v1.py --slot drink --tag v1 # ~4 min
```

Expect ~37 eval deaths against the oracle's 35, and — more importantly — expect the calibration to
show `solveScore` **non-monotone** under controlled degradation, with only `drink_rate_at_water`
qualifying as a detector across a 0.044 range. That is the finding: the environment cannot resolve
consumer quality.

The trainer prints an instability warning if `meanFrac` spread exceeds 0.25. It will.

---

## 5. Explorer — the module that did not work

```
python training/train_explorer_v1.py --tag e0 --seed 0        # ~5 min  (DQN)
python training/train_explorer_pg_v1.py --tag p0 --seed 0     # ~9 min  (policy gradient)
python tests/explorer_sensitivity_v1.py --tags e0             # ~2 min
python tests/explorer_sensitivity_v1.py --tags p0 --policy
python tests/explorer_stochastic_probe_v1.py --tags e0        # ~4 min
```

Expect all of them to land at or below the random floor of 0.1712.

The stochastic probe is the diagnostic worth watching: it re-evaluates *existing* weights as
`softmax(Q/T)` policies across temperatures. Performance rises monotonically toward the **uniform**
limit, meaning the best available use of the learned Q-function is to ignore it.

For the policy-gradient run, watch the `H` column (entropy, max ln 6 ≈ 1.792). At
`--entropy-beta 0.02` it stays near 1.75 — the policy never becomes selective. At `0.003` it
collapses to 0 and loops. There is no intermediate setting, which is what a policy gradient
carrying no discriminating signal looks like.

---

## 6. Figures

```
python figures/make_figures_v1.py
```

Writes six PNGs to `results/best_figures/`. Measurements are transcribed constants at the top of
the script with their provenance noted, so a re-run of the sweeps above can be diffed against them
rather than silently overwriting them.

---

## Notes on determinism

- Sweeps run under `ProcessPoolExecutor`; on Windows this is **spawn**, so live torch objects do
  not survive the process boundary. Modules receive a string tag and load their own weights inside
  the worker.
- `sim_instance` is deterministic given a seed — the determinism probe asserts this across
  `comfort_T`, `hydration_T`, `death_T`, `action_T` and `consumer_slot_T`.
- Evaluation seeds are **fixed and disjoint** from certification seeds. Training evaluates on
  10001–10008; sensitivity harnesses score on 0–7. Selecting a checkpoint on one set and
  certifying on the other is what keeps the certification honest.
- `torch.set_num_threads(1)` is set per worker; four parallel workers each spawning a full thread
  pool will thrash a laptop.
