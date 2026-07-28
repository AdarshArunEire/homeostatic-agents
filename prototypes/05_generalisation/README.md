# Prototype 05 — curriculum and earned memory (abandoned mid-hypothesis)

**Status: superseded by [`06_feudal`](../06_feudal). Kept because the reframe it triggered is
part of the argument, and because abandoning it was a decision worth recording.**

## What it was for

Prototype 04 ended with a reframe rather than a number: stop optimising for a configuration that
is robust across random seeds on one fixed map, and change the success criterion to **one weight
set that generalises across resampled maps**. Prototype 05 was the first attempt at that.

Two mechanisms were chosen:

**Curriculum over band width.** Commute distance is a non-leaking difficulty variable — it does
not reveal *where* water is, only how far. 3b's earlier coordinate curriculum had failed
(≈6/10 → 1/10) and the suspected cause was forgetting at each widening handoff rather than the
principle itself, so it was retried with band width as the axis.

**`touch_memory`.** A displacement vector to the last *personally visited* water and food tile —
map-invariant, with a never-seen sentinel on fresh maps. This was the honest
partial-observability intervention: earned memory rather than a pre-populated coordinate.

The hypothesis under test (H1) was whether `touch_memory` improves generalisation: a two-arm
sweep, 8 seeds, noisy DQN, control `(smell, vision)` against memory `(smell, vision,
touch_memory)`.

## Why it stopped

The sweep was launched and never verdicted. Work moved to Prototype 06 before the result was
read, for a reason that had become clear while building the harness:

**Prototype 05 still bundled every capability into one policy.** Survival, navigation, consume
timing and exploration were all being learned at once and measured through a single number, so a
null result could not be attributed to any of them. Proto 04 had already isolated the bottleneck
to directed exploration — but 05's design could not *test* exploration in isolation, only observe
its consequences mixed with three other failure modes.

Decomposing the agent (Proto 06) was the way to make the question answerable. That reframe is
what 05 produced, and it was worth more than finishing the sweep.

## What carried forward

- **Segment-aware `compute_eval_metrics`** — each evaluation slice scored against its own map's
  coordinates, then pooled. Without this, a trip on map B is scored against map A's water.
- **Scaffold-cached `HexWorld`** — everything keyed on `(radius, band)` is seed-independent, so
  the expensive candidate scan is computed once per floor and shared read-only. Still in use.
- **Convicted metrics.** Comfort and death-rate are unreliable under resampling and soft decay:
  camping produces high comfort and low death rate, indistinguishable from genuine learning.
  Proto 06 uses `solveScore` and segment-aware trip metrics instead.
- **The honesty test for oracles.** God-sense (coordinate pre-population) is legitimate only for
  plumbing smoke tests; earned memory — discover, then exploit — is the only honest orchestrator
  oracle. Proto 06's `world_v1` makes this a greppable boundary.

## What is in this folder

Sweep outputs from the curriculum and self-imitation experiments, plus the cached-world and
curriculum modules that Proto 06 inherited. The notebook and `oracle_v2.py` are the pre-decomposition
design, retained for reference.

The open question 05 flagged and never answered — *does resampling re-motivate the memory channel
that Proto 04 H4 retired?* — was inherited by Proto 06 and answered there: memory has a
non-redundant job in principle, but the explorer's observation contract cannot support it. See
[`06_feudal/README.md`](../06_feudal/README.md).
