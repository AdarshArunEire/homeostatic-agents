"""
checkpoint_v1.py  —  weight store for Proto 06 learned modules.

One file per trained module, under `prototypes/06_feudal/results/weights/<kind>/<tag>.pt`.
The .pt blobs are gitignored; MANIFEST.json is committed. The manifest plus the committed
trainer + rig is the reproduction path: every entry records the training config, the seed,
and the metric achieved, so a regenerated file can be checked against its record rather
than trusted.

THREE CONSTRAINTS THIS FILE EXISTS TO SATISFY, all consequences of the sweep harness:

  1. Sweeps run under ProcessPoolExecutor on Windows (spawn, not fork). Live torch objects
     do not survive the pickle across the process boundary. So `oracle_params` carries a
     STRING TAG, and the module loads its own weights inside the worker on construction.
     Never put a net in a config dict.

  2. 4 workers x N seeds re-reading the same .pt is pure waste. `load_net` caches per
     process, keyed by resolved path.

  3. Torch defaults to a full thread pool per process. Four of those on a 4-core box
     thrashes. `prepare_worker()` pins to 1 thread; call it once at worker entry.

Payload is never a bare state_dict — without `arch` you cannot rebuild the net, and
without the metrics/config you cannot audit the ledger entry.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

# contract version this checkpoint format is valid against; bump on any Obs change,
# because a stale net silently fed a reshaped observation is the worst failure mode here.
CONTRACT_VERSION = "v1"

VALID_KINDS = ("pathfinder", "orchestrator", "eat", "drink", "explorer")


# --- paths ----------------------------------------------------------------------

def prototype_dir() -> Path:
    """
    Resolved from __file__, deliberately, NOT from cwd.

    sweep_fn_v5.repo_root() walks up from Path.cwd() looking for .git, which is correct
    for notebooks launched at repo root but wrong for a worker process whose cwd is
    wherever the executor put it. Weights must resolve identically from a notebook, a
    test, and a spawned worker.
    """
    return Path(__file__).resolve().parents[1]


def weights_dir() -> Path:
    return prototype_dir() / "results" / "weights"


def manifest_path() -> Path:
    return weights_dir() / "MANIFEST.json"


def resolve(kind: str, tag: str) -> Path:
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown module kind {kind!r}; expected one of {VALID_KINDS}")
    return weights_dir() / kind / f"{tag}.pt"


# --- git provenance -------------------------------------------------------------

def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(prototype_dir()),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(prototype_dir()),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


# --- save / load ----------------------------------------------------------------

def save_module(
    net,
    *,
    kind: str,
    tag: str,
    arch: dict,
    obs_fields: list[str],
    metrics: dict,
    train_config: dict,
    train_seed: int,
    notes: str = "",
) -> Path:
    """
    Write one module checkpoint and register it in MANIFEST.json.

    `arch` must be sufficient for nets_v1.build_net(arch) to rebuild the module blind.
    `obs_fields` records which Obs fields the encoder consumed, in order — this is the
    tripwire for train/inference skew when the contract changes underneath a stale file.
    """
    path = resolve(kind, tag)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "contract_version": CONTRACT_VERSION,
        "module_kind": kind,
        "tag": tag,
        "arch": arch,
        "obs_fields": list(obs_fields),
        "state_dict": net.state_dict(),
        "metrics": metrics,
        "train_config": train_config,
        "train_seed": train_seed,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)

    _register(payload)
    return path


def _register(payload: dict) -> None:
    """Update the committed manifest. Blob-free: everything here is reviewable in a diff."""
    mp = manifest_path()
    mp.parent.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text())
        except json.JSONDecodeError:
            manifest = {}

    entries = manifest.setdefault("entries", {})
    key = f"{payload['module_kind']}/{payload['tag']}"
    entries[key] = {
        k: payload[k]
        for k in (
            "contract_version", "module_kind", "tag", "arch", "obs_fields",
            "metrics", "train_config", "train_seed", "git_sha", "git_dirty",
            "created", "notes",
        )
    }
    manifest["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(mp)


# per-process cache; see constraint 2 in the module docstring
_CACHE: dict[str, tuple] = {}


def load_payload(kind: str, tag: str) -> dict:
    path = resolve(kind, tag)
    if not path.exists():
        raise FileNotFoundError(
            f"no weights at {path}.\n"
            f"Weights are gitignored — regenerate with the trainer recorded in "
            f"{manifest_path()} under key '{kind}/{tag}'."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"{path} was written against contract {payload.get('contract_version')!r}, "
            f"this tree is {CONTRACT_VERSION!r}. Retrain rather than coerce — a stale net "
            f"fed a reshaped observation fails silently."
        )
    if payload.get("module_kind") != kind:
        raise ValueError(
            f"{path} holds a {payload.get('module_kind')!r} module, loaded as {kind!r}."
        )
    return payload


def load_net(kind: str, tag: str):
    """
    Rebuild a frozen, eval-mode net from its checkpoint. Cached per process.

    Returns (net, payload) so callers can assert on obs_fields before trusting it.
    """
    key = str(resolve(kind, tag))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    from model_modules.learned_modules.nets_v1 import build_net

    payload = load_payload(kind, tag)
    net = build_net(payload["arch"])
    net.load_state_dict(payload["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    _CACHE[key] = (net, payload)
    return net, payload


def clear_cache() -> None:
    _CACHE.clear()


# --- worker hygiene -------------------------------------------------------------

def prepare_worker(n_threads: int = 1) -> None:
    """
    Call once at the top of a sweep worker. See constraint 3.
    Safe to call repeatedly; torch tolerates re-setting this.
    """
    torch.set_num_threads(n_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
