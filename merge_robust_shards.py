#!/usr/bin/env python3
"""Merge per-seed robustness shards into one directory per scenario.

The comparison is fail-loud but seed-aware:
  * each shard must contain exactly the seed encoded in its directory name;
  * tuned-model provenance must exist and be byte-identical across shards;
  * resolved configs must be semantically identical after removing only the
    top-level ``seeds`` field, which is expected to differ by construction;
  * the canonical LTE environment (environment.json -> lte_env) must match,
    excluding CUDA_VISIBLE_DEVICES, which the GPU queue assigns per unit and
    which therefore also differs by construction;
  * the merged resolved_config.json and environment.json are rewritten to
    describe the complete merged seed pool rather than the reference seed only.

Example:
    python merge_robust_shards.py --prefix robust_s3c5_cal \
        --scenarios "xlarge default stmsi voice_heavy streaming_heavy \
                     browsing_heavy medium small tiny" \
        --seeds "101 102 103 104 105 106 107 108 109 110"
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Copied from the reference shard after cross-shard validation where required.
SHARED_FILES = (
    "parameter_registry.csv",
    "feature_manifest_A.json",
    "feature_manifest_B.json",
    "tuned_model_provenance.json",
)
SHARED_DIRS = ("selected_best_params",)

# One file per seed; copied from every shard.
PER_SEED_GLOBS = (
    "preds_seed*.npz",
    "trace_seed*.csv.gz",
    "truth_hashes_seed*.json",
)

# These files are required and must be byte-for-byte identical across shards.
BYTE_IDENTICAL_FILES = ("tuned_model_provenance.json",)

# runner.py records LTE_* plus CUDA_VISIBLE_DEVICES and XGB_FORCE_CPU under
# environment.json -> lte_env. The queue assigns CUDA_VISIBLE_DEVICES per unit,
# so it is operational provenance rather than part of the experiment definition.
ENV_VOLATILE = ("CUDA_VISIBLE_DEVICES",)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    """Read a required JSON object or stop with a descriptive error."""
    if not path.is_file():
        raise SystemExit(f"{label}: missing required file: {path}")

    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label}: cannot parse {path}: {exc}") from exc

    if not isinstance(value, dict):
        raise SystemExit(f"{label}: expected a JSON object in {path}")

    return value


def _normalised_config(
    path: Path,
    expected_seed: int,
    scenario: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return raw config and comparison copy with only top-level seeds removed."""
    raw = _read_json(path, label=scenario)
    recorded = raw.get("seeds")

    if recorded != [expected_seed]:
        raise SystemExit(
            f"{scenario}: {path} records seeds={recorded!r}; expected exactly "
            f"[{expected_seed}] for this per-seed shard."
        )

    normalised = copy.deepcopy(raw)
    normalised.pop("seeds", None)
    return raw, normalised


def _environment(path: Path, scenario: str) -> dict[str, Any]:
    """Read environment.json and require an object-valued lte_env field."""
    env = _read_json(path, label=scenario)

    if not isinstance(env.get("lte_env"), dict):
        raise SystemExit(f"{scenario}: {path} has no object-valued 'lte_env'")

    return env


def _canonical_lte_env(env: dict[str, Any]) -> dict[str, Any]:
    """Return lte_env excluding queue-assigned, non-experimental variables."""
    return {
        key: value
        for key, value in env["lte_env"].items()
        if key not in ENV_VOLATILE
    }


def _first_difference(a: Any, b: Any, path: str = "$") -> str | None:
    """Return a compact description of the first structural/value difference."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"

    if isinstance(a, dict):
        a_keys, b_keys = set(a), set(b)
        if a_keys != b_keys:
            return (
                f"{path}: keys differ; "
                f"only-left={sorted(a_keys - b_keys)}, "
                f"only-right={sorted(b_keys - a_keys)}"
            )

        for key in sorted(a_keys):
            diff = _first_difference(a[key], b[key], f"{path}.{key}")
            if diff:
                return diff
        return None

    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: list lengths {len(a)} != {len(b)}"

        for idx, (left, right) in enumerate(zip(a, b)):
            diff = _first_difference(left, right, f"{path}[{idx}]")
            if diff:
                return diff
        return None

    if a != b:
        return f"{path}: {a!r} != {b!r}"

    return None


def _validate_required_identical_files(
    shards: list[tuple[int, Path]],
    scenario: str,
) -> None:
    """Require selected provenance files and compare them byte-for-byte."""
    ref_seed, ref = shards[0]

    for filename in BYTE_IDENTICAL_FILES:
        ref_path = ref / filename
        if not ref_path.is_file():
            raise SystemExit(
                f"{scenario}: reference shard seed {ref_seed} is missing "
                f"required file {filename}"
            )

        expected = ref_path.read_bytes()

        for seed, shard in shards[1:]:
            candidate = shard / filename
            if not candidate.is_file():
                raise SystemExit(
                    f"{scenario}: shard seed {seed} is missing required file "
                    f"{filename}"
                )

            if candidate.read_bytes() != expected:
                raise SystemExit(
                    f"{scenario}: {filename} differs between seed {ref_seed} "
                    f"and seed {seed}. The shards do not share the same "
                    "tuned-model provenance."
                )


def merge(prefix: str, scenario: str, seeds: list[int]) -> str:
    """Merge all requested seed shards for one robustness scenario."""
    if not seeds:
        raise SystemExit("at least one seed is required")

    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"{scenario}: duplicate requested seeds: {seeds}")

    shards = [(seed, Path(f"{prefix}_{scenario}__s{seed}")) for seed in seeds]

    missing = [
        str(shard)
        for _seed, shard in shards
        if not (shard / "results.csv").is_file()
    ]
    if missing:
        raise SystemExit(
            f"{scenario}: missing shard results.csv:\n  " + "\n  ".join(missing)
        )

    ref_seed, ref = shards[0]

    # Required provenance must exist and be byte-identical.
    _validate_required_identical_files(shards, scenario)

    # resolved_config.json must match after excluding only the per-shard seed.
    ref_config_raw, ref_config_cmp = _normalised_config(
        ref / "resolved_config.json",
        ref_seed,
        scenario,
    )

    for seed, shard in shards[1:]:
        _raw, config_cmp = _normalised_config(
            shard / "resolved_config.json",
            seed,
            scenario,
        )

        if config_cmp != ref_config_cmp:
            diff = _first_difference(ref_config_cmp, config_cmp) or "unknown difference"
            raise SystemExit(
                f"{scenario}: resolved_config.json differs between seed "
                f"{ref_seed} and seed {seed} after excluding only top-level "
                f"'seeds'. First difference: {diff}"
            )

    # Full environment.json may differ in timestamp, output, CLI seed and device.
    # Canonical lte_env must match after excluding CUDA_VISIBLE_DEVICES.
    ref_env = _environment(ref / "environment.json", scenario)
    ref_lte_env = _canonical_lte_env(ref_env)
    shard_envs: dict[int, dict[str, Any]] = {ref_seed: ref_env}

    for seed, shard in shards[1:]:
        env = _environment(shard / "environment.json", scenario)
        shard_envs[seed] = env
        lte_env = _canonical_lte_env(env)

        if lte_env != ref_lte_env:
            diff = _first_difference(ref_lte_env, lte_env) or "unknown difference"
            raise SystemExit(
                f"{scenario}: environment.json lte_env differs between seed "
                f"{ref_seed} and seed {seed}. First difference: {diff}"
            )

    # Validate and concatenate the one-row result from each shard.
    frames: list[pd.DataFrame] = []

    for seed, shard in shards:
        results_path = shard / "results.csv"
        try:
            frame = pd.read_csv(results_path)
        except Exception as exc:
            raise SystemExit(
                f"{scenario}: cannot read {results_path}: {exc}"
            ) from exc

        if len(frame) != 1:
            raise SystemExit(
                f"{scenario}: shard for seed {seed} has {len(frame)} rows, "
                "expected 1"
            )

        if "seed" not in frame.columns:
            raise SystemExit(
                f"{scenario}: {results_path} has no 'seed' column"
            )

        try:
            recorded_seed = int(frame["seed"].iloc[0])
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"{scenario}: {results_path} contains a non-integer seed "
                f"value {frame['seed'].iloc[0]!r}"
            ) from exc

        if recorded_seed != seed:
            raise SystemExit(
                f"{scenario}: shard {shard} holds seed {recorded_seed}, "
                f"not {seed}"
            )

        frames.append(frame)

    results = (
        pd.concat(frames, ignore_index=True)
        .sort_values("seed")
        .reset_index(drop=True)
    )

    if results["seed"].duplicated().any():
        duplicates = sorted(
            int(x) for x in results.loc[results["seed"].duplicated(), "seed"]
        )
        raise SystemExit(f"{scenario}: duplicate seeds after merge: {duplicates}")

    merged_seeds = sorted(int(x) for x in results["seed"])
    requested_seeds = sorted(seeds)

    if merged_seeds != requested_seeds:
        raise SystemExit(
            f"{scenario}: merged seed set {merged_seeds} does not equal "
            f"requested {requested_seeds}"
        )

    out = Path(f"{prefix}_{scenario}")
    out.mkdir(parents=True, exist_ok=True)

    results.to_csv(out / "results.csv", index=False)

    # Same convention as runner.py: mean and sample std (ddof=1) across seeds.
    numeric_columns = results.select_dtypes(include=[np.number]).columns.tolist()
    pd.DataFrame(
        {
            "mean": results[numeric_columns].mean(),
            "std": results[numeric_columns].std(ddof=1),
        }
    ).to_csv(out / "summary.csv")

    # Rewrite resolved_config.json to describe the complete merged seed pool.
    merged_config = copy.deepcopy(ref_config_raw)
    merged_config["seeds"] = merged_seeds
    (out / "resolved_config.json").write_text(
        json.dumps(merged_config, indent=2) + "\n"
    )

    # Rewrite environment.json to describe the merged result instead of the
    # reference shard. A merged experiment has no single CUDA mask, so remove
    # it from the representative lte_env and preserve per-seed assignments below.
    merged_env = copy.deepcopy(ref_env)

    if isinstance(merged_env.get("cli_args"), dict):
        merged_env["cli_args"]["seeds"] = merged_seeds
        merged_env["cli_args"]["out"] = str(out)

    if isinstance(merged_env.get("lte_env"), dict):
        merged_env["lte_env"].pop("CUDA_VISIBLE_DEVICES", None)

    merged_env["merged_from_shards"] = True
    merged_env["shard_devices"] = {
        str(seed): {
            "device": shard_envs[seed].get("device"),
            "CUDA_VISIBLE_DEVICES": shard_envs[seed]["lte_env"].get(
                "CUDA_VISIBLE_DEVICES"
            ),
        }
        for seed in merged_seeds
    }
    merged_env["shard_hostnames"] = {
        str(seed): shard_envs[seed].get("hostname")
        for seed in merged_seeds
    }

    (out / "environment.json").write_text(
        json.dumps(merged_env, indent=2) + "\n"
    )

    # Copy validated shared artifacts from the reference shard.
    for filename in SHARED_FILES:
        src = ref / filename
        if src.is_file():
            shutil.copy2(src, out / filename)

    for dirname in SHARED_DIRS:
        src = ref / dirname
        if src.is_dir():
            shutil.copytree(src, out / dirname, dirs_exist_ok=True)

    # Copy per-seed predictions, traces and truth hashes.
    for _seed, shard in shards:
        for pattern in PER_SEED_GLOBS:
            for src in shard.glob(pattern):
                destination = out / src.name
                if destination.exists():
                    raise SystemExit(
                        f"{scenario}: refusing to overwrite duplicate per-seed "
                        f"artifact {destination}"
                    )
                shutil.copy2(src, destination)

    (out / "merged_from_shards.json").write_text(
        json.dumps(
            {
                "prefix": prefix,
                "scenario": scenario,
                "seeds": merged_seeds,
                "shard_dirs": [str(path) for _seed, path in shards],
                "resolved_config_comparison": (
                    "semantic, excluding only top-level seeds"
                ),
                "environment_comparison": (
                    "environment.json:lte_env, excluding "
                    + ", ".join(ENV_VOLATILE)
                ),
                "required_byte_identical_files": list(BYTE_IDENTICAL_FILES),
            },
            indent=2,
        )
        + "\n"
    )

    return f"{out}/  <- {len(results)} seeds {merged_seeds}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--scenarios",
        required=True,
        help="space-separated scenario names",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        help="space-separated integer seeds",
    )
    args = parser.parse_args()

    try:
        seeds = [int(value) for value in args.seeds.split()]
    except ValueError as exc:
        raise SystemExit(f"invalid --seeds value: {args.seeds!r}") from exc

    scenarios = args.scenarios.split()
    if not scenarios:
        raise SystemExit("at least one scenario is required")

    for scenario in scenarios:
        print("  " + merge(args.prefix, scenario, seeds))

    print(f"\nmerged {len(scenarios)} scenarios x {len(seeds)} seeds")
    print(
        "The __s<seed>/ shard dirs are kept; delete them once the tables "
        "and provenance checks are complete."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
