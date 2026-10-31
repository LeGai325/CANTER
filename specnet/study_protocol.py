"""The user's 150-question E0/E1/E2/E4 protocol, independent of outcomes.

The primary matrix is closed: extra diagnostics never silently change its rows.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from .native_validation import ARMS
from .native_workload import build_arrival_plan, validate_observation_window


PROTOCOL_VERSION = "paper150-e0-e1-e2-e4-v1"
PRIMARY_MATRIX = {
    "light": ("native", "full"),
    "bottleneck": ("hard_shared", "hard_qos", "admit_shared", "fixed_total_2", "no_time", "full"),
    "overload": ("fixed_total_2", "full"),
    "capacity": ("no_net", "full"),
    "background_burst": ("no_net", "full"),
}
STUDY_ARMS = dict(ARMS, no_net=dict(admission=True, mode="qos", network_feedback=False),
    recent_only=dict(admission=True, mode="qos", recent_only=True),
    no_wait=dict(admission=True, mode="qos", admission_wait_enabled=False),
    bootstrap_relaxed=dict(admission=True, mode="qos", bootstrap_group_limit=4))
METRIC_CONTRACT = dict(
    primary=["all_arrival_f1", "effective_otd", "wire_kib_per_arrival", "background_relative_loss"],
    f1="all arrivals, invalid or undelivered zero; frozen benchmark scorer",
    otd="protocol-valid delivery at or before final deadline / all arrivals, correctness not required",
    wire="private tool veth bidirectional original Ethernet frame lengths, retransmission/cleanup included, warmup excluded; not model traffic",
    background="1 - sum(receiver payload in fixed windows) / sum(matched no-Agent payload); negative retained, zero reference NA",
    latency="raw timely valid delivered latencies P95/P99 with undelivered rate, never average block percentiles",
    nq="derived answer-set text EM/F1, not official span leaderboard",
    repetition="3 disjoint 50-task blocks, each task once per configuration; not three repeats of 150 questions")


def source_manifest(root):
    root = Path(root)
    paths = list((root / "specnet").rglob("*.py")) + list((root / "tools").glob("*.py"))
    paths += list((root / "specnet/vendor").glob("*.json"))
    for name in ("pyproject.toml", "uv.lock", "requirements.txt", "requirements-experiment.txt"):
        if (root / name).is_file(): paths.append(root / name)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def validate_protocol(config, *, formal):
    if config.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("unknown paper150 protocol")
    if formal and (config.get("tasks_per_benchmark") != 150 or config.get("block_size") != 50):
        raise ValueError("formal protocol requires 150 tasks and three disjoint 50-task blocks")
    if config.get("dataset_adapter") != "provided-context-v1":
        raise ValueError("this execution scope is provided-context-v1")
    if set(config["scenarios"]) != set(PRIMARY_MATRIX):
        raise ValueError("E0/E1/E2 scenario matrix differs from the user protocol")
    if not 0 < config["acquisition_seconds"] < config["deadline_seconds"]:
        raise ValueError("acquisition cutoff must precede final deadline")
    for name in ("max_wire_bytes_per_arrival", "max_background_loss", "min_effective_otd", "e0_max_quality_loss"):
        if not math.isfinite(config["service_targets"][name]):
            raise ValueError("service targets must be frozen finite numbers")
    for name, settings in config["scenarios"].items():
        if list(settings.get("arms", PRIMARY_MATRIX[name])) != list(PRIMARY_MATRIX[name]):
            raise ValueError("primary group definitions cannot be silently changed")
        if settings["capacity_kbit"] <= 0:
            raise ValueError("capacity must be positive")
        if name == "capacity" and settings.get("background_phases"):
            raise ValueError("capacity experiment must not also change background")
        if name == "background_burst" and settings.get("capacity_schedule"):
            raise ValueError("background experiment must not also change capacity")
    return True


def scenario_config(config, name, tasks):
    """Freeze all timing from task count and settings, never actual completion."""
    result = deepcopy(config)
    settings = deepcopy(config["scenarios"][name])
    result.update({key: value for key, value in settings.items() if key != "arms"})
    result["arm_settings"] = deepcopy(STUDY_ARMS)
    result["development_service_targets"] = dict(
        max_background_loss=config["service_targets"]["max_background_loss"],
        min_effective_otd=config["service_targets"]["min_effective_otd"],
        max_agent_wire_bytes_per_arrival=config["service_targets"]["max_wire_bytes_per_arrival"])
    arrivals = build_arrival_plan(tasks, result)
    last = max((r["offset_seconds"] for r in arrivals), default=0.)
    minimum = last + result["deadline_seconds"] + config.get("observation_tail_seconds", 10)
    result["window_seconds"] = max(minimum, settings.get("minimum_window_seconds", 0))
    if name in {"capacity", "background_burst"}:
        # Task blocks span all three phases even for small development cohorts.
        span = max(last, config.get("dynamic_minimum_arrival_span_seconds", 90))
        if len(arrivals) > 1:
            for i, row in enumerate(arrivals): row["offset_seconds"] = i * span / (len(arrivals) - 1)
        down, restore = span / 3, 2 * span / 3
        result["time_segments"] = [dict(name="before", offset_seconds=0.),
            dict(name="intervention", offset_seconds=down), dict(name="recovery", offset_seconds=restore)]
        result["window_seconds"] = span + result["deadline_seconds"] + config.get("observation_tail_seconds", 10)
        if name == "capacity":
            result["capacity_schedule"] = [dict(offset_seconds=down, capacity_kbit=settings["impaired_capacity_kbit"], phase="intervention"),
                dict(offset_seconds=restore, capacity_kbit=settings["capacity_kbit"], phase="recovery")]
        else:
            result["background_phases"] = [
                dict(offset_seconds=0., object_bytes=result["background_object_bytes"], release_interval_seconds=result["background_release_interval_seconds"], phase="before"),
                dict(offset_seconds=down, object_bytes=result["background_object_bytes"], release_interval_seconds=settings["burst_release_interval_seconds"], phase="intervention"),
                dict(offset_seconds=restore, object_bytes=result["background_object_bytes"], release_interval_seconds=result["background_release_interval_seconds"], phase="recovery")]
    validate_observation_window(arrivals, result)
    return result, arrivals


def cells_for_block(bench, block_id, scenario, *, seed, include_ablations=False):
    prefix = f"{bench}_b{block_id:02d}_{scenario}"
    reference = prefix + "_reference"
    arms = list(PRIMARY_MATRIX[scenario])
    if include_ablations and scenario == "capacity":
        arms += ["recent_only", "no_wait", "bootstrap_relaxed"]
    stable = int(hashlib.sha256(f"{seed}:{bench}:{block_id}:{scenario}".encode()).hexdigest()[:16], 16)
    random.Random(stable).shuffle(arms)
    common = dict(benchmark=bench, block_id=block_id, phase=scenario, reference_key=reference)
    return [dict(common, key=reference, arm=None, table="reference")] + [
        dict(common, key=prefix + "_" + arm, arm=arm,
             table="primary" if arm in PRIMARY_MATRIX[scenario] else "ablation") for arm in arms]


def check_unique_cohort(loaded, formal):
    tasks = [t for _, batch, _ in loaded["evaluation"] for t in batch]
    warm = [t.id for _, batch, _ in loaded["warmup"] for t in batch]
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids) or set(ids) & set(warm) or len(set(warm)) != len(warm):
        raise ValueError("duplicate task IDs or evaluation/warmup overlap")
    if formal and len(tasks) != 50:
        raise ValueError("formal block must contain exactly 50 distinct tasks")
    return tasks
