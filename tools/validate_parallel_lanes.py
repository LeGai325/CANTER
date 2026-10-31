"""Freeze and run single-A/single-B/parallel-AB transport checks, with no model."""
import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
sys.path.insert(0, str(ROOT))
from specnet.checkpoints import atomic_json, utc_now
from specnet.parallel_execution import run_parallel
from tools.native_network_helper import discovery_filter_present


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def topology(cpus):
    result = {}
    for cpu in cpus:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        result[str(cpu)] = dict(package=int((base / "physical_package_id").read_text()),
                               core=int((base / "core_id").read_text()),
                               siblings=(base / "thread_siblings_list").read_text().strip())
    return result


def prepare(args):
    base = args.output.resolve()
    cpus = {"A": sorted(args.lane_a_cpus), "B": sorted(args.lane_b_cpus)}
    allowed = os.sched_getaffinity(0)
    if not set(cpus["A"] + cpus["B"]) <= allowed or set(cpus["A"]) & set(cpus["B"]):
        raise ValueError("invalid lane affinities")
    physical = {lane: {(r["package"], r["core"]) for r in topology(values).values()} for lane, values in cpus.items()}
    if physical["A"] & physical["B"]:
        raise ValueError("lanes must not use siblings of the same physical core")
    base.mkdir(parents=True, exist_ok=False)
    source = base / "source"
    sources = list((ROOT / "specnet").rglob("*.py")) + list((ROOT / "tools").glob("*.py"))
    for path in sources:
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob("*.py")}
    atomic_json(base / "SOURCE.json", dict(files=hashes, git_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    (base / "specs").mkdir(); (base / "plans").mkdir()
    predecessors = [str(p.resolve()) for p in args.predecessor_status]
    protocol = dict(schema="parallel-transport-validation-v2", created_utc=utc_now(),
        cpu_affinity=cpus, topology=topology(cpus["A"] + cpus["B"]), predecessor_status=predecessors,
        case_order=["single_a", "single_b", "parallel_ab"], duration_seconds=18.,
        thresholds=dict(fct_relative=.10, fct_absolute_seconds=.020, release_lateness_seconds=.020,
                        capacity_apply_lateness_seconds=.100, common_start_tolerance_seconds=.001),
        capture_contract="All captured IPv4 must be exact own-pair experiment TCP or separately counted strict local IGMP/mDNS. Own-host IPv4 multicast and broadcast are blocked on the private interface before host netem under the point-to-point TCP experiment contract; rules and counters retained. ARP and all unicast remain unchanged. Unknown and cross-lane traffic fail.",
        scope="One bounded zero-model HTTP check. Affinity is not exclusive CPU allocation; no full-load or API-concurrency claim.")
    for case, lanes in (("single_a", ["A"]), ("single_b", ["B"]), ("parallel_ab", ["A", "B"])):
        batch = base / "batches" / case
        nonce = uuid.uuid4().hex
        jobs = []
        for lane in lanes:
            requests = [dict(request_id=f"{lane}-q1-{i}", target=f"/{lane}/q1/{i}", role="Q1",
                             payload_bytes=4096, release_seconds=1. + .7 * i) for i in range(8)]
            if lane == "B":
                requests += [dict(request_id=f"B-q0-{i}", target=f"/B/q0/{i}", role="Q0", payload_bytes=256,
                                  control_bound_bytes=256, release_seconds=1.2 + .7 * i) for i in range(8)]
            spec = dict(nonce=nonce, lane=lane, requests=requests, cpu_affinity=cpus[lane],
                capacity_kbit=256 if lane == "A" else 128, queue_residence=lane == "B", duration_seconds=18.,
                capacity_events=[dict(offset_seconds=2., capacity_kbit=64), dict(offset_seconds=6., capacity_kbit=256)] if lane == "A" else [],
                peer_ready=[dict(lane=peer, path=str(batch / "jobs" / peer / "run/READY.json")) for peer in lanes if peer != lane])
            path = base / "specs" / f"{case}_{lane}.json"
            atomic_json(path, spec)
            jobs.append(dict(id=lane, kind="validation", entrypoint=str(source / "tools/run_parallel_probe.py"),
                python=str(PROJECT / ".venv/bin/python"), cwd=str(source), args=["--spec", str(path)],
                output=str(batch / "jobs" / lane / "run"), cpu_affinity=cpus[lane], input_files=[str(path)]))
        atomic_json(base / "plans" / f"{case}.json", dict(schema="specnet-parallel-v1", jobs=jobs, predecessor_status=predecessors))
    protocol["input_sha256"] = {str(p.relative_to(base)): sha(p) for d in ("specs", "plans") for p in (base / d).glob("*.json")}
    atomic_json(base / "PROTOCOL.json", protocol)
    atomic_json(base / "STATUS.json", dict(state="prepared", api_calls=0, updated_utc=utc_now()))


def evaluate(results, protocol):
    checks = []
    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))
    threshold = protocol["thresholds"]
    for case, lanes in results.items():
        for lane, result in lanes.items():
            prefix = f"{case}/{lane}"
            check(prefix + ":bodies", result["all_bodies_verified"])
            check(prefix + ":termination", result["all_locally_terminated"])
            identity = result["capture_identity"]
            check(prefix + ":capture", result["capture_drops"] == 0 and identity["own_ipv4_packets"] > 0
                  and identity["foreign_ipv4_packets"] == identity["malformed_ipv4_packets"] == 0
                  and identity.get("own_unexpected_ipv4_packets") == 0)
            isolation = result.get("host_discovery_isolation", {})
            check(prefix + ":discovery_isolation", isolation.get("installed") is True
                  and isolation.get("final_filter_verified") is True
                  and isolation.get("interface") == result["network"]["host_interface"]
                  and isolation.get("source_ip") == result["network"]["host_ip"]
                  and isolation.get("destination_ip") == str(ipaddress.ip_network(
                      result["network"]["host_ip"] + "/30", strict=False).broadcast_address)
                  and discovery_filter_present(isolation.get("final_kernel_state", []),
                      result["network"]["host_ip"], isolation.get("destination_ip")))
            check(prefix + ":release", result["max_release_lateness_seconds"] <= threshold["release_lateness_seconds"],
                  observed=result["max_release_lateness_seconds"])
            check(prefix + ":zero_model", result["api_calls"] == 0)
            check(prefix + ":affinity", result["cpu_after"]["cpu_affinity"] == protocol["cpu_affinity"][lane])
            children = result.get("child_affinities", {})
            required = {"helper", "capture", "tracer"} if lane == "B" else {"helper", "capture"}
            check(prefix + ":child_affinity", required <= children.keys() and
                  all(v["cpu_affinity"] == protocol["cpu_affinity"][lane] for v in children.values()))
            check(prefix + ":init_lock", bool(result["network"].get("initialization_lock")))
            # Helper reports the disabled feature set per interface.
            check(prefix + ":offloads", bool(result["network"].get("offload_disabled")))
            if lane == "B":
                check(prefix + ":bpf", result["queue_residence_valid"])
                check(prefix + ":constant_capacity", bool(result["telemetry"]) and not result["capacity"]["events"] and
                      all(row["capacity_kbit"] == 128 for row in result["telemetry"]))
            else:
                events = result["capacity"]["events"]
                check(prefix + ":capacity_changes", [e["capacity_kbit"] for e in events] == [64, 256] and
                    all(0 <= e["applied_at"] - e["scheduled_at"] <= threshold["capacity_apply_lateness_seconds"] for e in events))
    pair = results["parallel_ab"]
    for key in ("name", "host_interface", "host_ip", "server_ip"):
        check("pair:distinct_" + key, pair["A"]["network"][key] != pair["B"]["network"][key])
    check("pair:common_start", abs(pair["A"]["window_start"] - pair["B"]["window_start"]) <= threshold["common_start_tolerance_seconds"])
    comparisons = []
    for lane in ("A", "B"):
        single, parallel = results["single_" + lane.lower()][lane], pair[lane]
        for role in ("Q1",) if lane == "A" else ("Q0", "Q1"):
            samples_valid = all(item["roles"][role]["fct_seconds"] and
                all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in item["roles"][role]["fct_seconds"])
                and isinstance(item["roles"][role]["fct_p95_seconds"], (int, float))
                and math.isfinite(item["roles"][role]["fct_p95_seconds"]) for item in (single, parallel))
            check(f"pair:{lane}:{role}:samples", samples_valid)
            if not samples_valid:
                comparisons.append(dict(lane=lane, role=role, comparable=False, reason="missing or invalid completed-request FCT samples"))
                continue
            for metric in ("median", "p95"):
                def value(item):
                    return statistics.median(item["roles"][role]["fct_seconds"]) if metric == "median" else item["roles"][role]["fct_p95_seconds"]
                original, concurrent = value(single), value(parallel)
                limit = max(threshold["fct_absolute_seconds"], original * threshold["fct_relative"])
                row = dict(lane=lane, role=role, metric=metric, single_seconds=original,
                           parallel_seconds=concurrent, absolute_difference=abs(concurrent - original), allowed_difference=limit)
                comparisons.append(row)
                check(f"pair:{lane}:{role}:{metric}", row["absolute_difference"] <= limit, **row)
    return dict(passed=all(r["passed"] for r in checks), checked=len(checks), violations=sum(not r["passed"] for r in checks),
                checks=checks, comparisons=comparisons, scope=protocol["scope"], api_calls=0)


def run(args):
    base = args.output.resolve()
    protocol = json.loads((base / "PROTOCOL.json").read_text())
    source = json.loads((base / "SOURCE.json").read_text())["files"]
    if any(sha(base / "source" / p) != v for p, v in source.items()):
        raise ValueError("frozen source changed")
    if any(sha(base / p) != v for p, v in protocol["input_sha256"].items()):
        raise ValueError("frozen validation inputs changed")
    results = {}
    try:
        for case in protocol["case_order"]:
            atomic_json(base / "STATUS.json", dict(state="waiting_or_running", case=case, completed=list(results), api_calls=0, updated_utc=utc_now()))
            plan = json.loads((base / "plans" / f"{case}.json").read_text())
            status = run_parallel(plan, base / "batches" / case, wait=True, poll_seconds=5.)
            if status["state"] != "completed":
                raise RuntimeError("incomplete validation batch")
            results[case] = {job["id"]: json.loads((Path(job["output"]) / "RESULT.json").read_text()) for job in plan["jobs"]}
        result = evaluate(results, protocol)
        atomic_json(base / "VERIFICATION.json", result)
        atomic_json(base / "RESULTS.json", results)
        atomic_json(base / "STATUS.json", dict(state="completed", validation_passed=result["passed"], api_calls=0, updated_utc=utc_now()))
        (base / "REPORT.md").write_text("# 双通道小规模隔离验证\n\n"
            f"检查 {result['checked']} 项，未通过 {result['violations']} 项。准入结论：{'在本次小规模条件下通过' if result['passed'] else '暂不启用正式并行'}。\n\n"
            "顺序为 A 单跑、B 单跑、A+B 同时跑；无模型调用。A 有容量下降与恢复，B 有 Q0/Q1 和内核驻留采集。"
            "CPU 亲和性不代表独占 CPU；一次小规模通过不证明高负载或同提供商多 Agent 并行无干扰。\n\n"
            "实验合同仅含点对点 TCP；私有 veth 在 host netem 之前阻断本宿主 IPv4 组播及广播，保留 ARP 和全部单播。规则、丢弃计数和原始 pcap 保留，"
            "未将发现广播计为 Agent Wire，也未放宽跨链路或未知包的失败门槛。旧验证结果保持不变。\n\n"
            "比较值、预先固定的容差、逐项失败记录见 VERIFICATION.json；原始请求、pcap、队列和进程日志均保留。\n")
    except BaseException as exc:
        atomic_json(base / "STATUS.json", dict(state="failed", completed=list(results), error_type=type(exc).__name__, api_calls=0, updated_utc=utc_now()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor-status", type=Path, action="append", default=[])
    parser.add_argument("--lane-a-cpus", type=int, nargs="+", default=[8, 9, 136, 137])
    parser.add_argument("--lane-b-cpus", type=int, nargs="+", default=[10, 11, 138, 139])
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    main()
