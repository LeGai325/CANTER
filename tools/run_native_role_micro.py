"""Fixed HTTP fixture for role scheduling; no Agent quality/API experiment.

Both modes receive identical absolute releases, payloads and fresh TCP
connections. Background remains independent, with a no-Agent reference.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from specnet.checkpoints import atomic_json, utc_now
from specnet.experiment import exclusive_run
from specnet.native_network import NetworkLab, NativeContentService, NativeHTTPClient, parse_pcap_window

WEIGHTS = dict(Q0=4, Q1=4, Q2=1, Q3=1, BG=2)
SOURCE_FILES = ("tools/run_native_role_micro.py", "tools/native_network_helper.py",
                "tools/network_helper.py", "specnet/native_network.py",
                "specnet/native_telemetry.py", "specnet/network.py",
                "specnet/termination.py", "specnet/checkpoints.py")


def schedule():
    rows = [dict(request_id=f"bulk-{role}", role=role, release_seconds=0., payload_bytes=131072,
                 role_evidence=evidence) for role, evidence in (
        ("Q1", "fixture native continuation awaits this body"),
        ("Q2", "fixture role unknown; default class"),
        ("Q3", "fixture explicitly permits native continuation without prefetch result"))]
    rows += [dict(request_id=f"control-{index}", role="Q0", release_seconds=float(index),
                  payload_bytes=1024, role_evidence="bounded 1024-byte protocol control fixture")
             for index in range(1, 6)]
    return rows


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(len(values) * p) - 1]


def summarize_requests(records, window_end):
    result = {}
    for role in ("Q0", "Q1", "Q2", "Q3"):
        group = [row for row in records if row["role"] == role]
        completed = [row for row in group if row.get("body_done_at", math.inf) < window_end
                     and row.get("body_verified")]
        fcts = [row["body_done_at"] - row["started_at"] for row in completed]
        result[role] = dict(requests=len(group), completed=len(completed),
            unfinished=len(group) - len(completed),
            unfinished_rate=(len(group) - len(completed)) / len(group) if group else None,
            fct_mean_seconds=statistics.mean(fcts) if fcts else None,
            fct_p95_seconds=percentile(fcts, .95), fct_seconds=fcts,
            received_payload_bytes=sum(sum(chunk["bytes"] for chunk in row.get("received_chunks", [])
                                            if chunk["at"] < window_end) for row in group),
            status_counts=dict(Counter(row.get("status", "not_started") for row in group)))
    return result


async def run_cell(directory, mode, protocol):
    directory.mkdir()
    rows, background, service_records = [], None, []
    fixture = {row["request_id"]: row for row in protocol["requests"]}
    def resolve(target, role):
        prefix = "/fixture/"
        if not target.startswith(prefix):
            raise ValueError("unknown fixture path")
        row = fixture[target[len(prefix):]]
        if role != row["role"]:
            raise ValueError("fixture role mismatch")
        return (role.encode("ascii") * ((row["payload_bytes"] + 1) // 2))[:row["payload_bytes"]]

    async with NetworkLab(directory / "network") as lab:
        await lab.configure(mode="shared" if mode == "reference" else mode,
                            capacity_kbit=protocol["capacity_kbit"],
                            delay_ms=protocol["delay_ms"], weights=protocol["weights"])
        http = NativeHTTPClient(lab, telemetry_interval=protocol["sampling_seconds"])
        async with NativeContentService(lab, resolve, telemetry=http.telemetry,
                                        telemetry_interval=protocol["sampling_seconds"]) as service:
            await lab.sample()
            start = time.monotonic() + .25
            wall_start = time.time() + start - time.monotonic()
            end = start + protocol["duration_seconds"]
            async def sample_queues():
                while True:
                    await lab.sample()
                    await asyncio.sleep(protocol["sampling_seconds"])
            sampler = asyncio.create_task(sample_queues())

            async def transfer(spec):
                row = dict(spec, scheduled_release_at=start + spec["release_seconds"],
                           cancellation_contract="native_detachable" if spec["role"] == "Q3" else "required_or_unknown")
                rows.append(row)
                await asyncio.sleep(max(0., row["scheduled_release_at"] - time.monotonic()))
                row["released_at"] = time.monotonic()
                row["release_jitter_seconds"] = row["released_at"] - row["scheduled_release_at"]
                try:
                    async with asyncio.timeout_at(end):
                        body = await http.fetch(f"/fixture/{spec['request_id']}", spec["role"],
                                                record=row, response_cap=spec["payload_bytes"],
                                                control_bound_bytes=spec["payload_bytes"] if spec["role"] == "Q0" else None)
                    row["body_verified"] = body == resolve(f"/fixture/{spec['request_id']}", spec["role"])
                    row["body_sha256"] = hashlib.sha256(body).hexdigest()
                except TimeoutError:
                    row["window_timeout"] = True
                except Exception as exc:
                    row["fixture_error_type"] = type(exc).__name__

            transfers = [asyncio.create_task(transfer(row)) for row in protocol["requests"]] if mode != "reference" else []
            bg = asyncio.create_task(http.background(protocol["duration_seconds"],
                object_bytes=protocol["background_object_bytes"],
                release_interval_seconds=protocol["background_interval_seconds"],
                max_inflight=protocol["background_max_inflight"], start_at=start))
            try:
                values = await asyncio.gather(bg, *transfers)
                background = values[0]
            finally:
                for task in [bg, *transfers, sampler]:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(bg, *transfers, sampler, return_exceptions=True)
                await lab.sample()
                atomic_json(directory / "requests.json", rows)
                atomic_json(directory / "background_requests.json", http.background_records)
                atomic_json(directory / "background.json", background)
                service_records = service.records
        atomic_json(directory / "server_requests.json", service_records)
        manifest = dict(lab.manifest)
    # Cleanup/in-flight tail remains in the complete pcap and packet_counts;
    # main wire summary uses the same fixed window as receiver goodput.
    wire = parse_pcap_window(directory / "network/traffic.pcap", manifest["server_ip"],
                             manifest["ports"], wall_start,
                             wall_start + protocol["duration_seconds"])
    queues = json.loads((directory / "network/telemetry.json").read_text())
    valid_queues = [row for row in queues if row.get("queue_valid")
                    and start <= row["observed_at"] < end]
    queue_summary = {}
    for row in valid_queues:
        for name, queue in row["queue_by_class"].items():
            queue_summary.setdefault(name, []).append(queue["queue_bytes"])
    summary = dict(mode=mode, window_start=start, window_end=end,
        window_wall_start=wall_start, window_wall_end=wall_start + protocol["duration_seconds"],
        roles=summarize_requests(rows, end), background=background, wire=wire,
        queue_samples=len(valid_queues),
        queue_by_class={name: dict(mean_bytes=statistics.mean(values), max_bytes=max(values))
                        for name, values in queue_summary.items()},
        max_release_jitter_seconds=max([row["release_jitter_seconds"] for row in rows], default=0.),
        all_started_connections_locally_terminated=all(row.get("local_termination_confirmed") for row in rows),
        errors=[dict(request_id=row["request_id"], error=row["fixture_error_type"])
                for row in rows if "fixture_error_type" in row])
    atomic_json(directory / "SUMMARY.json", summary)
    return summary


async def run(output, duration, capacity):
    if duration < 6 or capacity < 64:
        raise ValueError("duration must cover all releases; capacity must be at least 64 kbit/s")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = dict(schema="fixed-http-role-micro-v1", duration_seconds=duration,
        capacity_kbit=capacity, delay_ms=5, weights=WEIGHTS, sampling_seconds=.1,
        background_object_bytes=1024, background_interval_seconds=.5,
        background_max_inflight=16, requests=schedule(),
        order=["reference", "shared", "qos"], admission="disabled; fixed native eligible release schedule",
        connections="fresh TCP connection per HTTP request; HTTP/1.1 Connection: close",
        Q3="synthetic executor contract explicitly permits detachment; no early cancellation intervention",
        scope="independent scheduling fixture; not real Agent role coverage, QA quality, or public Internet",
        limitations=["one run per mode and fixed order; no independent-repeat confidence intervals",
                     "flat DRR active-class shares may differ; background comparison is reported",
                     "queue backlog sampled; packet queue residence is not directly measured",
                     "tail FCT uses completed requests with unfinished count reported separately"],
        source={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        created_at=utc_now())
    for name in SOURCE_FILES:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    atomic_json(output / "PROTOCOL.json", protocol)
    summaries = []
    try:
        for mode in protocol["order"]:
            if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                   for name, digest in protocol["source"].items()):
                raise RuntimeError("runtime source changed after protocol freeze")
            summaries.append(await run_cell(output / mode, mode, protocol))
            atomic_json(output / "STATUS.json", dict(completed=[row["mode"] for row in summaries], running=True))
            print(json.dumps(dict(completed=mode, roles=summaries[-1]["roles"],
                                  background_bytes=summaries[-1]["background"]["received_payload_bytes"])), flush=True)
        reference = summaries[0]["background"]["received_payload_bytes"]
        for row in summaries:
            row["background_relative_loss"] = 1 - row["background"]["received_payload_bytes"] / reference if reference else None
        report = ["# 固定 HTTP 请求角色隔离微实验", "",
            "真实 TCP/HTTP 与 Linux DRR；不调用模型。这是人工固定执行器契约的调度微实验，不代表真实 Agent 出现了 Q0–Q3。",
            "", f"链路 {capacity} kbit/s，每个条件 {duration:g} 秒；相同释放时间、正文大小、独立连接，准入关闭。",
            "", "| 模式 | 角色 | 完成/请求 | 平均 FCT 秒 | P95 FCT 秒 | 未完成率 |",
            "|---|---|---:|---:|---:|---:|"]
        for row in summaries[1:]:
            for role, values in row["roles"].items():
                mean = f"{values['fct_mean_seconds']:.4f}" if values["fct_mean_seconds"] is not None else "—"
                p95 = f"{values['fct_p95_seconds']:.4f}" if values["fct_p95_seconds"] is not None else "—"
                report.append(f"| {row['mode']} | {role} | {values['completed']}/{values['requests']} | {mean} | {p95} | {values['unfinished_rate']:.1%} |")
        report += ["", "| 模式 | 背景接收正文 bytes | 相对无 Agent 损失 | Agent wire bytes |", "|---|---:|---:|---:|"]
        for row in summaries:
            loss = f"{row['background_relative_loss']:.2%}" if row["background_relative_loss"] is not None else "不可计算"
            agent_wire = sum(value["wire_bytes"] for role, value in row["wire"]["roles"].items() if role != "BG")
            report.append(f"| {row['mode']} | {row['background']['received_payload_bytes']} | {loss} | {agent_wire} |")
        report += ["", "单次固定顺序运行，只说明该固定请求配置下的差异，不提供普遍收益或置信区间。",
                   "Flat DRR 合并会改变活跃类份额；背景独立且同一固定开放到达流量，但不能据此声称完全剥离了所有服务份额差异。",
                   "FCT 为客户端开始连接到完整正文接收；未完成请求不填造 FCT。队列数据是采样积压，不是逐包驻留时间。",
                   "原始请求、partial background、pcap、队列、版本快照均保留。"]
        (output / "REPORT.md").write_text("\n".join(report) + "\n")
        atomic_json(output / "RESULTS.json", summaries)
        atomic_json(output / "STATUS.json", dict(completed=protocol["order"], running=False,
            elapsed_seconds=time.monotonic() - started, api_calls=0, errors=sum(len(row["errors"]) for row in summaries)))
    except BaseException as exc:
        atomic_json(output / "FAILURE.json", dict(error_type=type(exc).__name__, error=str(exc),
            completed=[row["mode"] for row in summaries], elapsed_seconds=time.monotonic() - started))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=18.)
    parser.add_argument("--capacity-kbit", type=int, default=256)
    args = parser.parse_args()
    with exclusive_run():
        asyncio.run(run(args.output, args.duration, args.capacity_kbit))


if __name__ == "__main__":
    main()
