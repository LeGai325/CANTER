"""Recent, attributable network observations for native request admission.

Linux TCP_INFO's RTT is a sender-side smoothed estimate. Polling an unchanged
estimate is not a new observation: only acknowledged progress or a changed
estimate refreshes it. Client and content-server observations are identified
separately. No unavailable ECN/queue observation is imputed as zero.
"""
from collections import defaultdict, deque
import math
import socket
import struct
import time


def decode_tcp_info(raw):
    """Decode the stable Linux tcp_info prefix; optional counters need length.

    ECN means Linux's sender feedback counter, not a packet-capture CE ratio.
    The raw fields and negotiation bit are retained for measurement auditing.
    """
    if len(raw) < 104:
        raise ValueError("short Linux TCP_INFO")
    row = dict(rtt_seconds=struct.unpack_from("=I", raw, 68)[0] / 1e6,
               rtt_variance_seconds=struct.unpack_from("=I", raw, 72)[0] / 1e6,
               total_retransmissions=struct.unpack_from("=I", raw, 100)[0],
               ecn_negotiated=bool(raw[5] & 8), tcp_info_length=len(raw))
    if len(raw) >= 128:
        row["bytes_acked"] = struct.unpack_from("=Q", raw, 120)[0]
    if len(raw) >= 200:
        row["delivered"] = struct.unpack_from("=I", raw, 192)[0]
        row["delivered_ce"] = struct.unpack_from("=I", raw, 196)[0]
    return row


class RecentNetworkTelemetry:
    def __init__(self, *, window_seconds=30., max_age_seconds=3., max_samples=4096,
                 clock=time.monotonic):
        if not 0 < max_age_seconds <= window_seconds or max_samples < 1:
            raise ValueError("invalid telemetry window/age/sample cap")
        self.window_seconds = float(window_seconds)
        self.max_age_seconds = float(max_age_seconds)
        self.max_samples = max_samples
        self.clock = clock
        self.rtt = defaultdict(lambda: deque(maxlen=max_samples))
        self.ecn = defaultdict(lambda: deque(maxlen=max_samples))
        self.connections = {}
        self.observation_counts = dict(rtt=0, ecn=0, duplicate_rtt=0, unavailable=0)

    def observe_socket(self, sock, *, path_id, role, connection_id, source, now=None):
        now = self.clock() if now is None else now
        try:
            raw = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 256)
            info = decode_tcp_info(raw)
        except (AttributeError, OSError, ValueError, struct.error) as exc:
            self.observation_counts["unavailable"] += 1
            return dict(at=now, valid=False, reason=type(exc).__name__, source=source)
        return self.observe(info, path_id=path_id, role=role,
                            connection_id=connection_id, source=source, now=now)

    def observe(self, info, *, path_id, role, connection_id, source, now=None):
        now = self.clock() if now is None else now
        if not math.isfinite(now):
            raise ValueError("invalid observation time")
        key = (path_id, role)
        old = self.connections.get(connection_id)
        signature = tuple(info.get(k) for k in
                          ("bytes_acked", "delivered", "rtt_seconds", "rtt_variance_seconds"))
        rtt = info.get("rtt_seconds")
        valid = isinstance(rtt, (int, float)) and math.isfinite(rtt) and rtt > 0
        fresh = valid and (old is None or signature != old["signature"])
        row = dict(at=now, path_id=path_id, role=role, connection_id=connection_id,
                   source=source, **info, valid=valid, new_rtt_observation=fresh,
                   ecn_observation=False)
        if fresh:
            self.rtt[key].append(dict(at=now, seconds=rtt, source=source,
                                      connection_id=connection_id))
            self.observation_counts["rtt"] += 1
        elif valid:
            self.observation_counts["duplicate_rtt"] += 1
        if old and info.get("ecn_negotiated") and old["info"].get("ecn_negotiated"):
            previous = old["info"]
            fields = ("delivered", "delivered_ce")
            if all(type(value.get(k)) is int for value in (info, previous) for k in fields):
                delivered = info["delivered"] - previous["delivered"]
                ce = info["delivered_ce"] - previous["delivered_ce"]
                # Counter reset/wrap, absent traffic, and inconsistent counters
                # supply no fraction; they never supply an inferred zero.
                if delivered > 0 and 0 <= ce <= delivered:
                    self.ecn[key].append(dict(at=now, delivered=delivered, ce=ce,
                                             source=source, connection_id=connection_id))
                    row.update(ecn_observation=True, ecn_delivered_delta=delivered,
                               ecn_ce_delta=ce)
                    self.observation_counts["ecn"] += 1
        self.connections[connection_id] = dict(signature=signature, info=dict(info))
        return row

    def forget_connection(self, connection_id):
        self.connections.pop(connection_id, None)

    def _recent(self, history, now):
        # Reject future timestamps too; do not silently relabel an old sample
        # with the current decision time.
        return [row for row in history if 0 <= now - row["at"] <= self.window_seconds]

    def snapshot(self, *, path_id, role, now=None):
        now = self.clock() if now is None else now
        key = (path_id, role)
        recent = self._recent(self.rtt.get(key, ()), now)
        rtt = dict(value=None, valid=False, timestamp=None, identity=f"{path_id}/{role}",
                   reason="no_recent_rtt", sample_count=len(recent),
                   window_seconds=self.window_seconds)
        if recent:
            mean = sum(row["seconds"] for row in recent) / len(recent)
            minimum = min(row["seconds"] for row in recent)
            stamp = max(row["at"] for row in recent)
            valid = now - stamp <= self.max_age_seconds
            rtt.update(value=max(0., (mean - minimum) / mean) if valid else None,
                       mean_seconds=mean, minimum_seconds=minimum, timestamp=stamp,
                       valid=valid, reason=None if valid else "stale_rtt",
                       sources=sorted({row["source"] for row in recent}))
        recent_ecn = self._recent(self.ecn.get(key, ()), now)
        ecn = dict(value=None, valid=False, timestamp=None, identity=f"{path_id}/{role}",
                   reason="no_verified_ecn_feedback", sample_count=len(recent_ecn),
                   measurement="Linux sender delivered_ce / delivered deltas after ECN negotiation")
        if recent_ecn:
            stamp = max(row["at"] for row in recent_ecn)
            delivered = sum(row["delivered"] for row in recent_ecn)
            ce = sum(row["ce"] for row in recent_ecn)
            valid = now - stamp <= self.max_age_seconds
            ecn.update(value=ce / delivered if valid else None, valid=valid, timestamp=stamp,
                       delivered=delivered, delivered_ce=ce,
                       reason=None if valid else "stale_ecn_feedback")
        return dict(rtt_excess=rtt, ecn=ecn)


def queue_observation(sample, *, role, path_id, normalization_bytes, now,
                      max_age_seconds):
    if normalization_bytes <= 0:
        raise ValueError("queue normalization must be positive")
    sample = sample or {}
    name = (sample.get("role_to_queue") or {}).get(role)
    entry = (sample.get("queue_by_class") or {}).get(name) or {}
    stamp = sample.get("observed_at")
    count = entry.get("queue_bytes")
    valid = (bool(sample.get("queue_valid")) and name is not None
             and isinstance(stamp, (int, float)) and math.isfinite(stamp)
             and 0 <= now - stamp <= max_age_seconds
             and type(count) is int and count >= 0)
    return dict(value=min(1., count / normalization_bytes) if valid else None,
                valid=bool(valid), timestamp=stamp, identity=f"{path_id}/{name}",
                bytes=count, normalization_bytes=normalization_bytes,
                reason=None if valid else "missing_invalid_or_stale_queue")
