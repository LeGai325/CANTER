"""Offline TCP incarnation accounting after all task execution has stopped.

Port pairs can be reused. Split those pairs by SYN sequence numbers and check
both sequence spaces; acknowledged FIN endpoints tighten the conservative
sequence bounds. Never assign a delayed packet solely by its timestamp.
This parser is intentionally limited to the controlled IPv4 service (one HTTP
exchange per connection, at most 8 MiB body). Ambiguity remains a hard error.
"""
from collections import defaultdict
import json
from pathlib import Path
import socket
import struct


VERSION = 'tcp-incarnation-offline-v2'
SEQUENCE_SPAN = 16 * 1024 * 1024
MASK = (1 << 32) - 1


def conflicting_packets(path, manifest, selected):
    packets = defaultdict(list)
    with Path(path).open('rb') as stream:
        header = stream.read(24)
        if len(header) != 24:
            raise ValueError('missing pcap header')
        endian = '<' if header[:4] in (b'\xd4\xc3\xb2\xa1', b'\x4d\x3c\xb2\xa1') else '>'
        if struct.unpack(endian+'I', header[20:24])[0] != 1:
            raise ValueError('expected Ethernet capture')
        ordinal = 0
        while raw := stream.read(16):
            if len(raw) != 16:
                raise ValueError('truncated pcap record')
            _, _, captured, original = struct.unpack(endian+'IIII', raw)
            data = stream.read(captured)
            if len(data) != captured:
                raise ValueError('truncated pcap packet')
            ordinal += 1
            if len(data) < 54 or data[12:14] != b'\x08\x00' or data[23] != 6:
                continue
            offset = 14 + (data[14] & 15)*4
            source, destination = socket.inet_ntoa(data[26:30]), socket.inet_ntoa(data[30:34])
            source_port, destination_port = struct.unpack('!HH', data[offset:offset+4])
            outgoing = source == manifest['server_ip']
            port, client_port = (source_port, destination_port) if outgoing else (destination_port, source_port)
            key = f'{port}:{client_port}'
            if key not in selected:
                continue
            if ({source, destination} != {manifest['host_ip'], manifest['server_ip']}
                    or len(data) < offset+20
                    or struct.unpack('!H', data[20:22])[0] & 0x3fff):
                raise ValueError('unsupported TCP attribution packet')
            sequence, ack = struct.unpack('!II', data[offset+4:offset+12])
            ip_size = struct.unpack('!H', data[16:18])[0]
            payload = ip_size - (offset-14) - (data[offset+12] >> 4)*4
            if payload < 0:
                raise ValueError('invalid TCP payload length')
            packets[key].append(dict(ordinal=ordinal, outgoing=outgoing, seq=sequence,
                ack=ack, flags=data[offset+13], payload=payload, wire_bytes=original))
    return packets


def _packet_matches(packet, session):
    """Conservative TCP sequence-space membership, never timestamp proximity."""
    if packet['ordinal'] < session['first_packet']:
        return False
    direction = 'server' if packet['outgoing'] else 'client'
    peer = 'client' if packet['outgoing'] else 'server'
    own, other = session[direction+'_isn'], session[peer+'_isn']
    if packet['flags'] & 2:
        return (packet['seq'] == own and
                (not packet['flags'] & 16 or packet['ack'] == ((other+1) & MASK)))
    offset = (packet['seq']-own) & MASK
    consumed = packet['payload'] + int(bool(packet['flags'] & 1))
    if offset + consumed >= SEQUENCE_SPAN:
        return False
    bound = session['sequence_spaces'][direction]['bounded_end_offset']
    if bound is not None:
        # FIN consumes its own sequence number. Payload may not use that byte;
        # an ACK after FIN can use the next sequence number (the bound itself).
        if offset + consumed > bound:
            return False
        if packet['payload'] and offset + packet['payload'] >= bound:
            return False
        if packet['flags'] & 1 and offset + consumed != bound:
            return False
    if packet['flags'] & 16:
        ack_offset = (packet['ack']-other) & MASK
        peer_bound = session['sequence_spaces'][peer]['bounded_end_offset']
        if ack_offset >= SEQUENCE_SPAN or (peer_bound is not None and ack_offset > peer_bound):
            return False
    return True


def _prove_sequence_bounds(session, assigned):
    """Only a uniquely attributable FIN plus its peer's ACK supplies an end.

    Observed data/ACK maxima are lower bounds, not proof that no further data
    exists. In particular RST, HTTP completion and a subsequent SYN cannot turn
    those observed maxima into upper bounds. Incomplete captures stay open.
    """
    changed = False
    for direction, outgoing in [('client', False), ('server', True)]:
        own = session[direction+'_isn']
        sent = [p for p in assigned if p['outgoing'] == outgoing]
        acknowledged = [((p['ack']-own) & MASK) for p in assigned
                        if p['outgoing'] != outgoing and p['flags'] & 16]
        ends = {((p['seq']-own) & MASK) + p['payload'] + 1
                for p in sent if p['flags'] & 1}
        if len(ends) > 1:
            raise ValueError('inconsistent TCP FIN sequence')
        state = session['sequence_spaces'][direction]
        state['observed_end_offset'] = max(
            [((p['seq']-own) & MASK) + p['payload']
             + int(bool(p['flags'] & 1)) + int(bool(p['flags'] & 2)) for p in sent], default=0)
        state['observed_ack_offset'] = max(acknowledged, default=0)
        state['fin_end_offset'] = next(iter(ends), None)
        state['fin_acknowledged'] = bool(ends and state['fin_end_offset'] in acknowledged)
        if state['fin_acknowledged'] and state['bounded_end_offset'] is None:
            state['bounded_end_offset'] = state['fin_end_offset']
            changed = True
    session['close_proven'] = all(s['fin_acknowledged'] for s in session['sequence_spaces'].values())
    session['reset_observed'] = any(p['flags'] & 4 for p in assigned)
    return changed


def split_incarnations(packets, records, expected):
    sessions = {}
    for packet in packets:
        if packet['flags'] & 2 and (packet['payload'] or packet['flags'] & 5):
            raise ValueError('unsupported TCP SYN payload or flags')
        if not packet['outgoing'] and packet['flags'] & 2 and not packet['flags'] & 16:
            sessions.setdefault(packet['seq'], dict(client_isn=packet['seq'], server_isn=None,
                first_packet=packet['ordinal'], wire_bytes=0, packets=0,
                sequence_spaces={side: dict(bounded_end_offset=None) for side in ('client', 'server')}))
    for packet in packets:
        if packet['outgoing'] and packet['flags'] & 2 and packet['flags'] & 16:
            session = sessions.get((packet['ack']-1) & MASK)
            if session is None or session['server_isn'] not in (None, packet['seq']):
                raise ValueError('unmatched TCP handshake')
            session['server_isn'] = packet['seq']
    ordered = sorted(sessions.values(), key=lambda s: s['first_packet'])
    records = sorted(records, key=lambda r: r['started_at'])
    if len(ordered) != len(records) or any(s['server_isn'] is None for s in ordered):
        raise ValueError('TCP incarnation/HTTP record count mismatch')
    if any(a['finished_at'] > b['started_at'] for a, b in zip(records, records[1:])):
        raise ValueError('overlapping HTTP records cannot be ordered safely')

    # Unique assignments can prove finite sequence spaces, which may resolve
    # other packets on a reused port. Each iteration only removes candidates;
    # no packet with multiple plausible incarnations is guessed into a session.
    while True:
        candidates = [[i for i, session in enumerate(ordered) if _packet_matches(packet, session)]
                      for packet in packets]
        if any(not matches for matches in candidates):
            raise ValueError('TCP packet has no consistent incarnation')
        assigned = [[] for _ in ordered]
        for packet, matches in zip(packets, candidates):
            if len(matches) == 1:
                assigned[matches[0]].append(packet)
        changed = False
        for session, unique in zip(ordered, assigned):
            changed |= _prove_sequence_bounds(session, unique)
        if not changed:
            break
    if any(len(matches) != 1 for matches in candidates):
        raise ValueError('TCP packet has ambiguous incarnation')
    for packet, matches in zip(packets, candidates):
        session = ordered[matches[0]]
        session['wire_bytes'] += packet['wire_bytes']
        session['packets'] += 1
    if (sum(s['wire_bytes'] for s in ordered) != expected['wire_bytes']
            or sum(s['packets'] for s in ordered) != expected['packets']):
        raise ValueError('TCP incarnation byte conservation failed')
    return [dict(**session, task_id=record['task_id'], kind=record['kind'],
                 job_id=record.get('job_id')) for session, record in zip(ordered, records)]


def attribute_wire_bytes(directory, packet, records, results):
    grouped = defaultdict(list)
    for record in records:
        if 'flow_id' in record:
            grouped[record['flow_id']].append(record)
    repeated = {flow for flow, rows in grouped.items() if len(rows) > 1}
    directory = Path(directory)
    parsed = {}
    if repeated:
        manifest = json.loads((directory/'network/MANIFEST.json').read_text())
        parsed = conflicting_packets(directory/'network/traffic.pcap', manifest, repeated)
    per_task, search = defaultdict(int), defaultdict(int)
    repaired = {}
    for flow, rows in grouped.items():
        counts = packet['flows'].get(flow, {'wire_bytes': 0, 'packets': 0})
        if flow in repeated:
            allocations = split_incarnations(parsed.get(flow, []), rows, counts)
            repaired[flow] = allocations
        else:
            allocations = [dict(task_id=rows[0]['task_id'], kind=rows[0]['kind'], wire_bytes=counts['wire_bytes'])]
        for allocation in allocations:
            per_task[allocation['task_id']] += allocation['wire_bytes']
            if allocation['kind'] == 'search':
                search[allocation['task_id']] += allocation['wire_bytes']
    for result in results:
        result.wire_bytes = per_task[result.task_id]
        result.search_network_bytes = search[result.task_id]
    return dict(version=VERSION, repeated_port_pairs=repaired,
                assigned_wire_bytes=sum(per_task.values()),
                agent_wire_bytes=sum(v['wire_bytes'] for role, v in packet['roles'].items() if role != 'BG'),
                scope='offline attribution only; captured lengths, task execution and aggregate network metrics unchanged')
