"""Derived stage/background diagnostics from immutable run records, no model calls."""
import argparse
import json
from pathlib import Path
import re
import statistics


def background_observations(rows):
    samples=[]
    for row in rows:
        classes=re.split(r'(?=class htb )',row['classes_text'])
        block=next((b for b in classes if b.startswith('class htb 1:14 ')),None)
        queues=row['queues_text'].split('qdisc ')
        queue=next((b for b in queues if b.startswith('netem 14: ')),None)
        if not block or not queue:continue
        sent=re.search(r'Sent (\d+) bytes',block);backlog=re.search(r'backlog (\d+)b',queue)
        rate=re.search(r' rate ([\d.]+)([KM]?)bit',block)
        if sent and backlog and rate:
            samples.append(dict(at=row['at'],sent=int(sent[1]),backlog=int(backlog[1]),
                guarantee_Bps=float(rate[1])*{'':1,'K':1000,'M':1000000}[rate[2]]/8))
    intervals=[];longest=0.;zero_run=0.
    for a,b in zip(samples,samples[1:]):
        dt=b['at']-a['at'];n=b['sent']-a['sent']
        if a['backlog']>0 and b['backlog']>0 and dt>0 and n>=0:
            intervals.append(dict(seconds=dt,served_bytes=n,reference_bytes=a['guarantee_Bps']*dt))
            zero_run=zero_run+dt if n==0 else 0.;longest=max(longest,zero_run)
        else:zero_run=0.
    seconds=sum(r['seconds'] for r in intervals);served=sum(r['served_bytes'] for r in intervals);expected=sum(r['reference_bytes'] for r in intervals)
    return dict(sampled_backlogged_intervals=len(intervals),sampled_backlogged_seconds=seconds,
        throughput_Bps=served/seconds if seconds else None,aggregate_reference_service_deficit_bytes=max(0,expected-served),
        sum_interval_service_deficit_bytes=sum(max(0,r['reference_bytes']-r['served_bytes']) for r in intervals),
        longest_observed_zero_service_seconds=longest,
        note='Periodic tc samples: both endpoint backlogs positive. Queue may empty between samples; not a continuous-backlog guarantee. Shared mode has no separate BG reservation.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);reports=[]
    for row in json.loads((args.runs/'RUNS.json').read_text()):
        d=Path(row['directory']);events=json.loads((d/'events.json').read_text());results=json.loads((d/'results.json').read_text())
        telemetry=json.loads((d/'network/telemetry.json').read_text());packets=json.loads((d/'network/packet_counts.json').read_text());stages=[]
        for r in results:
            e=[v for v in events if v['task_id']==r['task_id']]
            first=lambda kind:next((v['at'] for v in e if v['kind']==kind),None)
            arrival,barrier,seal,generation=first('arrived'),first('baseline_barrier'),first('sealed'),first('generation_start')
            stages.append(dict(task_id=r['task_id'],baseline_seconds=barrier-arrival if barrier else None,
                optional_window_seconds=seal-barrier if seal and barrier else None,
                packing_and_generation_queue_seconds=generation-seal if generation and seal else None,
                generation_and_validation_seconds=r['delivered_at']-generation if r['delivered_at'] and generation else None,
                end_to_end_seconds=r['delivered_at']-arrival if r['delivered_at'] else None))
        decisions=json.loads((d/'decisions.json').read_text());changes=json.loads((d/'network_changes_applied.json').read_text());recovery=[]
        for change in changes:
            after=[v['snapshot']['at']-change['applied_at'] for v in decisions if v['action']=='ADMIT_NEXT' and v['snapshot']['at']>=change['applied_at']]
            recovery.append(dict(applied_at=change['applied_at'],rate_kbit=change['rate'],first_subsequent_admission_seconds=min(after) if after else None,
                note='First observed admission after intervention, not a counterfactual recovery guarantee.'))
        reports.append(dict(directory=str(d),benchmark=row['benchmark'],policy=row['policy'],mode=row['mode'],stages=stages,
            background=background_observations(telemetry),capacity_change_followup=recovery,
            all_link_bytes=packets.get('all_link_bytes'),non_role_link_bytes=packets.get('non_role_link_bytes'),
            cpu_scope='METRICS cpu_seconds counts this Python process including its threads; excludes helper/tcpdump/BPF/kernel/provider CPU'))
    (args.output/'DIAGNOSTICS.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(dict(runs=len(reports),output=str(args.output/'DIAGNOSTICS.json')),indent=2))
if __name__=='__main__':main()
