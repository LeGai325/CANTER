"""Offline checks of recorded real runs; never calls a model or alters runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specnet.run_validation import verify_output_limit_failure


def verified_model_call(call, directory, results):
    if (call.get('status') == 'failed' and call.get('reason') == 'response_not_complete'
            and call.get('finish_reason') == 'length'):
        try:
            verify_output_limit_failure(call, directory, 'glm-5.2', results)
            return True
        except ValueError:
            return False
    return ((call.get('status')=='refused' and call.get('reason')=='provider_content_refusal'
             and call.get('http_status')==400 and call.get('remote_terminal_confirmed'))
            or (call.get('status')=='succeeded' and call.get('response_model')=='glm-5.2'
                and call.get('remote_terminal_confirmed')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    calibration=json.loads((args.calibration/'CALIBRATION.json').read_text());reports=[]
    for item in json.loads((args.runs/'RUNS.json').read_text()):
        directory=Path(item['directory']);m=json.loads((directory/'MANIFEST.json').read_text())
        results=json.loads((directory/'results.json').read_text());jobs=json.loads((directory/'jobs.json').read_text())
        events=json.loads((directory/'events.json').read_text());calls=json.loads((directory/'provider_calls.json').read_text())
        contracts=json.loads((directory/'contracts.json').read_text());inputs=json.loads((directory/'packed_inputs.json').read_text())
        packet=json.loads((directory/'network/packet_counts.json').read_text());capturelog=(directory/'network/traffic.log').read_text()
        seals={e['task_id']:e for e in events if e['kind']=='sealed'};arrivals={e['task_id']:e for e in events if e['kind']=='arrived'}
        checks={}
        checks['checksum_files']=True
        for line in (directory/'SHA256SUMS').read_text().splitlines():
            expected,filename=line.split('  ',1)
            checks['checksum_files'] &= hashlib.sha256((directory/filename).read_bytes()).hexdigest()==expected
        checks['complete_arrival_rows']={r['task_id'] for r in results}==set(m['task_ids']) and len(results)==len(m['task_ids'])
        checks['calibration_test_task_disjoint']=not(set(m['task_ids'])&set(calibration[m['benchmark']]['calibration_task_ids']))
        checks['all_jobs_terminated']=all(j['state']=='TERMINAL' and j['reserved']==0 for j in jobs)
        checks['payload_conserved']=all(r['all_body_payload_bytes']==sum(j['spent'] for j in jobs if j['task_id']==r['task_id']) for r in results)
        checks['optional_count_conserved']=all(r['optional_admitted']==sum(j['task_id']==r['task_id'] and j['role']=='Q3' for j in jobs) for r in results)
        checks['all_included_accepted']=all(set(r['included_ids'])<=set(r['accepted_ids']) for r in results)
        checks['all_cited_included']=all(set(r['cited_ids'])<=set(r['included_ids']) for r in results)
        checks['one_generation_per_task']=len({c['purpose'] for c in calls})==len(calls)
        checks['returned_model_verified']=all(verified_model_call(c,directory,results) for c in calls)
        checks['provider_caps']=all(r['provider_input_tokens']<=c['input_tokens'] and r['output_tokens']<=c['output_tokens'] for r,c in zip(results,contracts) if r['delivered_at'])
        checks['inputs_frozen_before_cutoff']=all(e['accepted_at']<arrivals[task]['cutoff'] for task,s in seals.items() for e in s['evidence'])
        checks['prompt_hashes']=all(hashlib.sha256(row['prompt'].encode()).hexdigest()==row['sha256'] for row in inputs.values())
        checks['gold_not_in_prompt_structure']=all(set(json.loads(row['prompt']))=={'instruction','question','documents'} for row in inputs.values())
        checks['deadline_reserve_fixed']=all(abs(arrivals[c['task_id']]['deadline']-arrivals[c['task_id']]['cutoff']-c['final_reserve_seconds'])<1e-7 for c in contracts)
        checks['wire_conserved']=sum(r['wire_bytes'] for r in results)==sum(v['wire_bytes'] for role,v in packet['roles'].items() if role!='BG')
        counts=re.search(r'(\d+) packets captured\n(\d+) packets received by filter\n(\d+) packets dropped by kernel',capturelog)
        checks['packet_capture_complete']=bool(counts and counts[1]==counts[2] and counts[3]=='0')
        q=json.loads((directory/'QUEUE_METRICS.json').read_text())
        checks['queue_probe_no_loss']=q['packets']>0 and not q['bpf_log'].strip() and not q.get('unparsed_lines',[])
        reports.append(dict(directory=str(directory),passed=all(checks.values()),checks=checks))
    report=dict(passed=all(r['passed'] for r in reports),runs=len(reports),checks=sum(len(r['checks']) for r in reports),reports=reports)
    (args.output/'VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
