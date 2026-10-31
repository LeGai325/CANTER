"""E4 fixed-transfer queue isolation: same traffic, roles, release times and ports."""
import argparse
import asyncio
import json
from pathlib import Path
import socket
import time

from .cost_model import quantile
from .experiment import dump,digest,exclusive_run
from .network import NetworkLab
from .queue_trace import QueueTrace


def traffic():
    items=[dict(id=f'control-{i}',role='Q0',bytes=256,release=.1+i*.04,client_port=24000+i) for i in range(24)]
    items += [dict(id='base-0',role='Q1',bytes=524288,release=0,client_port=24100),
              dict(id='base-1',role='Q1',bytes=524288,release=.2,client_port=24101),
              dict(id='optional-0',role='Q3',bytes=1048576,release=0,client_port=24102),
              dict(id='background-0',role='BG',bytes=1048576,release=0,client_port=24103)]
    return items


async def run(directory,mode,rate,trace=True):
    directory.mkdir(parents=True,exist_ok=False);items=traffic();dump(directory/'TRAFFIC.json',items)
    records=[]
    async with NetworkLab(directory/'network') as lab:
        await lab.configure(mode=mode,capacity_kbit=rate,delay_ms=8,minimum_kbit=1000)
        handlers=set()
        async def serve(reader,writer):
            handlers.add(asyncio.current_task())
            try:
                raw=await reader.readline();size=int(raw)
                writer.transport.set_write_buffer_limits(high=16384,low=4096)
                for start in range(0,size,4096):writer.write(b'x'*min(4096,size-start));await writer.drain()
            finally:
                writer.close();await writer.wait_closed();handlers.discard(asyncio.current_task())
        servers=[await asyncio.start_server(serve,sock=s) for s in lab.sockets.values()]
        probe=QueueTrace(lab.manifest['server_interface'],directory/'queue_trace') if trace else None
        if probe:await probe.__aenter__()
        origin=time.monotonic()
        async def get(item):
            await asyncio.sleep(max(0,origin+item['release']-time.monotonic()))
            start=time.monotonic();role=item['role'];r,w=await asyncio.open_connection(lab.manifest['server_ip'],lab.manifest['ports'][role],
                local_addr=(lab.manifest['host_ip'],item['client_port']))
            try:
                w.write((str(item['bytes'])+'\n').encode());await w.drain();data=await r.readexactly(item['bytes'])
                records.append(dict(**item,started_at=start,finished_at=time.monotonic(),fct_seconds=time.monotonic()-start,received_bytes=len(data)))
            finally:w.close();await w.wait_closed()
        try:await asyncio.wait_for(asyncio.gather(*(get(i) for i in items)),120)
        finally:
            for server in servers:server.close()
            await asyncio.gather(*(s.wait_closed() for s in servers))
            await asyncio.gather(*list(handlers),return_exceptions=True)
            if probe:await probe.__aexit__()
        await lab.sample()
    summary=dict(mode=mode,capacity_kbit=rate,traffic_sha256=digest(items),critical_guaranteed_share=.50,
        roles={role:dict(completed=sum(r['role']==role for r in records),fct_p99_seconds=quantile([r['fct_seconds'] for r in records if r['role']==role],.99)) for role in ('Q0','Q1','Q3','BG')},
        queue_trace=probe.report() if probe else None)
    dump(directory/'RESULTS.json',records);dump(directory/'METRICS.json',summary)
    if probe and (not summary['queue_trace']['packets'] or summary['queue_trace']['bpf_log'].strip() or summary['queue_trace']['unparsed_lines']):
        dump(directory/'STATUS.json',dict(state='failed',reason='queue_probe_incomplete'))
        raise RuntimeError('queue probe incomplete')
    dump(directory/'STATUS.json',dict(state='completed',all_transfers_completed=len(records)==len(items)))
    return summary


async def main_async(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    results=[await run(root/mode,mode,args.capacity_kbit,not args.no_trace) for mode in ('qos','merged')]
    dump(root/'COMPARISON.json',results)
    print(json.dumps(results,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--capacity-kbit',type=int,default=2000);p.add_argument('--no-trace',action='store_true')
    with exclusive_run():asyncio.run(main_async(p.parse_args()))
if __name__=='__main__':main()
