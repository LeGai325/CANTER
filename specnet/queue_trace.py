"""Read-only kernel probes restricted to one experiment veth name."""
import asyncio
import json
from pathlib import Path
import re
import signal
import subprocess
import time


class QueueTrace:
    def __init__(self,interface,directory):
        if not re.fullmatch(r'sn[0-9a-f]{10}s',interface):raise ValueError('experiment server veth required')
        self.interface,self.directory=interface,Path(directory)
    async def __aenter__(self):
        self.directory.mkdir(parents=True,exist_ok=False)
        program='''
        BEGIN { printf("READY\\n"); }
        tracepoint:net:net_dev_queue /str(args->name) == "INTERFACE"/ {
          $skb = (struct sk_buff *)args->skbaddr;
          $port = *(uint16 *)($skb->head + $skb->transport_header);
          @port[args->skbaddr] = (($port & 255) << 8) | ($port >> 8);
          @entered[args->skbaddr] = nsecs;
        }
        tracepoint:net:net_dev_start_xmit /str(args->name) == "INTERFACE"/ {
          $t = @entered[args->skbaddr];
          if ($t) {
            printf("{\\"port\\":%d,\\"protocol\\":%d,\\"gso_segs\\":%d,\\"queued_ns\\":%llu,\\"sent_ns\\":%llu,\\"wait_ns\\":%llu}\\n", @port[args->skbaddr],args->protocol,args->gso_segs,$t,nsecs,nsecs-$t);
            delete(@entered[args->skbaddr]); delete(@port[args->skbaddr]);
          }
        }
        END { clear(@entered); clear(@port); }
        '''.replace('INTERFACE',self.interface)
        (self.directory/'probe.bt').write_text(program)
        self.out=(self.directory/'packets.jsonl').open('w');self.err=(self.directory/'probe.log').open('w')
        self.process=subprocess.Popen(['sudo','-n','bpftrace','-q',str(self.directory/'probe.bt')],stdout=self.out,stderr=self.err)
        for _ in range(100):
            await asyncio.sleep(.1)
            if self.process.poll() is not None:raise RuntimeError('queue probe failed; inspect probe.log')
            if (self.directory/'packets.jsonl').read_text().startswith('READY'):return self
        await self.__aexit__();raise RuntimeError('queue probe startup timed out')
    async def __aexit__(self,*args):
        if self.process.poll() is None:
            try:self.process.send_signal(signal.SIGINT)
            except PermissionError:subprocess.run(['sudo','-n','kill','-INT',str(self.process.pid)],check=True)
            await asyncio.to_thread(self.process.wait,15)
        self.out.close();self.err.close()
    def report(self):
        from .cost_model import quantile
        rows=[];unparsed=[];other=0
        for line in (self.directory/'packets.jsonl').read_text().splitlines():
            if line.startswith('{'):
                row=json.loads(line)
                if 19000 <= row['port'] <= 19004 and row['protocol'] in (8,2048):rows.append(row)
                else:other+=1
            elif line.strip() and line!='READY':unparsed.append(line)
        result={}
        for port in sorted({r['port'] for r in rows}):
            durations=[r['wait_ns']/1e9 for r in rows if r['port']==port]
            result[str(port)]=dict(packets=len(durations),queue_to_xmit_p50_seconds=quantile(durations,.5),queue_to_xmit_p99_seconds=quantile(durations,.99))
        return dict(measurement='per skb (not post-segmentation packet): net_dev_queue to net_dev_start_xmit, server veth TCP only; includes HTB/netem residence',ports=result,
            bpf_log=(self.directory/'probe.log').read_text(),unparsed_lines=unparsed,packets=len(rows),non_role_frames=other)
