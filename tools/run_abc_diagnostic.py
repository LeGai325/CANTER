"""Start/resume a detached campaign; logs and completion status outlive SSH."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','resume','status'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--protocol',type=Path,help='start only; resume uses the frozen PLAN.json')
    args=parser.parse_args()
    if args.protocol and args.action!='start':
        parser.error('--protocol is only accepted for start; resume uses the frozen plan')
    project=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    session='specnet-abc-'+hashlib.sha256(str(output).encode()).hexdigest()[:10]
    live=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
    if args.action=='status':
        status=json.loads((output/'STATUS.json').read_text()) if (output/'STATUS.json').exists() else {'state':'not_started'}
        print(json.dumps(dict(tmux_session=session,session_alive=live,**status),ensure_ascii=False,indent=2))
        return
    if live:
        raise SystemExit('campaign session already running: '+session)
    python=project/'.venv/bin/python'
    command=[str(python),'-m','specnet.diagnostic','--output',str(output)]
    if args.action=='start':
        if output.exists():raise SystemExit('output already exists; use resume or a new output path')
        options=['--protocol',str(args.protocol.resolve())] if args.protocol else []
        subprocess.run(command+options+['--plan-only'],cwd=project,check=True)
        with tarfile.open(output/'CODE_SNAPSHOT.tar.gz','w:gz') as archive:
            for name in ('specnet','tools','tests','configs','docs'):
                for path in sorted((project/name).rglob('*')):
                    if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                        archive.add(path,arcname=str(path.relative_to(project)),recursive=False)
            for name in ('README.md','.gitignore','pyproject.toml','requirements-experiment.txt'):
                archive.add(project/name,arcname=name,recursive=False)
    elif not (output/'PLAN.json').is_file():
        raise SystemExit('no frozen PLAN.json to resume')
    launch=output/'launcher'/str(time.time_ns())
    launch.mkdir(parents=True)
    log=launch/'run.log';exit_file=launch/'EXIT.json'
    script=launch/'run.sh'
    script.write_text('#!/usr/bin/env bash\nset -u\ncd -- '+shlex.quote(str(project))+'\n'+
        shlex.join(command+['--resume'])+' > '+shlex.quote(str(log))+' 2>&1\n'+
        'experiment_exit_status=$?\n'+
        'printf \'{"exit_code":%s}\\n\' "$experiment_exit_status" > '+shlex.quote(str(exit_file))+'\n'+
        'exit "$experiment_exit_status"\n')
    script.chmod(0o700)
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(project),shlex.join(['bash',str(script)])],check=True)
    print(json.dumps(dict(session=session,output=str(output),log=str(log),exit_file=str(exit_file)),indent=2))


if __name__=='__main__':main()
