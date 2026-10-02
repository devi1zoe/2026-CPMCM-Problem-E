"""Selection precedes conditional test/special inference; rejected runs stop at report."""
from common import *
import subprocess


def run(script,*args):
    save_json(ROOT/'results/status.json',dict(stage=script,pid=os.getpid()))
    with (ROOT/'logs'/f'final_{script}.log').open('a') as f:
        subprocess.run([sys.executable,str(ROOT/script),*args],cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    for name in ['dual_gate','late_fusion']:
        proof=json.loads((ROOT/f'results/evaluated_valid_{name}.json').read_text());assert proof['protocol_sha256']==sha256(ROOT/'protocol.json')
    run('selection.py')
    if not (ROOT/'results/paired_cluster_bootstrap.json').exists():run('bootstrap.py')
    lock=json.loads((ROOT/'selection_lock.json').read_text())
    if lock['accepted_improvement']:
        run('evaluate.py','--split','test','--names',lock['selected']);run('infer.py')
    run('report.py')
    if lock['accepted_improvement']:run('package.py')
    run('audit.py')
    save_json(ROOT/'results/status.json',dict(stage='complete',selected=lock['selected'],accepted_improvement=lock['accepted_improvement']))


if __name__=='__main__':main()
