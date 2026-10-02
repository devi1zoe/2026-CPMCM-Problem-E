"""Run remaining analyses only after full validation is available."""
from common import *
import subprocess,time

def run(script,*args,gpu=None):
    env=os.environ.copy()
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=ROOT/f'logs/final_{Path(script).stem}.log';print('Starting',script,*args,flush=True)
    with log.open('w') as stream:
        subprocess.run([sys.executable,str(ROOT/script),*args],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('Completed',script,flush=True)

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    required=['old_gated',*protocol['configs'],'expert_text']
    print('Waiting for completed full validation of all four groups',flush=True)
    while not all((ROOT/f'results/evaluated_valid_{n}.json').exists() for n in required):time.sleep(10)
    run('summarize.py');run('compare.py')
    decision=json.loads((ROOT/'combination_lock.json').read_text())
    if decision['run']:
        jobs=[]
        for gpu,seed in enumerate(protocol['seeds'],1):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            stream=(ROOT/f'logs/combined_{seed}.log').open('w')
            jobs.append((subprocess.Popen([sys.executable,str(ROOT/'train_text.py'),'--seed',str(seed),'--names','combined'],env=env,stdout=stream,stderr=subprocess.STDOUT),stream))
        for process,stream in jobs:
            assert process.wait()==0;stream.close()
        run('evaluate.py','--names','combined',gpu=0);run('summarize.py');run('compare.py')
    run('diagnostics.py')
    while not (ROOT/'results/reconstruction_review_complete.json').exists():time.sleep(10)
    run('select_final.py');lock=json.loads((ROOT/'selection_lock.json').read_text())
    run('evaluate.py','--split','test','--names',*lock['final_test_models'],gpu=0)
    run('infer_special.py',gpu=1);run('report.py')
    save_json(ROOT/'results/finalization_ready.json',dict(report_ready=True,manual_plot_review_pending=True,completion_audit_pending=True))
    print('Report ready for visual inspection and final completion audit',flush=True)

if __name__=='__main__':main()
