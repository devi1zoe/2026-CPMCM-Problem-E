"""Resume-safe local process orchestration on designated idle GPUs. No package/environment changes."""
from common import *
import subprocess,time,concurrent.futures,argparse,queue

PYTHON='/usr/local/python3.10/bin/python3'

def command(script,args,gpu,log):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
    with open(ROOT/'logs'/log,'a',buffering=1) as f:
        result=subprocess.run([PYTHON,'-u',str(ROOT/script),*args],cwd=PROJECT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{script} failed ({result.returncode}); see {log}')

def fits(names):
    # GPU 3 was claimed by an unrelated job after initial inspection; use only GPU 2.
    tasks=[]
    for seed in [17,29,43]:
        for chunk in [names[:(len(names)+1)//2],names[(len(names)+1)//2:]]:
            if chunk:tasks.append((seed,chunk))
    pending=queue.Queue()
    for item in tasks:pending.put(item)
    def worker(index):
        gpu=2
        while True:
            try:seed,chunk=pending.get_nowait()
            except queue.Empty:return
            command('train.py',['--seed',str(seed),'--names',*chunk],gpu,f'train_{seed}_{chunk[0]}.log')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for f in [pool.submit(worker,i) for i in range(4)]:f.result()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pilot',action='store_true');args=ap.parse_args()
    if args.wait_pilot:
        while not (ROOT/'regression_weight_lock.json').exists():time.sleep(5)
    else:command('train.py',['--pilot'],2,'pilot.log')
    protocol=json.loads((ROOT/'protocol.json').read_text());names=list(protocol['configs'])
    save_json(ROOT/'results/run_status.json',dict(stage='main_training'))
    fits(names)
    save_json(ROOT/'results/run_status.json',dict(stage='full_validation'))
    # Use separate model groups; cache locks ensure identical shared corruptions.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(command,'evaluate.py',['--names',*names[:5]],2,'valid_group1.log'),
               pool.submit(command,'evaluate.py',['--names',*names[5:],'old_gated'],2,'valid_group2.log')]
        for f in tasks:f.result()
    command('selection.py',[],2,'selection.log')
    ablations=list(json.loads((ROOT/'ablation_configs.json').read_text()))
    if ablations:
        save_json(ROOT/'results/run_status.json',dict(stage='leave_one_out',models=ablations));fits(ablations)
        command('evaluate.py',['--names',*ablations],2,'valid_ablations.log')
    command('selection.py',['--final'],2,'selection.log')
    save_json(ROOT/'results/run_status.json',dict(stage='locked_final_evaluation'))
    lock=json.loads((ROOT/'selection_lock.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(command,'evaluate.py',['--split','test','--names',*lock['final_test_models']],2,'test_final.log'),
               pool.submit(command,'bootstrap.py',[],2,'bootstrap.log')]
        for f in tasks:f.result()
    command('infer.py',[],2,'special.log')
    command('report.py',[],2,'report.log')
    command('package.py',[],2,'package.log')
    save_json(ROOT/'results/run_status.json',dict(stage='complete',selected=lock['selected']))

if __name__=='__main__':main()
