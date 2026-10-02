"""Queue full validation after, and only after, all three fits finish."""
from common import *
import subprocess,time

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());names=list(protocol['configs'])+['expert_text']
    # The already-started historical comparison owns GPU0 until both files exist.
    while not all((ROOT/f'results/evaluated_valid_{n}.json').exists() for n in ['shape_single','old_gated']):time.sleep(10)
    while True:
        pending=[n for n in names if not (ROOT/f'results/evaluated_valid_{n}.json').exists()]
        if not pending:print('All four groups have full validation results',flush=True);return
        ready=[]
        for n in pending:
            prefix='oof_completed' if n=='expert_text' else f'completed_{n}'
            if all((ROOT/f'results/{prefix}_{s}.json').exists() for s in protocol['seeds']):ready.append(n)
        if not ready:time.sleep(10);continue
        adapted=[n for n in ready if protocol['configs'].get(n,{}).get('lora')]
        ordinary=[n for n in ready if n not in adapted]
        chosen=ordinary[:3] if ordinary else adapted[:1]
        log=ROOT/('logs/evaluate_'+'_'.join(chosen)+'.log');print('Evaluating',chosen,flush=True)
        with log.open('w') as stream:
            subprocess.run([sys.executable,str(ROOT/'evaluate.py'),'--names',*chosen],stdout=stream,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':main()
