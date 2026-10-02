"""Lock final choice using completed validation and unchanged guardrails only."""
from common import *
from engine import checkpoint_paths
import pandas as pd

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());combination=json.loads((ROOT/'combination_lock.json').read_text())
    names=list(protocol['configs'])+(['combined'] if combination['run'] else [])
    for name in ['old_gated',*names]:
        proof=json.loads((ROOT/f'results/evaluated_valid_{name}.json').read_text());assert proof['protocol_sha256']==sha256(ROOT/'protocol.json')
        assert all(sha256(PROJECT/p)==h for p,h in proof['checkpoints'].items())
    table=pd.read_csv(ROOT/'results/selection_comparison.csv').set_index('model')
    eligible=[n for n in names if bool(table.loc[n,'eligible'])]
    if eligible:
        best=min(float(table.loc[n,'J']) for n in eligible);tied=[n for n in eligible if float(table.loc[n,'J'])<best+.002]
        selected=min(tied,key=lambda n:(json.loads((ROOT/f'results/six_{n}.json').read_text())['parameters'],float(table.loc[n,'J'])))
    else:selected='old_gated'
    challenger=min(names,key=lambda n:float(table.loc[n,'J']))
    final_models=list(dict.fromkeys([selected,challenger]))
    hashes={n:{str(p.relative_to(PROJECT)):sha256(p) for p in checkpoint_paths(n)} for n in final_models}
    lock=dict(selected=selected,challenger=challenger,eligible=eligible,final_test_models=final_models,checkpoint_hashes=hashes,
        protocol_sha256=sha256(ROOT/'protocol.json'),rules_sha256=sha256(ROOT/'selection_rules.json'),combination_sha256=sha256(ROOT/'combination_lock.json'),
        validation_predictions={n:sha256(ROOT/f'results/valid_{n}_predictions.npz') for n in ['old_gated',*names]},
        previously_evaluated_test=True,reason='Passed all seven guardrails' if eligible else 'No new candidate passed all seven original guardrails; retain historical model')
    path=ROOT/'selection_lock.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:save_json(path,lock)
    print('Final selection locked:',selected,'; diagnostic challenger:',challenger)

if __name__=='__main__':main()
