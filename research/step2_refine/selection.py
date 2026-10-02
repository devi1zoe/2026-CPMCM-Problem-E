"""Immutable validation decision under the original seven acceptance gates."""
from common import *
from engine import checkpoint_paths
import pandas as pd


def summary(name):
    df=pd.read_csv(ROOT/f'results/valid_{name}_metrics.csv');e=df[df.seed=='ensemble'];clean=e[e.name=='clean'].iloc[0]
    robust=e[e.primary==True].groupby('name')[['macro_f1','mae']].mean().mean();six=json.loads((ROOT/f'results/six_{name}.json').read_text())
    return dict(model=name,J=six['score'],parameters=six['parameters'],clean_f1=float(clean.macro_f1),clean_mae=float(clean.mae),neutral_recall=float(clean.neutral_recall),
        strong_negative_mae=float(clean.strong_negative_mae),strong_positive_mae=float(clean.strong_positive_mae),robust_f1=float(robust.macro_f1),robust_mae=float(robust.mae))


def guardrails(r,old):
    e=1e-10
    return dict(clean_f1=r['clean_f1']>=old['clean_f1']-.005-e,clean_mae=r['clean_mae']<=old['clean_mae']+.01+e,
        robust_f1=r['robust_f1']>=old['robust_f1']+.01-e,robust_mae=r['robust_mae']<=old['robust_mae']+e,
        neutral=r['neutral_recall']>=old['neutral_recall']-.01-e,strong_negative=r['strong_negative_mae']<=old['strong_negative_mae']+.02+e,
        strong_positive=r['strong_positive_mae']<=old['strong_positive_mae']+.02+e)


def main():
    rows=[summary(n) for n in ['old_gated','text','gated','dual_gate','late_fusion']];old=rows[0]
    for r in rows:r['checks']=guardrails(r,old);r['eligible']=all(r['checks'].values()) and r['model'] in ['dual_gate','late_fusion']
    eligible=[r for r in rows if r['eligible']];selected='old_gated'
    if eligible:
        best=min(r['J'] for r in eligible);selected=min([r for r in eligible if r['J']-best<.002],key=lambda r:(r['parameters'],r['J'],r['model']))['model']
    lock=dict(selected=selected,accepted_improvement=selected!='old_gated',protocol_sha256=sha256(ROOT/'protocol.json'),rows=rows,
        checkpoints={str(p.relative_to(PROJECT)):sha256(p) for p in checkpoint_paths(selected)},final_test_models=[selected] if eligible else [],
        previous_test_results_seen=True,special_used_for_selection=False,rule='All original guardrails; minimum J; fewer total deployment parameters within .002; else old_gated')
    path=ROOT/'selection_lock.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:save_json(path,lock)
    pd.DataFrame([{**{k:v for k,v in r.items() if k!='checks'},**{'pass_'+k:v for k,v in r['checks'].items()}} for r in rows]).to_csv(ROOT/'results/selection_comparison.csv',index=False)
    print(json.dumps(lock,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
