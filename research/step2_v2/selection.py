"""Predeclared validation guardrails, leave-one-out registration, and immutable final selection."""
from common import *
from engine import checkpoint_paths
from model import VARIANTS
import argparse,pandas as pd,numpy as np

def normalize_legacy_boundary():
    """Upgrade the initial in-flight legacy evaluation to the predeclared common empty-input policy."""
    proof=ROOT/'results/evaluated_valid_old_gated.json'
    status=json.loads(proof.read_text())
    if status.get('all_missing_policy')=='train_prior':return
    from data import load_attachment2,adapt,corrupt,torch_view
    from engine import metrics,load_model,predict,score
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2()['valid'];base=adapt(raw)
    path=ROOT/'results/valid_old_gated_predictions.npz';saved=dict(np.load(path));out=saved['out']
    profiles=json.loads((ROOT/'results/valid_profiles.json').read_text());rows=[];affected=0
    for j,p in enumerate(profiles):
        o,_=corrupt(base,p,protocol['evaluation_seed']+p['repeat']);empty=~o.any((1,2));affected+=int(empty.sum())
        out[j,:,empty,:3]=np.array(protocol['prior'],np.float32);out[j,:,empty,3]=protocol['mean']
        for k,s in enumerate([17,29,43]):rows.append(dict(model='old_gated',seed=str(s),**p,**metrics(saved['c'],saved['y'],out[j,k,:,:3],out[j,k,:,3])))
        avg=out[j].mean(0);rows.append(dict(model='old_gated',seed='ensemble',**p,**metrics(saved['c'],saved['y'],avg[:,:3],avg[:,3])))
    saved['out']=out;np.savez_compressed(path,**saved)
    pd.DataFrame(rows).to_csv(ROOT/'results/valid_old_gated_metrics.csv',index=False)
    models=[load_model(p)[0] for p in checkpoint_paths('old_gated')];six=[]
    for j,p in enumerate(protocol['validation_profiles']):
        tv=torch_view(dict(np.load(ROOT/f'cache/valid_{j}.npz')));avg=np.mean([predict(m,tv) for m in models],0)
        six.append(dict(**p,**metrics(saved['c'],saved['y'],avg[:,:3],avg[:,3])))
    save_json(ROOT/'results/six_old_gated.json',dict(score=score(six),rows=six,parameters=sum(p.numel() for p in models[0].parameters())))
    status.update(all_missing_policy='train_prior',empty_sample_scenario_instances=affected,old_weights_and_files_unchanged=True)
    save_json(proof,status)
    print('Unified legacy empty-input prior fallback:',affected,'sample/scenario instances',flush=True)

def summary(name):
    df=pd.read_csv(ROOT/f'results/valid_{name}_metrics.csv');e=df[df.seed=='ensemble']
    clean=e[e.name=='clean'].iloc[0]
    robust=e[e.primary==True].groupby('name')[['macro_f1','mae']].mean().mean()
    six=json.loads((ROOT/f'results/six_{name}.json').read_text())
    return dict(model=name,J=six['score'],parameters=six['parameters'],clean_f1=float(clean.macro_f1),clean_mae=float(clean.mae),
        neutral_recall=float(clean.neutral_recall),strong_negative_mae=float(clean.strong_negative_mae),strong_positive_mae=float(clean.strong_positive_mae),
        robust_f1=float(robust.macro_f1),robust_mae=float(robust.mae))

def guardrails(r,old):
    eps=1e-10
    return dict(clean_f1=r['clean_f1']>=old['clean_f1']-.005-eps,clean_mae=r['clean_mae']<=old['clean_mae']+.01+eps,
        robust_f1=r['robust_f1']>=old['robust_f1']+.01-eps,robust_mae=r['robust_mae']<=old['robust_mae']+eps,
        neutral=r['neutral_recall']>=old['neutral_recall']-.01-eps,
        strong_negative=r['strong_negative_mae']<=old['strong_negative_mae']+.02+eps,
        strong_positive=r['strong_positive_mae']<=old['strong_positive_mae']+.02+eps)

def choose(rows):
    best=min(r['J'] for r in rows);tied=[r for r in rows if r['J']-best<.002]
    return min(tied,key=lambda r:(r['parameters'],r['J'],r['model']))['model']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');args=ap.parse_args()
    normalize_legacy_boundary()
    base_names=list(VARIANTS);rows=[summary(n) for n in ['old_gated']+base_names];old=rows[0]
    for r in rows:r['checks']=guardrails(r,old);r['eligible']=all(r['checks'].values()) and r['model'] not in ['text','audio','vision','old_gated']
    candidates=[r for r in rows if r['eligible']]
    # If none pass, audit the best multimodal research candidate while preserving fallback.
    audit_name=choose(candidates or [r for r in rows if r['model'] not in ['text','audio','vision','old_gated']])
    if not args.final:
        conf=VARIANTS[audit_name];ablations={}
        for module in ['aux','msd','cce','imtd','emc']:
            if conf.get(module):
                c=dict(conf);c[module]=False;ablations[audit_name+'_without_'+module]=c
        save_json(ROOT/'ablation_configs.json',ablations)
        save_json(ROOT/'results/preliminary_selection.json',dict(audit_candidate=audit_name,eligible=[r['model'] for r in candidates],rows=rows,
            explanation='Leave-one-out performed even if candidate fails final guardrails; test remains inaccessible.'))
        print('Leave-one-out candidate:',audit_name,ablations,flush=True);return
    ablations=json.loads((ROOT/'ablation_configs.json').read_text())
    for name in ablations:
        r=summary(name);r['checks']=guardrails(r,old);r['eligible']=all(r['checks'].values());rows.append(r)
    eligible=[r for r in rows if r['eligible']];selected=choose(eligible) if eligible else 'old_gated'
    lock=dict(selected=selected,accepted_improvement=selected!='old_gated',audit_candidate=audit_name,
        protocol_sha256=sha256(ROOT/'protocol.json'),regression_weight_lock_sha256=sha256(ROOT/'regression_weight_lock.json'),
        checkpoints={str(p.relative_to(PROJECT)):sha256(p) for p in checkpoint_paths(selected)},
        final_test_models=list(dict.fromkeys([selected,'old_gated','text',audit_name])),previous_test_results_seen=True,
        special_used_for_selection=False,rule='all guardrails then minimum six-profile J; within .002 fewer parameters',rows=rows)
    path=ROOT/'selection_lock.json'
    if path.exists():assert json.loads(path.read_text())==lock,'Selection already locked'
    else:save_json(path,lock)
    flat=[{**{k:v for k,v in r.items() if k!='checks'},**{'pass_'+k:v for k,v in r['checks'].items()}} for r in rows]
    pd.DataFrame(flat).to_csv(ROOT/'results/selection_comparison.csv',index=False)
    print('FINAL SELECTED',selected,'accepted',lock['accepted_improvement'],flush=True)

if __name__=='__main__':main()
