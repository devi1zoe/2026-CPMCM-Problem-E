# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""全728条验证集：预测、Shapley、IG、原输入删除/保留、随机对照与稳定性。"""
from common import *
from engine import *
from explain import *
from data import load_attachment2
import pandas as pd

def jaccard(a,b):
    left=set(range(int(a[0]),int(a[1])));right=set(range(int(b[0]),int(b[1])))
    return len(left&right)/max(1,len(left|right))

def main():
    torch.set_num_threads(4)
    models,lock=selected_models();protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha256(ROOT/'protocol.json')==lock['protocol_sha256']
    # 补充的全局充分性定义在任何解释实验输出前固定，不参与重新选模型。
    save_json(ROOT/'analysis_protocol.json',dict(selection_sha256=sha256(ROOT/'selection_lock.json'),
        target=protocol['explanation_target'],local_fraction=.2,random_controls=5,
        conditional_retention='keep window in selected modality; other modalities unchanged',
        unconditional_retention='keep only selected window in selected modality; remove all other modalities',
        stability_sample_seed=20260926,model_reselection_after_explanation=False,
        oracle_warning='exhaustive maximum deletion used only for special evidence confirmation, never presented as independent fidelity test'))
    raw=load_attachment2()['valid'];view=load_view('valid');out=predictions(models,view)
    target=out[:,:3].argmax(1);labels=raw['classification_labels'].astype(int);y=raw['regression_labels']
    np.savez(ROOT/'results/validation_predictions.npz',prob=out[:,:3],pred=out[:,3],c=labels,y=y)
    predrows=[dict(sample_id=str(id_),raw_text=str(raw['raw_text'][i]),true_class=int(labels[i]),true_intensity=float(y[i]),
        predicted_class=int(target[i]),predicted_intensity=float(out[i,3]),confidence=float(out[i,target[i]]),abs_error=float(abs(out[i,3]-y[i]))) for i,id_ in enumerate(raw['id'])]
    pd.DataFrame(predrows).to_csv(ROOT/'results/validation_predictions.csv',index=False)
    refs=dict(np.load(ROOT/'checkpoints/training_reference.npz'))
    print('computing all 8 coalitions, two baseline mechanisms',flush=True)
    values=coalition_values(models,view);refvalues=coalition_values(models,view,refs)
    records,phi,main_mod=modality_records(raw['id'],values,refvalues,target)
    pd.DataFrame(records).to_csv(ROOT/'results/validation_modality_contributions.csv',index=False)
    np.savez_compressed(ROOT/'results/validation_coalitions.npz',values=values,reference_values=refvalues,target=target,phi=phi)
    print('computing IG on all 728 validation samples',flush=True)
    ig32,residual32=integrated_gradients(models,view,target,nodes=protocol['ig']['nodes'])
    ig,residual,used_nodes=refine_integrals(models,view,target,ig32,residual32)
    chosen,bank=rank_windows(ig,view['p'],.2)
    np.savez_compressed(ROOT/'results/validation_ig.npz',ig=ig,residual=residual,chosen=chosen,target=target,
        original_ig32=ig32,original_residual32=residual32,used_nodes=used_nodes)
    rng=np.random.default_rng(20260926);specs=[];layout=[]
    for i in range(len(target)):
        for m in range(3):
            starts=[tuple(chosen[i,m])]+[bank[i][int(rng.integers(len(bank[i])))] for _ in range(5)]
            for role,(a,b) in enumerate(starts):
                for mode in ['drop','keep','only']:
                    specs.append((i,m,a,b,mode));layout.append((i,m,role,mode))
    engine=ProbeEngine(models,raw,view)
    outputs=[]
    for start in range(0,len(specs),1536):
        outputs.append(engine.probe(specs[start:start+1536]))
        print('probed',min(start+1536,len(specs)),'/',len(specs),flush=True)
    probes=np.concatenate(outputs)
    cube=probes.reshape(len(target),3,6,3,4)
    rows=[]
    for i,id_ in enumerate(raw['id']):
        f=float(out[i,target[i]])
        for m,name in enumerate(NAMES):
            a,b=chosen[i,m];chosen_p=cube[i,m,0,:,target[i]];random_p=cube[i,m,1:,:,target[i]]
            rows.append(dict(sample_id=str(id_),modality=name,is_main=bool(m==main_mod[i]),target_class=int(target[i]),prediction_correct=bool(target[i]==labels[i]),
                slot_start=int(a),slot_end=int(b),effective_length=int(view['p'][i].sum()),
                ig_window_sum=float(ig[i,m,a:b].sum()),full_score=f,
                deletion_drop=float(f-chosen_p[0]),random_deletion_drop=float(f-random_p[:,0].mean()),
                deletion_advantage=float(random_p[:,0].mean()-chosen_p[0]),
                conditional_sufficiency_gap=float(f-chosen_p[1]),random_conditional_gap=float(f-random_p[:,1].mean()),
                unconditional_sufficiency_gap=float(f-chosen_p[2]),random_unconditional_gap=float(f-random_p[:,2].mean()),
                conditional_class_retained=bool(cube[i,m,0,1,:3].argmax()==target[i]),
                unconditional_class_retained=bool(cube[i,m,0,2,:3].argmax()==target[i]),
                intensity_deletion_delta=float(out[i,3]-cube[i,m,0,0,3]),ig_completeness_abs_error=float(abs(residual[i]))))
    fidelity=pd.DataFrame(rows);fidelity.to_csv(ROOT/'results/validation_fidelity.csv',index=False)
    # raw probe log可重算每一个随机对照，避免只保留有利均值。
    np.savez_compressed(ROOT/'results/validation_probe_outputs.npz',outputs=cube,
        specs=np.array([[i,m,a,b,['drop','keep','only'].index(mode)] for i,m,a,b,mode in specs],np.int16),target=target)
    print('computing convergence and stability checks',flush=True)
    stable_idx=np.concatenate([rng.choice(np.flatnonzero(labels==c),32,replace=False) for c in range(3)])
    stable_idx=rng.permutation(stable_idx)
    subset={k:v[stable_idx] for k,v in view.items()};starget=target[stable_idx]
    conv={k:v[:48] for k,v in subset.items()}
    ig64,res64=integrated_gradients(models,conv,starget[:48],nodes=64)
    np.savez_compressed(ROOT/'results/ig_convergence.npz',indices=stable_idx[:48],ig32=ig32[stable_idx[:48]],ig64=ig64,residual32=residual32[stable_idx[:48]],residual64=res64)
    seed_windows=[]
    for model in models:
        si,se=integrated_gradients([model],subset,starget,nodes=32);si,se,_=refine_integrals([model],subset,starget,si,se)
        sw,_=rank_windows(si,subset['p']);seed_windows.append(sw)
    noisy={k:v.copy() for k,v in subset.items()}
    for m,name in enumerate(NAMES):
        noisy[name]=(noisy[name].astype(np.float32)+rng.normal(0,.02,noisy[name].shape)*noisy['o'][:,m,:,None]).clip(-8,8).astype(np.float16)
    noise_ig,ne=integrated_gradients(models,noisy,starget,nodes=32);noise_ig,ne,_=refine_integrals(models,noisy,starget,noise_ig,ne)
    noise_windows,_=rank_windows(noise_ig,noisy['p'])
    noise_values=coalition_values(models,noisy).mean(0);noise_phi,_=shapley(noise_values,starget);noise_main,_,_=dominant(noise_phi)
    ref_ig,re=integrated_gradients(models,subset,starget,nodes=32,reference=refs);ref_ig,re,_=refine_integrals(models,subset,starget,ref_ig,re,refs)
    ref_windows,_=rank_windows(ref_ig,subset['p'])
    stability=[]
    for j,i in enumerate(stable_idx):
        m=int(main_mod[i]);origin=chosen[i,m]
        stability.append(dict(sample_id=str(raw['id'][i]),main_modality=NAMES[m],noise_main_agrees=bool(noise_main[j]==m),
            noise_window_jaccard=jaccard(origin,noise_windows[j,m]),reference_ig_window_jaccard=jaccard(origin,ref_windows[j,m]),
            seed_window_pairwise_jaccard=float(np.mean([jaccard(seed_windows[a][j,m],seed_windows[b][j,m]) for a,b in [(0,1),(0,2),(1,2)]]))))
    pd.DataFrame(stability).to_csv(ROOT/'results/validation_stability.csv',index=False)
    # 按源视频聚类的配对bootstrap，只评价主模态IG窗口相对等长随机窗口的优势。
    primary=fidelity[fidelity.is_main].copy();groups={}
    for i,id_ in enumerate(primary.sample_id):groups.setdefault(id_.split('$_$')[0],[]).append(i)
    groups=list(groups.values());delta=primary.deletion_advantage.to_numpy();draws=[]
    for _ in range(1000):
        idx=np.concatenate([groups[g] for g in rng.integers(len(groups),size=len(groups))]);draws.append(float(delta[idx].mean()))
    summary=dict(n=len(target),prediction=metrics(labels,y,out[:,:3],out[:,3]),
        ig=dict(nodes='adaptive 32/64/128/256, absolute completeness tolerance 0.001',nodes_used={str(n):int((used_nodes==n).sum()) for n in np.unique(used_nodes)},
            initial32_max_error=float(np.max(abs(residual32))),failed_tolerance=int((abs(residual)>.001).sum()),
            completeness_median=float(np.median(abs(residual))),completeness_p95=float(np.quantile(abs(residual),.95)),
            completeness_max=float(np.max(abs(residual))),convergence_64_p95=float(np.quantile(abs(res64),.95)),
            convergence_ig_mean_absolute_change=float(np.mean(abs(ig64-ig32[stable_idx[:48]])))),
        primary_fidelity={k:float(primary[k].mean()) for k in ['deletion_drop','random_deletion_drop','deletion_advantage','conditional_sufficiency_gap',
            'random_conditional_gap','unconditional_sufficiency_gap','random_unconditional_gap','conditional_class_retained','unconditional_class_retained']},
        positive_deletion_fraction=float((primary.deletion_drop>0).mean()),random_advantage_ci95=np.quantile(draws,[.025,.975]).tolist(),
        modality_baseline_main_agreement=float(pd.DataFrame(records).groupby('sample_id').baseline_main_agrees.first().mean()),
        modality_seed_main_agreement=float(pd.DataFrame(records).groupby('sample_id').main_seed_agreement.first().mean()),
        stability=pd.DataFrame(stability).select_dtypes(include=['number','bool']).mean().to_dict(),
        selected_model=lock['selected'],fixed_target_verified=True,
        interpretation='IG ranking evaluated by independent raw-token deletion; representation baselines and deletion games are explicitly distinct')
    save_json(ROOT/'results/validation_explanation_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
