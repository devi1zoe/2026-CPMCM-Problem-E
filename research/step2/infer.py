"""附件3最终推理；只加载锁定的模型与训练统计，不读取任何标签。"""
from common import *
from data import *
from train import load_model,predict
import pandas as pd


def main():
    torch.set_num_threads(4)
    lock=json.loads((ROOT/'selection_lock.json').read_text())
    ids,d=load_special();base=adapt(d)
    encoder=TextEncoder();stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    view=make_view(base,encoder,stats,dict(subset='',rate=0.,position='random'),0)
    tv=torch_view(view);outputs=[]
    for seed in lock['seeds']:
        name=f'{lock["selected"]}_{seed}.pt'
        assert sha256(ROOT/'checkpoints'/name)==lock['checkpoint_sha256'][name]
        model,_=load_model(ROOT/'checkpoints'/name);outputs.append(predict(model,tv))
    prob,pred,gates=[np.mean([o[j] for o in outputs],0) for j in range(3)]
    std=np.std([o[1] for o in outputs],axis=0,ddof=1)
    names=['negative','neutral','positive'];rows=[];details=[]
    for i,id_ in enumerate(ids):
        cls=int(prob[i].argmax())
        rows.append(dict(sample_id=id_,polarity=names[cls],intensity=float(pred[i])))
        details.append(dict(**rows[-1],class_id=cls,prob_negative=float(prob[i,0]),prob_neutral=float(prob[i,1]),prob_positive=float(prob[i,2]),
            intensity_seed_std=float(std[i]),unknown_token_count=int(np.sum(base['tokens'][i]==100)),
            valid_slots=int(base['p'][i].sum()),span_uncertain=i in base['uncertain'],
            nonneutral_sign_disagreement=bool(cls!=1 and np.sign(pred[i])+1!=cls),
            **{f'observed_fraction_{m}':float(view['quality'][i,j,0]) for j,m in enumerate('TAV')},
            **{f'fusion_weight_{m}':float(gates[i,j]) for j,m in enumerate('TAV')}))
    df=pd.DataFrame(rows)
    assert len(df)==30 and df.sample_id.is_unique and df.intensity.between(-3,3).all()
    df.to_csv(ROOT/'results/附件3_问题二预测.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(details).to_csv(ROOT/'results/附件3_预测审阅.csv',index=False,encoding='utf-8-sig')
    save_json(ROOT/'results/special_inference_audit.json',dict(n=30,selected=lock['selected'],input_version='aligned',
        no_labels_accessed=True,span_uncertain_sample_ids=[ids[i] for i in base['uncertain']],
        unknown_token_total=int((base['tokens']==100).sum()),
        zero_values_note='Observed availability is not identifiable artificial missing rate; no retrieval of original text or other feature version.',
        special_sha256={f.name:sha256(f) for f in sorted((DATA/'附件3-模态缺失特征样本/对齐版本').glob('*.pkl'))}))
    print(df.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
