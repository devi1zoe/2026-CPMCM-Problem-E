from common import *
from data import *
from engine import *
import pandas as pd


def main():
    torch.set_num_threads(1);p=json.loads((ROOT/'protocol.json').read_text());lock=json.loads((ROOT/'selection_lock.json').read_text())
    assert lock['protocol_sha256']==sha256(ROOT/'protocol.json')
    name=lock['selected_attention'];models=[]
    for s in p['seeds']:
        assert sha256(ROOT/f'checkpoints/{name}_{s}.pt')==lock['weights'][f'{name}_{s}.pt'];models.append(load_model(name,s)[0])
    ids,d=load_special();b=adapt(d);encoder=TextEncoder();stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    v=make_view(b,encoder,stats,dict(subset='',rate=0.,position='random'),0);tv=torch_view(v)
    each=np.stack([predict(m,tv) for m in models]);out=each.mean(0);cls=out[:,:3].argmax(1)
    rows=[];details=[]
    for i,sid in enumerate(ids):
        r=dict(sample_id=sid,polarity=['negative','neutral','positive'][cls[i]],intensity=float(out[i,3]));rows.append(r)
        details.append(dict(**r,**{f'prob_{j}':float(out[i,j]) for j in range(3)},intensity_seed_std=float(each[:,i,3].std(ddof=1)),
            **{f'observed_{m}':float(v['quality'][i,j,0]) for j,m in enumerate('TAV')},unk_count=int((b['tokens'][i]==100).sum()),
            directional_disagreement=bool(cls[i]!=1 and np.sign(out[i,3])+1!=cls[i])))
    pd.DataFrame(rows).to_csv(ROOT/'results/附件3_注意力模型预测.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(details).to_csv(ROOT/'results/附件3_预测审阅.csv',index=False,encoding='utf-8-sig')
    save_json(ROOT/'results/special_audit.json',dict(n=len(ids),selected=name,unknown_tokens=int((b['tokens']==100).sum()),uncertain_span=b['uncertain'],
        sources={f.name:sha256(f) for f in sorted((DATA/'附件3-模态缺失特征样本/对齐版本').glob('*.pkl'))}))
    # A label-independent fixed validation example shows attention routing, not causal attribution.
    raw=load_attachment2()['valid'];base=adapt({k:v[:1] for k,v in raw.items()})
    cm,_=load_model('cross',17)
    for cond,profile in [('clean',dict(subset='',rate=0.,position='middle')),('missing',dict(subset='TAV',rate=.3,position='middle'))]:
        view=make_view(base,encoder,stats,profile,1);bt=torch_view(view)
        with torch.no_grad():r=cm(bt,return_attention=True)
        np.savez(ROOT/f'results/attention_{cond}.npz',cross=r['cross_attention'].mean(1).cpu().numpy(),
            self_weights=np.stack([w.mean(1).cpu().numpy() for w in r['self_attention']],1),o=view['o'],p=view['p'],tokens=base['tokens'])
    save_json(ROOT/'results/attention_example.json',dict(sample_id=raw['id'][0],raw_text=raw['raw_text'][0],model='cross',seed=17,interpretation='internal routing, not causal attribution'))
    print('30 predictions and attention routing example written.',flush=True)


if __name__=='__main__':main()
