from common import *
from data import *
from engine import *
import argparse,pandas as pd


def get_view(base,split,profile,encoder,stats,proto):
    stamp=hashlib.sha256(json.dumps(dict(split=split,profile=profile,source=proto['source_sha256'],encoder=proto['encoder'],
        stats=proto['normalization_sha256'],seed=proto['evaluation_seed']),sort_keys=True).encode()).hexdigest()[:20]
    existing=PROJECT/f'step2_v2/cache/eval_{split}_{stamp}.npz'
    if existing.is_file():
        view=dict(np.load(existing));o,removed=corrupt(base,profile,proto['evaluation_seed']+profile['repeat'])
        assert np.array_equal(view['p'],base['p']) and np.array_equal(view['o'],o) and np.array_equal(view['removed'],removed)
        return view
    return make_view(base,encoder,stats,profile,proto['evaluation_seed']+profile['repeat'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['valid','test'],default='valid');args=ap.parse_args()
    torch.set_num_threads(1);proto=json.loads((ROOT/'protocol.json').read_text());split=args.split
    if split=='valid':names=['old_gated']+proto['models']
    else:
        lock=json.loads((ROOT/'selection_lock.json').read_text());assert lock['protocol_sha256']==sha256(ROOT/'protocol.json')
        names=['old_gated',lock['selected_attention']]
    for name in names:
        if name!='old_gated':
            for s in proto['seeds']:
                done=json.loads((ROOT/f'results/completed_{name}_{s}.json').read_text());assert done['checkpoint_sha256']==sha256(ROOT/f'checkpoints/{name}_{s}.pt')
    models={n:[load_model(n,s)[0] for s in proto['seeds']] for n in names}
    d=load_attachment2()[split];base=adapt(d);c=d['classification_labels'].astype(int);y=d['regression_labels']
    encoder=TextEncoder();stats=dict(np.load(ROOT/'checkpoints/normalization.npz'));profiles=list(scenarios())
    outputs={n:np.empty((len(profiles),3,len(c),4),np.float32) for n in names};rows=[];audit=[]
    if split=='valid':
        six=[]
        for j,p in enumerate(proto['validation_profiles']):
            v=torch_view(dict(np.load(ROOT/f'cache/valid_{j}.npz')))
            for name,ms in models.items():
                out=np.mean([predict(m,v) for m in ms],0)
                six.append(dict(model=name,profile=p['name'],**metrics(c,y,out)))
        pd.DataFrame(six).to_csv(ROOT/'results/validation_six.csv',index=False)
    for k,p in enumerate(profiles):
        v=get_view(base,split,p,encoder,stats,proto);tv=torch_view(v)
        for name,ms in models.items():
            for j,(seed,m) in enumerate(zip(proto['seeds'],ms)):
                out=predict(m,tv);outputs[name][k,j]=out
                rows.append(dict(model=name,seed=str(seed),**p,**metrics(c,y,out)))
            out=outputs[name][k].mean(0);rows.append(dict(model=name,seed='ensemble',**p,**metrics(c,y,out)))
        audit.append(dict(**p,**{f'actual_fraction_{m}':float((v['removed'][:,j].sum(-1)/np.maximum(v['p'].sum(-1),1)).mean()) for j,m in enumerate('TAV')}))
        if k%10==0:print(f'{split} {k+1}/{len(profiles)} {p["name"]}',flush=True)
        del v,tv
    pd.DataFrame(rows).to_csv(ROOT/f'results/{split}_metrics.csv',index=False)
    for n,out in outputs.items():np.savez_compressed(ROOT/f'results/{split}_{n}_predictions.npz',out=out,c=c,y=y,ids=np.array(d['id']))
    pd.DataFrame(audit).to_csv(ROOT/f'results/{split}_corruption_audit.csv',index=False)
    save_json(ROOT/f'results/{split}_profiles.json',profiles)
    save_json(ROOT/f'results/completed_evaluation_{split}.json',dict(models=names,rows=len(rows),profiles=len(profiles),protocol_sha256=sha256(ROOT/'protocol.json'),
        weights={f'{n}_{s}.pt':sha256(ROOT/f'checkpoints/{n}_{s}.pt') for n in names for s in proto['seeds']}))
    print('Evaluation completed:',split,flush=True)


if __name__=='__main__':main()
