"""Attachment 3 inference after final validation selection only."""
from common import *
from data import *
from engine import load_model,predict,checkpoint_paths
import pandas as pd

def main():
    torch.set_num_threads(2);lock=json.loads((ROOT/'selection_lock.json').read_text());protocol=json.loads((ROOT/'protocol.json').read_text())
    assert lock['protocol_sha256']==sha256(ROOT/'protocol.json');name=lock['selected']
    ids,raw=load_special();base=adapt(raw);encoder=TextEncoder();stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    view=make_view(base,encoder,stats,dict(subset='',rate=0.,position='random'),0);predictions=[]
    for path in checkpoint_paths(name):
        assert sha256(path)==lock['checkpoint_hashes'][name][str(path.relative_to(PROJECT))]
        model,ck=load_model(path)
        b={**view,'tokens':base['tokens'],'attention':base['attention']} if hasattr(model,'language') else view
        predictions.append(predict(model,b,batch=64));del model
    array=np.stack(predictions);average=array.mean(0);classes=['negative','neutral','positive'];rows=[];details=[]
    for i,sid in enumerate(ids):
        rows.append(dict(sample_id=sid,polarity=classes[average[i,:3].argmax()],intensity=float(average[i,3])))
        details.append(dict(**rows[-1],prob_negative=float(average[i,0]),prob_neutral=float(average[i,1]),prob_positive=float(average[i,2]),
            intensity_seed_std=float(array[:,i,3].std(ddof=1)),**{f'observed_fraction_{m}':float(base['o'][i,j].sum()/max(1,base['p'][i].sum())) for j,m in enumerate('TAV')}))
    assert len(rows)==30 and len(set(ids))==30 and np.isfinite(array).all() and np.all(np.abs(average[:,3])<=3)
    assert np.allclose(average[:,:3].sum(-1),1,atol=1e-6)
    path=ROOT/'results/附件3_问题二预测.csv';pd.DataFrame(rows).to_csv(path,index=False);pd.DataFrame(details).to_csv(ROOT/'results/附件3_预测审阅.csv',index=False)
    source={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in raw.items()}
    save_json(ROOT/'results/special_inference_complete.json',dict(selected=name,samples=30,source_arrays=source,selection_sha256=sha256(ROOT/'selection_lock.json'),
        predictions_sha256=sha256(path),labels_available=False,used_for_selection=False,uncertain_valid_ranges=len(base['uncertain'])))
    print('Saved all 30 attachment3 predictions for',name)

if __name__=='__main__':main()
