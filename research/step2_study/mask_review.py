"""Audit every actually trained group-1 mask and distinguish selected/lost slots."""
from common import *
from data import *
from dynamic import epoch_masks,context_key
from evaluation_data import profiles_all,get_masks
import pandas as pd

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2();base=adapt(raw['train']);rows=[];evaluations=[]
    for seed in protocol['seeds']:
        histories={name:{r['epoch'] for r in json.loads((ROOT/f'results/history_{name}_{seed}.json').read_text())} for name in protocol['groups']['1']}
        for epoch in sorted(set.union(*histories.values())):
            masks={s:epoch_masks(base,raw['train']['id'],seed,epoch,s) for s in ['single','multi','unequal']}
            a=masks['single'][1];b=masks['multi'][1];c=masks['unequal'][1]
            assert np.array_equal(a.sum(-1),b.sum(-1)) and np.array_equal(a.any(-1),c.any(-1))
            for name,epochs in histories.items():
                if epoch not in epochs:continue
                shape=name.removeprefix('shape_');o,r=masks[shape];key=context_key(base,o,protocol['encoder']['weights_sha256']);mh=hashlib.sha256(r.tobytes()).hexdigest()
                old=PROJECT/f'step2_v2/cache/epoch_{seed}_{epoch}_{key[:16]}.npz'
                path=old if shape=='single' and old.exists() else ROOT/f'cache/{shape}_{seed}_{epoch}_{mh[:16]}.npz'
                with np.load(path) as stored:assert np.array_equal(stored['o'],o) and np.array_equal(stored['removed'],r)
                rows.append(dict(model=name,seed=seed,epoch=epoch,clean_samples=int((~r.any((1,2))).sum()),
                    selected_slots=int(r.sum()),newly_lost_observed_slots=int((r&base['o']).sum()),mask_sha256=mh))
    for split in ['valid','test']:
        b=adapt(raw[split]);length=np.maximum(b['p'].sum(-1),1)
        for profile in profiles_all():
            o,r=get_masks(b,profile,protocol['evaluation_seed']+profile['repeat']);entry=dict(split=split,**profile)
            for m,name in enumerate('TAV'):
                entry['selected_slot_fraction_'+name]=float((r[:,m].sum(-1)/length).mean())
                entry['newly_lost_observed_fraction_'+name]=float(((r[:,m]&b['o'][:,m]).sum(-1)/length).mean())
                entry['remaining_observed_fraction_'+name]=float((o[:,m].sum(-1)/length).mean())
            evaluations.append(entry)
    pd.DataFrame(rows).to_csv(ROOT/'results/training_mask_review.csv',index=False)
    pd.DataFrame(evaluations).to_csv(ROOT/'results/observed_missing_review.csv',index=False)
    save_json(ROOT/'results/mask_review_complete.json',dict(passed=True,training_views_checked=len(rows),evaluation_views_checked=len(evaluations),
        all_group1_trained_masks_recomputed=True,single_multi_selected_counts_equal=True,selected_slots_distinct_from_originally_observed_loss=True))
    print('Every trained group1 mask checked:',len(rows),'evaluation masks:',len(evaluations))

if __name__=='__main__':main()
