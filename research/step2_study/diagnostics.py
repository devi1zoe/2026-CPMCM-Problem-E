"""Actual error types, rescued/harmed predictions, and reviewable examples."""
from common import *
from data import *
from evaluation_data import get_masks
import pandas as pd
import re

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2()['valid'];base=adapt(raw)
    profiles=json.loads((ROOT/'results/valid_profiles.json').read_text());c=np.asarray(raw['classification_labels']).astype(int);y=np.asarray(raw['regression_labels'])
    texts=np.asarray(raw['raw_text']).astype(str)
    groups={'全部':np.ones(len(c),bool),'中性':c==1,'强负向':y<=-1.5,'强正向':y>=1.5,
        '评分表达线索':np.array([bool(re.search(r'out of|\bstars?\b|\d\s*/\s*(?:5|10)',t,re.I)) for t in texts]),
        '比较转折线索':np.array([bool(re.search(r'\b(?:than|compared|but|however|although|though|yet)\b',t,re.I)) for t in texts])}
    names=['old_gated',*protocol['configs'],'expert_text']
    if (ROOT/'results/evaluated_valid_combined.json').exists():names.append('combined')
    predictions={}
    for name in names:
        path=ROOT/f'results/valid_{name}_predictions.npz'
        if not path.exists():continue
        d=dict(np.load(path));assert np.array_equal(d['ids'],np.asarray(raw['id'],dtype=str));predictions[name]=d['out'].mean(1)
    masks=[get_masks(base,p,protocol['evaluation_seed']+p['repeat'])[0] for p in profiles]
    error_rows=[];rescue_rows=[];examples=[]
    for name,out in predictions.items():
        for si,profile in enumerate(profiles):
            for group,keep in groups.items():
                if not keep.any():continue
                pred=out[si,keep];error_rows.append(dict(model=name,**profile,group=group,n=int(keep.sum()),
                    accuracy=float((pred[:,:3].argmax(-1)==c[keep]).mean()),mae=float(np.abs(pred[:,3]-y[keep]).mean()),
                    true_mean=float(y[keep].mean()),predicted_mean=float(pred[:,3].mean())))
        reference='expert_text' if name.startswith('gate_') else ('text_lora' if name in ['multi_lora','combined'] else 'text_frozen')
        if reference in predictions and name!=reference:
            ref=predictions[reference]
            for si,profile in enumerate(profiles):
                a=ref[si,:,:3].argmax(-1)==c;b=out[si,:,:3].argmax(-1)==c
                for group,mask in groups.items():
                    for scope,keep in [('全部样本',mask),('文本仍有观测',mask&masks[si][:,0].any(-1))]:
                        rescue_rows.append(dict(model=name,reference=reference,**profile,group=group,scope=scope,n=int(keep.sum()),
                            text_wrong=int((~a&keep).sum()),rescued=int((~a&b&keep).sum()),text_correct=int((a&keep).sum()),harmed=int((a&~b&keep).sum())))
    pairs=[('shape_multi','shape_single'),('shape_unequal','shape_single'),('text_lora','text_frozen'),('multi_lora','shape_single'),
        ('gate_reliability','gate_quality'),('gate_correction','gate_reliability'),('reconstruct','shape_single')]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ENCODER,local_files_only=True)
    scenes=[0,next(i for i,p in enumerate(profiles) if p['name']=='TAV_r70_random' and p['repeat']==0)]
    for model,reference in pairs:
        if model not in predictions or reference not in predictions:continue
        for si in scenes:
            a=predictions[reference][si];b=predictions[model][si];ac=a[:,:3].argmax(-1);bc=b[:,:3].argmax(-1)
            gain=np.abs(a[:,3]-y)-np.abs(b[:,3]-y)
            categories={'类别改对':np.flatnonzero((ac!=c)&(bc==c)),'类别改错':np.flatnonzero((ac==c)&(bc!=c)),
                '强度误差减小':np.flatnonzero(gain>0),'强度误差增大':np.flatnonzero(gain<0)}
            for kind,indices in categories.items():
                chosen=sorted(indices,key=lambda i:abs(gain[i]),reverse=True)[:3]
                for i in chosen:
                    examples.append(dict(model=model,reference=reference,scenario=profiles[si]['name'],kind=kind,sample_id=str(raw['id'][i]),
                        original_text_for_review=texts[i],observed_text=tokenizer.decode(base['tokens'][i][masks[si][i,0]],skip_special_tokens=True),
                        true_class=int(c[i]),reference_class=int(ac[i]),new_class=int(bc[i]),true_intensity=float(y[i]),
                        reference_intensity=float(a[i,3]),new_intensity=float(b[i,3]),absolute_error_reduction=float(gain[i])))
    pd.DataFrame(error_rows).to_csv(ROOT/'results/error_types.csv',index=False)
    pd.DataFrame(rescue_rows).to_csv(ROOT/'results/rescued_harmed.csv',index=False)
    pd.DataFrame(examples).to_csv(ROOT/'results/representative_cases.csv',index=False)
    save_json(ROOT/'results/diagnostic_notes.json',dict(text_patterns_are_review_clues_not_causal_labels=True,
        original_text_used_only_for_review=True,missing_cases_show_observed_text=True,classification_labels={'0':'negative','1':'neutral','2':'positive'},
        strong_negative='y <= -1.5',strong_positive='y >= 1.5',neutral='y == 0',models=list(predictions)))
    print('Saved error types, rescue/harm counts and',len(examples),'examples')

if __name__=='__main__':main()
