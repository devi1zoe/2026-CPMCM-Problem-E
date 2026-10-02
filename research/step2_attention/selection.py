from common import *
from engine import score
import pandas as pd


def main():
    p=json.loads((ROOT/'protocol.json').read_text());six=pd.read_csv(ROOT/'results/validation_six.csv')
    df=pd.read_csv(ROOT/'results/valid_metrics.csv');ens=df[df.seed=='ensemble'];rows=[]
    for name in ['old_gated']+p['models']:
        s=[six[(six.model==name)&(six.profile==v['name'])].iloc[0].to_dict() for v in p['validation_profiles']]
        clean=ens[(ens.model==name)&(ens.name=='clean')].iloc[0]
        robust=ens[(ens.model==name)&ens.primary].groupby('name')[['macro_f1','mae']].mean().mean()
        rows.append(dict(model=name,J=score(s),clean_accuracy=clean.accuracy,clean_f1=clean.macro_f1,clean_mae=clean.mae,clean_pearson=clean.pearson,
            neutral_recall=clean.neutral_recall,strong_negative_mae=clean.strong_negative_mae,strong_positive_mae=clean.strong_positive_mae,
            robust_f1=robust.macro_f1,robust_mae=robust.mae))
    table=pd.DataFrame(rows).set_index('model');ref=table.loc['old_gated'];guards={}
    for n in p['attention_candidates']:
        r=table.loc[n];guards[n]=dict(clean_f1=r.clean_f1>=ref.clean_f1-.005,clean_mae=r.clean_mae<=ref.clean_mae+.01,
            robust_f1=r.robust_f1>=ref.robust_f1+.01,robust_mae=r.robust_mae<=ref.robust_mae,neutral=r.neutral_recall>=ref.neutral_recall-.01,
            strong_negative=r.strong_negative_mae<=ref.strong_negative_mae+.02,strong_positive=r.strong_positive_mae<=ref.strong_positive_mae+.02)
        guards[n]={k:bool(v) for k,v in guards[n].items()}
    chosen=min(p['attention_candidates'],key=lambda n:table.loc[n,'J'])
    lock=dict(selected_attention=chosen,selection_rule=p['selection'],reference_guardrails=guards,
        meets_previous_replacement_criteria=all(guards[chosen].values()),protocol_sha256=sha256(ROOT/'protocol.json'),
        weights={f'{chosen}_{s}.pt':sha256(ROOT/f'checkpoints/{chosen}_{s}.pt') for s in p['seeds']},
        previous_outputs_replaced=False,test_previously_evaluated=True)
    path=ROOT/'selection_lock.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:save_json(path,lock)
    table.to_csv(ROOT/'results/selection_comparison.csv');print(table.to_string(),flush=True);print('LOCKED',chosen,guards[chosen],flush=True)


if __name__=='__main__':main()
