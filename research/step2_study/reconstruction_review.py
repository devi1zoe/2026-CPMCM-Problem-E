"""Evaluate feature estimates using held-out artificial deletion targets only."""
from common import *
from data import *
from engine import load_model
from evaluation_data import EvaluationViews,profiles_all
import pandas as pd

@torch.inference_mode()
def main():
    torch.set_num_threads(2);protocol=json.loads((ROOT/'protocol.json').read_text());raw=load_attachment2()['valid'];views=EvaluationViews(raw,'valid',protocol)
    clean=dict(np.load(ROOT/'cache/valid_0.npz'));training=dict(np.load(ROOT/'cache/train_0.npz'));models=[];targets=[];means=[];rows=[]
    for seed in protocol['seeds']:
        model,ck=load_model(ROOT/f'checkpoints/reconstruct_{seed}.pt');models.append(model);target=[];average=[]
        for m,name in enumerate(['text','audio','vision']):
            target.append(model.target_input[str(m)](torch.as_tensor(clean[name],device='cuda').float()))
            total=torch.zeros(64,device='cuda');count=0
            for start in range(0,len(training['p']),128):
                observed=torch.as_tensor(training['o'][start:start+128,m],device='cuda')
                projected=model.target_input[str(m)](torch.as_tensor(training[name][start:start+128],device='cuda').float())
                total+=projected[observed].sum(0);count+=int(observed.sum())
            average.append(total/count)
        targets.append(torch.stack(target,1));means.append(torch.stack(average))
    for si,profile in enumerate(profiles_all()):
        if profile['name']=='clean':continue
        view=views.get(profile);mask=torch.as_tensor(view['removed']&views.base['o'],device='cuda');counts=mask.sum((0,2)).cpu().numpy()
        for seed,model,target,mean in zip(protocol['seeds'],models,targets,means):
            sums=np.zeros(3);reference=np.zeros(3)
            for start in range(0,len(view['p']),128):
                b=torch_view({k:v[start:start+128] for k,v in view.items()});out=model(b);supervised=mask[start:start+128]
                for m in range(3):
                    keep=supervised[:,m];true=target[start:start+128,m][keep]
                    sums[m]+=float((out['reconstruction'][:,m][keep]-true).square().sum())
                    reference[m]+=float((mean[m]-true).square().sum())
            for m,name in enumerate('TAV'):
                if counts[m]:rows.append(dict(seed=seed,**profile,modality=name,deleted_observed_slots=int(counts[m]),
                    reconstruction_mse=float(sums[m]/(counts[m]*64)),training_mean_mse=float(reference[m]/(counts[m]*64))))
        if si%24==0:print('Feature reconstruction review',si,'/252',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/reconstruction_quality.csv',index=False)
    save_json(ROOT/'results/reconstruction_review_complete.json',dict(rows=len(rows),targets_used_only_for_evaluation=True,
        checkpoint_hashes={str(s):sha256(ROOT/f'checkpoints/reconstruct_{s}.pt') for s in protocol['seeds']}))

if __name__=='__main__':main()
