"""Standalone inference using exported weights; no training caches, teachers, labels, or old checkpoints."""
from common import *
from data import *
from engine import load_model,predict
import argparse,pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.set_num_threads(4)
    manifest=json.loads((ROOT/'deployment/manifest.json').read_text())
    assert sha256(ENCODER/'model.safetensors')==manifest['encoder_sha256']
    models=[]
    for name,digest in manifest['checkpoints'].items():
        p=ROOT/'deployment'/name;assert sha256(p)==digest;models.append(load_model(p)[0])
    ids,d=load_special();base=adapt(d);stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    view=make_view(base,TextEncoder(),stats,dict(subset='',rate=0.,position='none'),0)
    tv=torch_view(view);out=np.mean([predict(m,tv,backend=manifest['backend']) for m in models],axis=0)
    labels=['negative','neutral','positive']
    pd.DataFrame([dict(sample_id=s,polarity=labels[int(out[i,:3].argmax())],intensity=float(out[i,3])) for i,s in enumerate(ids)]).to_csv(args.output,index=False)
    print('Standalone package inference finished.',flush=True)

if __name__=='__main__':main()
