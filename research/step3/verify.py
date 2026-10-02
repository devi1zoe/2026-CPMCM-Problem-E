"""解释关键性质检查：固定目标、Shapley完备性、IG已知线性解、文本删除无泄露。"""
from common import *
from engine import *
from explain import integrated_gradients,ProbeEngine
from data import load_attachment2

class LinearProbability(torch.nn.Module):
    def forward(self,b):
        terms=[b[name].float()[:,:,0].mean(1)*w for name,w in zip(NAMES,[.02,.03,.04])]
        f=.25+sum(terms);p=torch.stack([f,torch.ones_like(f)*.3,.7-f],-1)
        return p.log(),sum(terms),torch.ones(len(f),3,device=f.device)/3

def main():
    torch.set_num_threads(4)
    val=np.zeros((2,8,4));expected=np.array([[.1,-.2,.3],[.2,.1,-.1]])
    for i in range(2):
        for mask in range(8):
            val[i,mask,i]=.4+sum(expected[i,j] for j in range(3) if mask&(1<<j))
            val[i,mask,3]=1+sum((j+1)*.2 for j in range(3) if mask&(1<<j))
    phi,res=shapley(val,np.array([0,1]));assert np.allclose(phi[:,:,0],expected) and np.max(abs(res))<1e-7
    toy=dict(p=np.ones((2,50),bool),o=np.ones((2,3,50),bool),quality=np.ones((2,3,3),np.float32),
        text=np.ones((2,50,768),np.float32),audio=np.full((2,50,74),2,np.float32),vision=np.full((2,50,35),3,np.float32))
    ig,err=integrated_gradients([LinearProbability().cuda()],toy,np.zeros(2,int),nodes=8)
    assert np.allclose(ig.sum(-1),[.02,.06,.12],atol=1e-6) and np.max(abs(err))<1e-6
    models,lock=selected_models();raw=load_attachment2()['train'];raw={k:v[:2] for k,v in raw.items()}
    clean={k:v[:2] for k,v in load_view('train').items()};engine=ProbeEngine(models,raw,clean)
    specs=[(0,0,3,7,'drop'),(1,0,2,5,'drop')];a=engine.probe(specs)
    for i,_,lo,hi,_ in specs:engine.base['tokens'][i,lo:hi]=2987
    b=engine.probe(specs);assert np.array_equal(a,b)
    keep=np.ones((2,3),bool);keep[:,0]=False
    absent=predictions(models,coalitional_view(clean,keep))
    zero=engine.probe([(i,0,0,50,'drop') for i in range(2)])
    assert np.allclose(absent,zero,atol=1e-6)
    save_json(ROOT/'results/verification.json',dict(shapley_known_signed_game=True,fixed_target_classes=True,
        ig_known_linear_probability=True,shapley_efficiency=True,deleted_tokens_cannot_leak=True,
        whole_text_absence_equals_input_deletion=True,selected_checkpoint_hashes=True))
    print('All explanation invariants passed.',flush=True)

if __name__=='__main__':main()
