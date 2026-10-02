"""问题三逐项交付核验与累计50MB预算。核验真实输出，不用布尔清单替代证据。"""
from common import *
from engine import *
from explain import rank_windows
from data import adapt,make_view,TextEncoder
import zipfile,platform,importlib.metadata
import pandas as pd
from html.parser import HTMLParser
from urllib.parse import unquote
import av

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.cards=0
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='details' and d.get('class')=='card':self.cards+=1
        for key in ['href','src','poster']:
            if key in d:self.links.append(d[key])

def main():
    torch.set_num_threads(4)
    models,lock=selected_models();protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha256(ROOT/'protocol.json')==lock['protocol_sha256']
    assert lock['selected']==min(lock['scores'],key=lock['scores'].get)
    assert sha256(DATA/'附件2-数据集特征文件/aligned_50.pkl')==protocol['input_sha256']
    checks=json.loads((ROOT/'results/verification.json').read_text());assert all(checks.values())
    assert sha256(ROOT/'checkpoints/normalization.npz')==sha256(P2/'checkpoints/normalization.npz')
    reference=dict(np.load(ROOT/'checkpoints/training_reference.npz'));train=load_view('train')
    for j,name in enumerate(NAMES):
        recomputed=np.median(train[name][train['o'][:,j]].astype(np.float32),axis=0)
        assert np.array_equal(recomputed,reference[name]),name
    del train
    validation=pd.read_csv(ROOT/'results/validation_prediction_metrics.csv')
    for name,value in lock['scores'].items():
        r=validation[(validation.model==name)&(validation.seed=='ensemble')].iloc[0]
        assert abs(r.mae+1-r.macro_f1-value)<1e-8
    for seed in protocol['seeds']:
        history=json.loads((ROOT/f'results/history_{seed}.json').read_text())
        ck=torch.load(ROOT/f'checkpoints/final_{seed}.pt',map_location='cpu',weights_only=True)
        assert abs(ck['score']-min(h['score'] for h in history))<1e-5
        assert (ROOT/f'results/completed_{seed}.json').exists()
    arrays=np.load(ROOT/'results/validation_coalitions.npz');target=arrays['target']
    for name in ['values','reference_values']:
        assert arrays[name].shape==(3,728,8,4)
        for values in arrays[name]:
            phi,res=shapley(values,target);assert abs(res).max()<2e-6
    ig=np.load(ROOT/'results/validation_ig.npz');assert len(ig['ig'])==728 and np.max(abs(ig['residual']))<=.001
    clean=load_view('valid');ranked,_=rank_windows(ig['ig'],clean['p']);assert np.array_equal(ranked,ig['chosen'])
    probes=np.load(ROOT/'results/validation_probe_outputs.npz');cube=probes['outputs'];specs=probes['specs']
    assert cube.shape==(728,3,6,3,4) and specs.shape==(39312,5)
    assert np.isfinite(cube).all() and np.allclose(cube[:,:,:,:,:3].sum(-1),1,atol=1e-6)
    fidelity=pd.read_csv(ROOT/'results/validation_fidelity.csv');assert len(fidelity)==2184
    saved=np.load(ROOT/'results/validation_predictions.npz');full=saved['prob'][np.arange(728),target]
    for i in range(728):
        for m in range(3):
            r=fidelity.iloc[i*3+m]
            assert abs((full[i]-cube[i,m,0,0,target[i]])-r.deletion_drop)<1e-7
            assert abs((cube[i,m,1:,0,target[i]].mean()-cube[i,m,0,0,target[i]])-r.deletion_advantage)<1e-7
            block=specs[(i*3+m)*18:(i*3+m+1)*18]
            assert len(np.unique(block[:,3]-block[:,2]))==1
    assert len(pd.read_csv(ROOT/'results/validation_stability.csv'))==96
    # 独立重新载入20条输入与最终权重，复算预测，并与CSV/解释数据交叉核验。
    raw=load_special();b=adapt(raw);encoder=TextEncoder();stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    view=make_view(b,encoder,stats,dict(subset='',rate=0.,position='random'),0)
    out=predictions(models,view)
    csv=pd.read_csv(ROOT/'results/附件4_问题三预测与解释.csv',dtype={'sample_id':str})
    assert list(csv.columns)==['sample_id','polarity','intensity','main_modality','contribution_text','contribution_audio','contribution_vision','evidence']
    assert len(csv)==20 and list(csv.sample_id)==raw['id'] and csv.sample_id.is_unique
    assert np.allclose(csv.intensity,out[:,3],atol=1e-5,rtol=0)
    assert csv.polarity.tolist()==[['negative','neutral','positive'][j] for j in out[:,:3].argmax(1)]
    special=json.loads((ROOT/'results/special_explanations.json').read_text());assert len(special)==20
    arr=np.load(ROOT/'results/special_explanation_arrays.npz');assert np.max(abs(arr['ig_residual']))<=.001
    phi,res=shapley(arr['values'].mean(0),arr['target']);assert np.allclose(phi,arr['phi'],atol=1e-7)
    local=pd.read_csv(ROOT/'results/special_local_effects.csv',dtype={'sample_id':str})
    source_audit=json.loads((ROOT/'results/interface_audit.json').read_text())
    frame_checks=[]
    for i,record in enumerate(special):
        id_=record['sample_id'];assert id_==raw['id'][i]
        assert sha256(SPECIAL/(id_+'.pkl'))==source_audit['input_sha256'][id_+'.pkl']
        assert sha256(PROJECT/record['source_video'])==record['source_video_sha256']
        evidence=json.loads(csv.iloc[i].evidence);assert len(evidence)==3
        mapping=json.loads((ROOT/f'results/mapping/{id_}.json').read_text())
        assert mapping['token_ids_match'] and record['raw_text']==str(raw['raw_text'][i])
        for m,e in enumerate(evidence):
            assert e['modality']==NAMES[m] and e['fixed_target_class']==record['class_id']
            assert 0<=e['char_start']<e['char_end']<=len(record['raw_text'])
            assert record['raw_text'][e['char_start']:e['char_end']]==e['text']
            assert 1<=e['slot_start']<e['slot_end']<=49
            match=local[(local.sample_id==id_)&(local.modality==NAMES[m])&(local.slot_start==e['slot_start'])&(local.slot_end==e['slot_end'])]
            assert len(match)==1 and abs(match.iloc[0].class_probability_drop-e['class_probability_drop'])<1e-7
            assert abs(csv.iloc[i][f'contribution_{NAMES[m]}']-phi[i,m,0])<1e-7
            if e['time_status']=='estimated_CTC':
                assert not mapping['requires_timing_review'] and 0<=e['start_seconds']<e['end_seconds']<=mapping['duration']+.001
                tokens=[t for t in mapping['tokens'] if e['slot_start']<=t['slot']<e['slot_end'] and t['word_indices']]
                assert tokens and all(t['timing_usable'] for t in tokens)
            else:assert e['start_seconds'] is None and e['end_seconds'] is None
        # 独立解码记录的帧序号，核对真实PTS以及输出图片来自该原始帧。
        import io
        frame=record['keyframe'];found=False
        with av.open(str(PROJECT/record['source_video'])) as container:
            stream=container.streams.video[0];origin=float((stream.start_time or 0)*stream.time_base)
            for index,vf in enumerate(container.decode(video=0)):
                if index==frame['frame_index']:
                    pts=float(vf.pts*vf.time_base)-origin;assert abs(pts-frame['pts_seconds'])<1e-7
                    im=vf.to_image();im.thumbnail((360,260));buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=82)
                    assert buffer.getvalue()==(ROOT/'results'/frame['image']).read_bytes()
                    found=True;break
        assert found
        if frame['type']=='context_preview_not_localized_evidence':assert record['evidence'][2]['start_seconds'] is None
        frame_checks.append(dict(sample_id=id_,actual_frame_index=frame['frame_index'],pts_verified=True,pixels_verified=True))
    parser=Links();parser.feed((ROOT/'results/explanation_review.html').read_text());assert parser.cards==20
    for link in parser.links:
        if not link.startswith(('http:','https:','#')):assert (ROOT/'results'/unquote(link)).resolve().is_file(),link
    required=['模型定义.md','代码说明.md','问题三实验报告.md','results/error_groups.csv','results/representative_errors.csv',
        'results/fidelity_correct_vs_wrong.csv','results/time_mapping_audit.csv','results/token_time_mapping.csv']
    for p in required:assert (ROOT/p).is_file() and (ROOT/p).stat().st_size>0
    assert len(list((ROOT/'results/figures').glob('*.png')))==8
    assert len(list((ROOT/'results/mapping').glob('*.json')))==20
    save_json(ROOT/'environment.json',dict(python=sys.version,cuda_runtime=torch.version.cuda,
        packages={p:importlib.metadata.version(p) for p in ['torch','numpy','transformers','tokenizers','pandas','scipy','scikit-learn','matplotlib','av','Pillow','num2words']},
        root_environment_modified=False,new_environment_created=False,
        pretrained_assets=json.loads((PROJECT/'step1/assets_manifest.json').read_text()),
        reuse_manifest=json.loads((ROOT/'reuse_manifest.json').read_text()),
        gpu_models=[torch.cuda.get_device_name(j) for j in range(torch.cuda.device_count())]))
    audit=dict(source_hashes=True,training_only_reference_recomputed=True,validation_selection_recomputed=True,
        all_8_coalitions_checked=True,fixed_targets_and_shapley_efficiency=True,all_728_ig_within_tolerance=True,
        all_2184_local_fidelity_rows_recomputed=True,equal_random_window_budgets=True,probe_count=39312,
        independent_special_prediction_recompute=True,all_20_csv_json_and_char_offsets_checked=True,
        no_unverified_time_claims=True,original_frame_pixels_and_pts=frame_checks,html_cards=parser.cards,html_links=len(parser.links),
        prediction_and_explanation_validation=True,figures=8,
        limitations='15/16 need whole-clip timing review; other low-confidence local windows remain unlocalized; text/feature-slot evidence preserved for all 20')
    save_json(ROOT/'results/completion_audit.json',audit)
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in ['cache','__pycache__'] for x in p.relative_to(ROOT).parts)
        and p.suffix!='.zip' and p.name not in ['package.log','package_size.json','delivery_manifest.json']
        and not p.name.startswith('coalition_aug_')]
    manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(files)}
    save_json(ROOT/'delivery_manifest.json',manifest);files.append(ROOT/'delivery_manifest.json')
    output=ROOT/'问题三交付.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,arcname='step3/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    prior=sum((PROJECT/f'step{k}/问题{"一" if k==1 else "二"}交付.zip').stat().st_size for k in [1,2])
    total=prior+output.stat().st_size;assert total<=50_000_000,(total,'50MB limit exceeded')
    save_json(ROOT/'results/package_size.json',dict(step3_bytes=output.stat().st_size,three_problem_total_bytes=total,
        remaining_to_50MB=50_000_000-total,zip_sha256=sha256(output)))
    print(json.dumps(dict(files=len(files),step3_bytes=output.stat().st_size,total_bytes=total,all_checks_passed=True),indent=2),flush=True)

if __name__=='__main__':main()
