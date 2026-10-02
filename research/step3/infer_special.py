"""附件4全量预测、可复核模态贡献、局部扰动证据和真实视频关键帧。"""
from common import *
from engine import *
from explain import *
from data import adapt,make_view
import pandas as pd
import av

def locate(mapping,a,b):
    ts=[t for t in mapping['tokens'] if a<=t['slot']<b]
    lexical=[t for t in ts if t['word_indices']]
    if not ts:return dict(text='',char_start=None,char_end=None,time_status='no_text_mapping',start_seconds=None,end_seconds=None)
    lo=min(t['char_start'] for t in ts);hi=max(t['char_end'] for t in ts)
    timed=[t for t in lexical if t['candidate_start'] is not None]
    candidate=[min(t['candidate_start'] for t in timed),max(t['candidate_end'] for t in timed)] if timed else [None,None]
    usable=bool(lexical and all(t['timing_usable'] for t in lexical) and not mapping['requires_timing_review'])
    return dict(text=mapping['raw_text'][lo:hi],char_start=lo,char_end=hi,
        start_seconds=candidate[0] if usable else None,end_seconds=candidate[1] if usable else None,
        candidate_start_seconds=candidate[0],candidate_end_seconds=candidate[1],
        time_status='estimated_CTC' if usable else 'unverified_candidate',
        min_ctc_score=float(min(t['confidence'] for t in lexical if t['confidence'] is not None)) if lexical else None,
        mapping='task-aligned feature slots -> exact tokenizer offsets -> estimated CTC word interval',
        mapping_limitation='not official feature timestamps; punctuation has no independent acoustic interval')

def extract_frame(path,target,outfile,interval=None):
    best=None;inside=None
    with av.open(str(path)) as container:
        stream=container.streams.video[0];origin=float((stream.start_time or 0)*stream.time_base)
        for index,frame in enumerate(container.decode(video=0)):
            if frame.pts is None:continue
            time=float(frame.pts*frame.time_base)-origin
            distance=abs(time-target)
            if best is None or distance<best[0]:best=(distance,time,index,frame.to_image())
            if interval and interval[0]<=time<=interval[1] and (inside is None or distance<inside[0]):inside=(distance,time,index,frame.to_image())
    selected=inside or best;assert selected is not None
    image=selected[3];image.thumbnail((360,260));image.save(outfile,quality=82)
    return dict(pts_seconds=float(selected[1]),frame_index=int(selected[2]),inside_estimated_interval=inside is not None,
        image=outfile.relative_to(ROOT/'results').as_posix(),type='estimated_local_evidence_frame' if interval else 'context_preview_not_localized_evidence')

def main():
    torch.set_num_threads(4)
    models,lock=selected_models();protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha256(ROOT/'protocol.json')==lock['protocol_sha256']
    raw=load_special();base=adapt(raw);encoder=TextEncoder();stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
    view=make_view(base,encoder,stats,dict(subset='',rate=0.,position='random'),0)
    view={k:view[k] for k in ['text','audio','vision','p','o','quality']}
    out=predictions(models,view);target=out[:,:3].argmax(1)
    values=coalition_values(models,view);refs=dict(np.load(ROOT/'checkpoints/training_reference.npz'))
    refvalues=coalition_values(models,view,refs)
    records,phi,main_mod=modality_records(raw['id'],values,refvalues,target)
    pd.DataFrame(records).to_csv(ROOT/'results/special_modality_contributions.csv',index=False)
    ig32,res32=integrated_gradients(models,view,target,nodes=32)
    ig,residual,used_nodes=refine_integrals(models,view,target,ig32,res32);chosen,bank=rank_windows(ig,view['p'],.2)
    engine=ProbeEngine(models,raw,view,encoder)
    specs=[];indices=[]
    for i in range(20):
        for m in range(3):
            for a,b in bank[i]:specs.append((i,m,a,b,'drop'));indices.append((i,m,a,b))
    probed=engine.probe(specs)
    np.savez_compressed(ROOT/'results/special_explanation_arrays.npz',prob=out[:,:3],intensity=out[:,3],target=target,
        values=values,reference_values=refvalues,phi=phi,ig=ig,ig_residual=residual,ig_windows=chosen,ig_nodes=used_nodes,original_ig_residual32=res32,
        probe_indices=np.array(indices),probe_outputs=probed)
    local_rows=[];explanations=[];csvrows=[]
    for i,id_ in enumerate(raw['id']):
        mapping=json.loads((ROOT/f'results/mapping/{id_}.json').read_text())
        evidence=[];distributions={};details=[r for r in records if r['sample_id']==id_]
        for m,name in enumerate(NAMES):
            positions=[j for j,(ii,mm,_,_) in enumerate(indices) if ii==i and mm==m]
            drops=out[i,target[i]]-probed[positions,target[i]]
            ys=out[i,3]-probed[positions,3]
            windows_m=[indices[j][2:] for j in positions]
            candidate=tuple(chosen[i,m]);candidate_index=windows_m.index(candidate)
            support_index=int(np.argmax(drops));inhibition_index=int(np.argmin(drops))
            # 确认后的证据可来自穷举最大删除效应；独立忠实性评价仍仅使用预先IG排名。
            key_index=support_index if drops[support_index]>.001 else int(np.argmax(abs(drops)))
            a,b=windows_m[key_index]
            location=locate(mapping,a,b)
            item=dict(modality=name,slot_start=int(a),slot_end=int(b),**location,
                fixed_target_class=int(target[i]),class_probability_drop=float(drops[key_index]),intensity_delta=float(ys[key_index]),
                role='supporting' if drops[key_index]>.001 else ('inhibitory' if drops[key_index]<-.001 else 'weak'),
                selection='largest positive raw deletion effect among equal-length windows; otherwise strongest absolute effect',
                ig_candidate=dict(slot_start=int(candidate[0]),slot_end=int(candidate[1]),
                    ranking_score=float(ig[i,m,candidate[0]:candidate[1]].sum()),confirmed_deletion_drop=float(drops[candidate_index])),
                maximum_inhibitory_drop=float(drops[inhibition_index]))
            evidence.append(item)
            distributions[name]=[]
            for k,(lo,hi) in enumerate(windows_m):
                row=dict(sample_id=id_,modality=name,slot_start=int(lo),slot_end=int(hi),class_probability_drop=float(drops[k]),intensity_delta=float(ys[k]),
                    ig_window_sum=float(ig[i,m,lo:hi].sum()),is_ig_candidate=k==candidate_index,is_selected_evidence=k==key_index)
                local_rows.append(row);distributions[name].append(row)
        visual=evidence[2]
        interval=[visual['start_seconds'],visual['end_seconds']] if visual['time_status']=='estimated_CTC' else None
        center=sum(interval)/2 if interval else mapping['duration']/2
        frame=extract_frame(SPECIAL/'videos'/(id_+'.mp4'),center,ROOT/f'results/cards/{id_}_frame.jpg',interval)
        evidence[2]['keyframe']=frame
        row=dict(sample_id=id_,polarity=['negative','neutral','positive'][target[i]],intensity=float(out[i,3]),
            main_modality=details[0]['main_modality'],contribution_text=float(phi[i,0,0]),contribution_audio=float(phi[i,1,0]),
            contribution_vision=float(phi[i,2,0]),evidence=json.dumps(evidence,ensure_ascii=False,separators=(',',':')))
        csvrows.append(row)
        record=dict(**{k:v for k,v in row.items() if k!='evidence'},class_id=int(target[i]),probabilities=out[i,:3].tolist(),
            intensity_seed_std=float(values[:,i,7,3].std(ddof=1)),target='fixed original predicted class probability',
            baseline_probability=float(values.mean(0)[i,0,target[i]]),full_probability=float(out[i,target[i]]),
            modalities=details,evidence=evidence,raw_text=str(raw['raw_text'][i]),window_distributions=distributions,
            ig_completeness_abs_error=float(abs(residual[i])),requires_timing_review=bool(mapping['requires_timing_review']),
            timing_review_reasons=mapping['reasons'],diagnostic_wer=mapping['diagnostic_wer'],source_video=mapping['video_source'],
            source_video_sha256=mapping['video_sha256'],duration=mapping['duration'],keyframe=frame,
            model_truncated_text=bool(mapping['unrepresented_text_tail'].strip()))
        save_json(ROOT/f'results/cards/{id_}.json',record);explanations.append(record)
        print('explained',id_,row['polarity'],row['main_modality'],'timing_review',record['requires_timing_review'],flush=True)
    df=pd.DataFrame(csvrows);assert len(df)==20 and df.sample_id.is_unique and df.intensity.between(-3,3).all()
    df.to_csv(ROOT/'results/附件4_问题三预测与解释.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(local_rows).to_csv(ROOT/'results/special_local_effects.csv',index=False)
    save_json(ROOT/'results/special_explanations.json',explanations)
    save_json(ROOT/'results/special_inference_audit.json',dict(n=20,no_labels_used=True,input_version='aligned',
        selection_sha256=sha256(ROOT/'selection_lock.json'),model=lock['selected'],
        primary_main_counts=df.main_modality.value_counts().to_dict(),polarity_counts=df.polarity.value_counts().to_dict(),
        timing_review_sample_ids=[r['sample_id'] for r in explanations if r['requires_timing_review']],
        text_offsets_available_for_all=all(all(e['char_start'] is not None for e in r['evidence']) for r in explanations),
        localized_evidence_counts={name:sum(r['evidence'][m]['time_status']=='estimated_CTC' for r in explanations) for m,name in enumerate(NAMES)}))

if __name__=='__main__':main()
