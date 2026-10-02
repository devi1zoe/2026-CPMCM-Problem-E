"""附件4原文-词-子词-特征位置-媒体PTS映射；低可信候选不伪装官方时间。"""
from common import *
from alignment import extract_words,make_target,ctc_viterbi,words_from_path,word_error_rate
import numpy as np
import torch
import av
import pandas as pd
from transformers import AutoTokenizer,Wav2Vec2Processor,Wav2Vec2ForCTC

def audio_decode(path):
    with av.open(str(path)) as container:
        video=container.streams.video[0];origin=float((video.start_time or 0)*video.time_base)
        duration=float(video.duration*video.time_base);rate=float(video.average_rate)
        pts=[]
        for frame in container.decode(video=0):
            assert frame.pts is not None
            pts.append(float(frame.pts*frame.time_base)-origin)
    chunks=[]
    with av.open(str(path)) as container:
        resampler=av.AudioResampler(format='fltp',layout='mono',rate=16000)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):chunks.append((float(out.pts*out.time_base)-origin,out.to_ndarray().reshape(-1).copy()))
        for out in resampler.resample(None):chunks.append((float(out.pts*out.time_base)-origin,out.to_ndarray().reshape(-1).copy()))
    assert chunks
    start=min(t for t,_ in chunks);end=max(t+len(x)/16000 for t,x in chunks)
    waveform=np.zeros(round((end-start)*16000),np.float32);coverage=np.zeros(len(waveform),bool)
    for t,x in chunks:
        a=round((t-start)*16000);b=min(a+len(x),len(waveform));waveform[a:b]=x[:b-a];coverage[a:b]=True
    return waveform,start,max(duration,end),np.array(pts),rate,float(coverage.mean())

def main():
    torch.set_num_threads(4);torch.manual_seed(20260924)
    protocol=json.loads((ROOT/'protocol.json').read_text());rules=protocol['timing']['quality']
    d=load_special();tokenizer=AutoTokenizer.from_pretrained(ENCODER,local_files_only=True)
    processor=Wav2Vec2Processor.from_pretrained(ASSETS/'wav2vec2',local_files_only=True)
    model=Wav2Vec2ForCTC.from_pretrained(ASSETS/'wav2vec2',local_files_only=True).cuda().eval();model.requires_grad_(False)
    rows=[];all_tokens=[]
    for i,id_ in enumerate(d['id']):
        path=SPECIAL/'videos'/(id_+'.mp4');text=str(d['raw_text'][i]);words=extract_words(text)
        x,start,duration,pts,rate,coverage=audio_decode(path)
        inputs=processor(x,sampling_rate=16000,return_tensors='pt').input_values.cuda()
        with torch.inference_mode():logp=model(inputs).logits[0].log_softmax(-1).float().cpu().numpy()
        hypothesis=processor.decode(logp.argmax(-1))
        target,owners,chars=make_target(words,processor.tokenizer.get_vocab())
        stride=1;receptive=1
        for kernel,step in zip(model.config.conv_kernel,model.config.conv_stride):receptive+=(kernel-1)*stride;stride*=step
        centers=(np.arange(len(logp))*stride+(receptive-1)/2)/16000+start
        bounds=np.clip(np.c_[centers-stride/32000,centers+stride/32000],0,duration)
        error=None
        try:
            path_states,score=ctc_viterbi(logp,target,model.config.pad_token_id)
            wb,confidence,valid=words_from_path(path_states,target,owners,logp,words,bounds,rules['low_word_confidence'])
        except ValueError as e:
            error=str(e);wb=np.zeros((len(words),2));confidence=np.zeros(len(words));valid=np.zeros(len(words),bool)
            path_states=np.empty(0,int)
        reference=' '.join(w['normalized'] for w in words)
        wer=word_error_rate(reference,hypothesis)
        reasons=[]
        if confidence.mean()<rules['mean_confidence']:reasons.append('low_mean_CTC_confidence')
        if np.mean(confidence<rules['low_word_confidence'])>rules['max_low_fraction']:reasons.append('many_low_confidence_words')
        if wer>rules['max_diagnostic_wer']:reasons.append('ASR_transcript_mismatch_signal')
        if error:reasons.append('no_complete_CTC_path')
        tokenized=tokenizer(text,return_offsets_mapping=True,truncation=True,max_length=50,padding='max_length')
        token_match=np.array_equal(tokenized['input_ids'],d['text_bert'][i,0])
        if not token_match:reasons.append('token_interface_mismatch')
        tokens=[];same_word_groups=[]
        for t,(a,b) in enumerate(tokenized['offset_mapping']):
            if a==b:continue
            js=[j for j,w in enumerate(words) if min(b,w['char_end'])>max(a,w['char_start'])]
            reliable=bool(token_match and not reasons and js and all(valid[j] and confidence[j]>=rules['low_word_confidence'] for j in js))
            time=[float(min(wb[j,0] for j in js)),float(max(wb[j,1] for j in js))] if js and all(valid[j] for j in js) else None
            row=dict(sample_id=id_,slot=t,token=tokenizer.convert_ids_to_tokens(int(d['text_bert'][i,0,t])),token_id=int(d['text_bert'][i,0,t]),
                char_start=int(a),char_end=int(b),raw_fragment=text[a:b],word_indices=js,
                candidate_start=time[0] if time else None,candidate_end=time[1] if time else None,
                timing_usable=reliable,confidence=float(min(confidence[js])) if js else None,
                mapping='tokenizer_offsets + original_transcript_CTC' if js else 'punctuation_without_own_acoustic_interval')
            tokens.append(row);all_tokens.append(row)
        repeats={}
        for name in ['audio','vision']:
            tested=matched=0
            for j in range(len(words)):
                ts=[r['slot'] for r in tokens if j in r['word_indices']]
                if len(ts)>1:
                    tested+=1;matched+=int(np.allclose(d[name][i,ts],d[name][i,ts[0]],atol=1e-10,rtol=1e-7))
            repeats[name]=dict(multi_piece_words=tested,identical_feature_groups=matched)
        last_char=max([r['char_end'] for r in tokens] or [0])
        record=dict(sample_id=id_,raw_text=text,duration=duration,video_fps=rate,video_frames=len(pts),video_source=path.relative_to(PROJECT).as_posix(),
            video_sha256=sha256(path),audio_coverage_fraction=coverage,diagnostic_asr=hypothesis,diagnostic_wer=wer,
            mean_ctc_confidence=float(confidence.mean()),requires_timing_review=bool(reasons),reasons=reasons,
            words=words,tokens=tokens,token_ids_match=bool(token_match),feature_repetition_audit=repeats,
            represented_char_end=last_char,unrepresented_text_tail=text[last_char:],
            timing_claim='estimated CTC word intervals, not official feature timestamps; aligned AV correspondence follows task specification',alignment_error=error)
        save_json(ROOT/f'results/mapping/{id_}.json',record)
        np.savez_compressed(ROOT/f'cache/ctc_{id_}.npz',logp=logp.astype(np.float16),path=path_states,targets=target,owners=owners,
            frame_bounds=bounds,word_bounds=wb,word_confidence=confidence,all_video_pts=pts)
        rows.append(dict(sample_id=id_,duration=duration,words=len(words),feature_slots=len(tokens),usable_token_times=sum(r['timing_usable'] for r in tokens),
            diagnostic_wer=wer,mean_ctc_confidence=float(confidence.mean()),requires_timing_review=bool(reasons),token_ids_match=bool(token_match),
            audio_repeated_groups=repeats['audio']['identical_feature_groups'],vision_repeated_groups=repeats['vision']['identical_feature_groups'],
            multi_piece_groups=repeats['audio']['multi_piece_words'],truncated_text=bool(text[last_char:].strip()),reasons=';'.join(reasons)))
        print('mapped',id_,'WER',round(wer,3),'confidence',round(float(confidence.mean()),3),'review',bool(reasons),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/time_mapping_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{**r,'word_indices':json.dumps(r['word_indices'])} for r in all_tokens]).to_csv(ROOT/'results/token_time_mapping.csv',index=False,encoding='utf-8-sig')
    print('All 20 videos mapped, with explicit uncertainty flags.',flush=True)

if __name__=='__main__':main()
