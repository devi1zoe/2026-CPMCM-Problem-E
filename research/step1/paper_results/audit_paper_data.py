"""独立核对论文数字：直接读附件一、逐样本数组和原视频音轨，不调用原统计函数。"""
from pathlib import Path
import csv,json,re
from collections import Counter
import av,numpy as np,openpyxl
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PROJECT=ROOT.parent;DATA=ROOT/'results'
config=json.loads((ROOT/'config.json').read_text());threshold=config['silence_rms_threshold']

def rows(name):
 with (DATA/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def intersect(a,b):
 return np.maximum(0,np.minimum(a[:,None,1],b[None,:,1])-np.maximum(a[:,None,0],b[None,:,0]))

def pool(a,b,x,valid):
 w=intersect(a,b);w[:,~valid]=0;den=w.sum(axis=1);observed=den>0
 w[observed]/=den[observed,None]
 return (w@x).astype(np.float32),observed,w

def edit_distance(a,b):
 # 完整二维动态规划，与生产程序的一维滚动数组写法不同。
 a=a.split();b=b.split();d=np.zeros((len(a)+1,len(b)+1),dtype=int)
 d[:,0]=np.arange(len(a)+1);d[0]=np.arange(len(b)+1)
 for i,x in enumerate(a,1):
  for j,y in enumerate(b,1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(x!=y))
 return int(d[-1,-1]),len(a)

def decode_audio(path,origin):
 clips=[]
 with av.open(str(path)) as c:
  res=av.AudioResampler(format='fltp',layout='mono',rate=16000)
  for f in c.decode(audio=0):
   for out in res.resample(f):clips.append((float(out.pts*out.time_base)-origin,out.to_ndarray().reshape(-1)))
  for out in res.resample(None):clips.append((float(out.pts*out.time_base)-origin,out.to_ndarray().reshape(-1)))
 start=min(t for t,x in clips);end=max(t+len(x)/16000 for t,x in clips)
 signal=np.zeros(round((end-start)*16000),np.float32);covered=np.zeros(len(signal),bool)
 for t,x in clips:
  a=round((t-start)*16000);b=min(a+len(x),len(signal));signal[a:b]=x[:b-a];covered[a:b]=True
 return start,signal,covered

saved_windows={(r['sample_id'],float(r['window_seconds'])):r for r in rows('window_comparison.csv')}
saved_samples={r['sample_id']:r for r in rows('sample_summary.csv')}
word_rows=rows('word_alignment.csv');modality_rows=rows('modality_summary.csv')
xlsx=next((PROJECT/'E题_材料阅读/E题数据/附件1-数据集原始多模态样本').rglob('label-100.xlsx'))
wb=openpyxl.load_workbook(xlsx,read_only=True,data_only=True);data=list(wb['label'].values);wb.close()
labels={str(r[0])+'$_$'+str(r[1]):dict(zip(data[0],r)) for r in data[1:]}
assert len(labels)==100 and len(word_rows)==1932 and len(modality_rows)==300
modality_lookup={(r['sample_id'],r['modality']):r for r in modality_rows}
word_lookup={(r['sample_id'],int(r['index'])):r for r in word_rows}
count=Counter();confidence=[];diagnostics=[];recomputed_windows=[];audit_rows=[];constant=[];unassigned=Counter();sensitivity=Counter()
maxdiff=Counter();durations=[];candidate_typical=[]
for directory in sorted((DATA/'samples').iterdir()):
 if not (directory/'metadata.json').is_file():continue
 sid=directory.name;m=json.loads((directory/'metadata.json').read_text());label=labels[sid]
 assert m['raw_text']==label['text'] and m['label']==float(label['label']) and m['annotation']==label['annotation']
 with np.load(directory/'features.npz') as z, np.load(directory/'quality_masks.npz') as q:
  n=len(m['words']);count['samples']+=1;count['words']+=n;count['subwords']+=len(z['subword_ids'])
  count['sampled_frames']+=len(z['vision_pts']);count['valid_face_frames']+=int(z['vision_frame_valid'].sum())
  count['audio_records']+=len(z['audio_frames']);count['low_energy']+=int(z['audio_silence'].sum())
  count['audio_valid_records']+=int(z['audio_frame_valid'].sum());count['candidate_words']+=int(z['alignment_valid'].sum())
  count['padding']+=m['padding_count'];count['truncated']+=m['truncated_words']
  for s in m['subwords']:
   if s['word_index']<0:unassigned[s['token']]+=1
  for mod,dim in [('text',768),('audio',25),('vision',52)]:
   assert z[mod].shape==(n,dim);count[mod+'_observed_words']+=int(z[mod+'_valid'].sum())
   row=modality_lookup[(sid,mod)]
   assert int(row['feature_dimension'])==dim and int(row['stored_positions'])==n and int(row['valid_positions'])==int(z[mod+'_valid'].sum())
   assert int(row['padding_count'])==0 and row['alignment_granularity']=='original_transcript_word'
   assert np.isclose(float(row['duration_seconds']),m['duration'])
  states=z['ctc_state_path'];idx=(states-1)//2;c=[]
  for i,w in enumerate(m['words']):
   keep=(states%2==1)&np.isin(idx,np.flatnonzero(z['ctc_character_word']==i))
   tokens=z['ctc_target_ids'][idx[keep]];value=float(np.exp(z['ctc_log_probs'][keep,tokens].mean()));c.append(value)
   maxdiff['ctc_confidence']=max(maxdiff['ctc_confidence'],abs(value-float(z['word_confidence'][i])))
   assert np.isclose(value,z['word_confidence'][i],rtol=1e-6,atol=1e-9)
   rec=word_lookup[(sid,i)];assert rec['original']==w['original']
   assert np.allclose([float(rec['start']),float(rec['end'])],z['word_bounds'][i])
  c=np.asarray(c);confidence.extend(c.tolist());low=c<.1
  edits,nref=edit_distance(m['ctc_target_text'].replace('|',' '),m['diagnostic_asr']);wer=edits/nref
  assert np.isclose(wer,m['diagnostic_wer']);diagnostics.append(wer)
  risk=bool(c.mean()<.2 or low.mean()>.5 or wer>.8 or m['alignment_error'] is not None)
  expected=z['alignment_valid'] & (c>=.1) & ~np.full(n,risk)
  np.testing.assert_array_equal(expected,q['timing_usable'])
  quality=json.loads((directory/'quality.json').read_text());assert quality['requires_transcript_review']==risk
  saved=saved_samples[sid]
  expected_fields={'words':n,'subwords':len(z['subword_ids']),
   'sampled_video_frames':len(z['vision_pts']),'acoustic_frames':len(z['audio_frames']),
   'aligned_words':int(z['alignment_valid'].sum()),'low_confidence_words':int(low.sum()),
   'mean_word_confidence':float(c.mean()),'face_frame_ratio':float(z['vision_frame_valid'].mean()),
   'no_face_frames':int((~z['vision_frame_valid']).sum()),'silent_audio_frames':int(z['audio_silence'].sum()),
   'audio_frames_observed':int(z['audio_frame_valid'].sum()),'diagnostic_wer':wer,
   'padding_count':m['padding_count'],'truncated_words':m['truncated_words'],'duration_seconds':m['duration']}
  expected_fields.update({mod+'_observed_words':int(z[mod+'_valid'].sum()) for mod in ['text','audio','vision']})
  for key,value in expected_fields.items():assert np.isclose(float(saved[key]),value,rtol=1e-6,atol=1e-8),(sid,key)
  assert (saved['requires_transcript_review']=='True')==risk
  for mod in ['text','audio','vision']:
   np.testing.assert_array_equal(q[mod+'_aligned_usable'],expected & z[mod+'_valid'])
  count['risk_samples']+=risk;count['risk_words']+=n if risk else 0;count['low_words']+=int(low.sum())
  count['risk_low_words']+=int(low.sum()) if risk else 0;count['risk_high_words']+=int((~low).sum()) if risk else 0
  count['low_outside_risk']+=int(low.sum()) if not risk else 0;count['time_usable']+=int(expected.sum())
  joint=expected & z['text_valid'] & z['audio_valid'] & z['vision_valid'];count['joint_usable']+=int(joint.sum())
  count['usable_no_vision']+=int((expected & ~z['vision_valid']).sum())
  for ca in [.1,.2,.3]:
   for wa in [.6,.8,1.]:sensitivity[(ca,wa)]+=int(c.mean()<ca or low.mean()>.5 or wer>wa or m['alignment_error'] is not None)
  for mod in ['audio','vision']:
   y,observed,w=pool(z['word_bounds'],z[mod+'_bounds'],z[mod+'_frames'],z[mod+'_frame_valid'])
   np.testing.assert_allclose(y,z[mod],rtol=1e-5,atol=1e-5)
   np.testing.assert_array_equal(observed,z[mod+'_valid'])
   np.testing.assert_allclose(w,z[mod+'_weights'],rtol=1e-5,atol=1e-7)
  source=PROJECT/m['source_path'];pts=[]
  with av.open(str(source)) as container:
   stream=container.streams.video[0];origin=float((stream.start_time or 0)*stream.time_base);rate=float(stream.average_rate)
   stream_end=float(stream.duration*stream.time_base) if stream.duration else 0.
   for f in container.decode(video=0):pts.append(float(f.pts*f.time_base)-origin)
  np.testing.assert_allclose(pts,z['all_video_pts'],rtol=0,atol=1e-9)
  np.testing.assert_allclose(np.asarray(pts)[z['video_frame_indices']],z['vision_pts'],rtol=0,atol=1e-9)
  count['decoded_video_frames']+=len(pts)
  assert int(saved_samples[sid]['decoded_video_frames'])==len(pts)
  start,signal,coverage=decode_audio(source,origin)
  duration=max(stream_end,pts[-1]+1/rate,start+len(signal)/16000)
  assert abs(duration-m['duration'])<1e-7 and abs(len(signal)/16000-m['audio_duration'])<1e-7
  durations.append(duration)
  samples=np.rint((z['audio_bounds']-start)*16000).astype(int);samples=np.clip(samples,0,len(signal))
  rms=np.array([np.sqrt(np.mean(signal[a:b].astype(np.float64)**2)) for a,b in samples])
  maxdiff['raw_audio_rms']=max(maxdiff['raw_audio_rms'],float(abs(rms-z['audio_rms']).max()))
  np.testing.assert_allclose(rms,z['audio_rms'],rtol=2e-5,atol=2e-7)
  np.testing.assert_array_equal(rms<threshold,z['audio_silence'])
  actual_valid=np.array([b>a and coverage[a:b].all() for a,b in samples]) & np.isfinite(z['audio_frames']).all(axis=1)
  np.testing.assert_array_equal(actual_valid,z['audio_frame_valid'])
  for mod in ['audio','vision']:
   f=z[mod+'_frames'][z[mod+'_frame_valid']]
   if len(f) and not (f.std(axis=0)>1e-6).any():constant.append({'sample_id':sid,'modality':mod,'records':len(f),'audio_waveform_max_abs':float(abs(signal).max())})
  if not risk and z['vision_frame_valid'].mean()>=.8:candidate_typical.append((abs(duration-10),sid))
  for width in [.2,.5]:
   left=np.arange(0,duration,width);bins=np.column_stack([left,np.minimum(left+width,duration)])
   rec={'sample_id':sid,'window_seconds':width,'windows':len(bins),'timing_usable_words':int(expected.sum())}
   crossed=(intersect(z['word_bounds'][expected],bins)>1e-8).sum(axis=1)>=2
   rec['words_crossing_boundary']=int(crossed.sum());rec['word_boundary_crossing_ratio']=float(crossed.mean()) if len(crossed) else None
   with np.load(directory/f'windows_{width:.1f}s.npz') as out:
    np.testing.assert_allclose(out['bounds'],bins,rtol=0,atol=1e-9)
    for mod in ['audio','vision']:
     native=z[mod+'_frames'];valid=z[mod+'_frame_valid'];bounds=z[mod+'_bounds']
     pooled,mask,w=pool(bins,bounds,native,valid)
     np.testing.assert_allclose(pooled,out[mod],rtol=1e-5,atol=1e-5)
     np.testing.assert_array_equal(mask,out[mod+'_valid'])
     rebuilt,support,_=pool(bounds,bins,pooled,mask);keep=valid&support
     error=None
     if keep.any():
      sd=native[keep].std(axis=0);use=sd>1e-6
      error=float(np.mean(((rebuilt[keep][:,use]-native[keep][:,use])/sd[use])**2)) if use.any() else 0.
     rec[mod+'_normalized_reconstruction_mse']=error
   saved=saved_windows[(sid,width)]
   for k,v in rec.items():
    if k=='sample_id':continue
    if v is None:assert saved[k]==''
    else:
     delta=abs(v-float(saved[k]));maxdiff[k]=max(maxdiff[k],delta)
     assert np.isclose(v,float(saved[k]),rtol=1e-6,atol=1e-8),(sid,width,k,v,saved[k])
   recomputed_windows.append(rec)
  # 验证正文100行表的数值，直接比较生成前的语义数值。
  audit_rows.append({'sample_id':sid,'D':duration,'Da':m['audio_duration'],'L':n,
   'T':int(z['text_valid'].sum()),'A':int(z['audio_valid'].sum()),'V':int(z['vision_valid'].sum()),'U':int(expected.sum()),'J':int(joint.sum()),'R':int(risk)})
 if count['samples']%20==0:print('verified',count['samples'],'samples',flush=True)
assert count['samples']==100 and len(saved_windows)==len(recomputed_windows)==200
for line in (HERE/'all_samples_table.tex').read_text().splitlines():
 if not line.startswith('\\nolinkurl{'):continue
 sid=re.match(r'\\nolinkurl\{([^}]+)\}',line).group(1);r=next(x for x in audit_rows if x['sample_id']==sid)
 values=[x.strip().removesuffix('\\\\') for x in line.split('&')[1:]]
 assert values==[f"{r['D']:.3f}",f"{r['Da']:.3f}",str(r['L']),str(r['V']),str(r['U']),str(r['J'])],(sid,values)
statistics={};paired={}
for key in ['audio_normalized_reconstruction_mse','vision_normalized_reconstruction_mse','word_boundary_crossing_ratio']:
 statistics[key]={}
 for width in [.2,.5]:
  val=np.asarray([r[key] for r in recomputed_windows if r['window_seconds']==width and r[key] is not None]);q1,q2,q3=np.quantile(val,[.25,.5,.75]);iqr=q3-q1
  statistics[key][str(width)]={'n':len(val),'mean':float(val.mean()),'median':float(q2),'q1':float(q1),'q3':float(q3),'zero_count':int((val==0).sum()),'outliers':int(((val<q1-1.5*iqr)|(val>q3+1.5*iqr)).sum())}
 pairs=[]
 for sid in labels:
  a=next(r[key] for r in recomputed_windows if r['sample_id']==sid and r['window_seconds']==.2)
  b=next(r[key] for r in recomputed_windows if r['sample_id']==sid and r['window_seconds']==.5)
  if a is not None and b is not None:pairs.append(a-b)
 delta=np.array(pairs);paired[key]={'short_less':int((delta< -1e-8).sum()),'equal_within_1e8':int((abs(delta)<=1e-8).sum()),'short_greater':int((delta>1e-8).sum())}
conf=np.asarray(confidence);wer=np.asarray(diagnostics)
# 独立核对正文解读中新增的实例数值与旧图的实例清单。
examples={}
manifest=json.loads((HERE/'explanatory_manifest.json').read_text())
for item in manifest['observation_examples']:
 directory=DATA/'samples'/item['sample_id'];meta=json.loads((directory/'metadata.json').read_text())
 with np.load(directory/'features.npz') as z:
  k=next(i for i,w in enumerate(meta['words']) if w['original']==item['word'])
  mask=intersect(z['word_bounds'][k:k+1],z['vision_bounds'])[0]>0
  np.testing.assert_allclose(z['word_bounds'][k],item['word_interval'])
  np.testing.assert_allclose(z['vision_pts'][mask],item['frame_pts'])
  assert int(z['vision_frame_valid'][mask].sum())==item['valid_frames']
  assert bool(z['vision_valid'][k])==item['word_observed']
directory=DATA/'samples'/'-egA8-b7-3M$_$1';meta=json.loads((directory/'metadata.json').read_text())
with np.load(directory/'features.npz') as z:
 k=next(i for i,w in enumerate(meta['words']) if w['original']=='successful')
 vis=np.flatnonzero(z['vision_weights'][k]>0);aud=np.flatnonzero(z['audio_weights'][k]>0)
 weights=z['vision_weights'][k,vis];overlap=intersect(z['word_bounds'][k:k+1],z['vision_bounds'])[0,vis]
 assert (aud[0],aud[-1],len(aud))==(183,240,58)
 np.testing.assert_allclose(z['vision_pts'][vis],[1.8,2.,2.2,2.4])
 assert [f'{x*100:.2f}' for x in weights]==['10.27','35.71','35.71','18.30']
 center=z['audio_bounds'].mean(axis=1);loud=z['audio_frames'][:,meta['audio_feature_names'].index('Loudness_sma3')]
 selected=np.flatnonzero((center>=1.8)&(center<=3.6)&z['audio_frame_valid']);peak=selected[np.argmax(loud[selected])]
 assert np.isclose(center[peak],2.11)
 face={str(t):{name:float(z['vision_frames'][np.argmin(abs(z['vision_pts']-t)),meta['vision_feature_names'].index(name)]) for name in ['jawOpen','mouthSmileLeft']} for t in [2.,2.4,2.8,3.,3.4]}
 examples['successful']={'interval':z['word_bounds'][k].tolist(),'audio_indices':[int(aud[0]),int(aud[-1])],'audio_record_count':len(aud),
  'vision_pts':z['vision_pts'][vis].tolist(),'vision_overlaps':overlap.tolist(),'vision_weights':weights.tolist(),'overlap_past_2.4_seconds':float(z['word_bounds'][k,1]-2.4)}
 examples['typical_curve']={'loudness_peak_time':float(center[peak]),'loudness_peak_value':float(loud[peak]),'face_coefficients':face}
metrics=json.loads((DATA/'metrics.json').read_text())
assert np.isclose(metrics['mean_word_confidence'],conf.mean()) and np.isclose(metrics['median_word_confidence'],np.median(conf))
assert np.isclose(metrics['mean_diagnostic_wer'],wer.mean())
for row in metrics['window_comparison']:
 for key in statistics:assert np.isclose(row[key],statistics[key][str(row['window_seconds'])]['mean'])
report={'status':'passed','scope':'100条源视频解码、77,261条RMS重算、1932词CTC分数、100条WER、质量掩码、200条窗口对照与正文全量表核验；不重跑预训练模型，不提供人工词界真值',
 'counts':dict(count),'unassigned_subwords':dict(unassigned),'unassigned_subword_total':sum(unassigned.values()),
 'confidence':{'mean':float(conf.mean()),'median':float(np.median(conf)),'below_0.1':int((conf<.1).sum()),'at_least_0.95':int((conf>=.95).sum())},
 'wer':{'mean':float(wer.mean()),'above_0.8':int((wer>.8).sum()),'above_1':int((wer>1).sum()),'max':float(wer.max())},
 'sensitivity':[{'mean_threshold':a,'wer_threshold':b,'samples':v} for (a,b),v in sorted(sensitivity.items())],
 'duration_range':[min(durations),max(durations)],'selected_typical':min(candidate_typical)[1],
 'constant_features':constant,'window_statistics':statistics,'paired_window_comparison':paired,'examples':examples,'maximum_absolute_differences':dict(maxdiff)}
(HERE/'data_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with (HERE/'audited_window_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=recomputed_windows[0].keys());writer.writeheader();writer.writerows(recomputed_windows)
print(json.dumps(report,ensure_ascii=False,indent=2))
