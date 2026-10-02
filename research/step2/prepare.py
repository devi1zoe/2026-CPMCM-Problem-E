"""接口审计、训练统计、冻结文本特征及预注册实验协议。"""
from common import *
from data import *
import importlib.metadata
import openpyxl
from transformers import AutoTokenizer


def main():
    torch.set_num_threads(4)
    raw = load_attachment2()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
    report = {'splits':{}, 'split_overlap':{}, 'version':'aligned_50', 'raw_text_used_for':'vocabulary audit only'}
    id_sets, vid_sets = {}, {}
    for split, d in raw.items():
        ids = d['id']; id_sets[split] = set(ids); vid_sets[split] = {x.split('$_$')[0] for x in ids}
        assert len(ids) == len(set(ids))
        retokenized = tokenizer(list(d['raw_text']), padding='max_length', truncation=True, max_length=50, return_tensors='np')['input_ids']
        exact = np.all(retokenized == d['text_bert'][:,0], axis=1)
        assert exact.all(), 'Supplied token IDs do not match the frozen encoder vocabulary'
        y,c = d['regression_labels'], d['classification_labels'].astype(int)
        assert np.array_equal(c, np.sign(y).astype(int)+1)
        assert np.isfinite(y).all() and np.max(np.abs(y)) <= 3
        b = adapt(d)
        report['splits'][split] = dict(n=len(ids), classes=np.bincount(c,minlength=3).tolist(), token_exact_matches=int(exact.sum()),
            uncertain_span_count=len(b['uncertain']), unk_tokens=int(np.sum(b['tokens']==100)),
            observed_fraction_within_valid=[float(b['o'][:,j].sum()/b['p'].sum()) for j in range(3)],
            nonfinite={k:int((~np.isfinite(d[k])).sum()) for k in ['text','audio','vision']})
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        overlap=dict(sample=len(id_sets[a]&id_sets[b]), video=len(vid_sets[a]&vid_sets[b]))
        assert overlap == dict(sample=0, video=0)
        report['split_overlap'][a+'_'+b] = overlap
    wb = openpyxl.load_workbook(DATA/'附件2-数据集特征文件/label.xlsx', read_only=True, data_only=True)
    rows = list(wb.active.values); header = rows[0]
    labels = [dict(zip(header,row)) for row in rows[1:] if row[0] is not None]
    indexed = {str(r['video_id'])+'$_$'+str(r['clip_id']):r for r in labels}
    assert len(indexed) == sum(len(d['id']) for d in raw.values())
    for split,d in raw.items():
        for id_,y in zip(d['id'],d['regression_labels']):
            r = indexed[id_]
            assert abs(float(r['label'])-float(y)) < 1e-5 and r['mode'] == split
    report['label_xlsx_rows_matched'] = len(indexed)
    wb.close()
    save_json(ROOT/'results/data_audit.json',report)
    protocol = dict(version=1,seeds=[17,29,43], training_views=6, clean_sampling_probability=.25,
        epochs=35,early_stopping_patience=7,batch_size=64,lr=.001,weight_decay=.001,hidden=64,dropout=.2,
        loss='weighted_cross_entropy + smooth_l1(beta=1)', class_weight='sqrt(N/(3*count)), normalized mean 1, train only',
        selection='0.5*clean(MAE+1-MacroF1) + 0.1*sum(five corrupted validation MAE+1-MacroF1); lower better',
        validation_profiles=VALID_PROFILES, validation_corruption_seed=20260924,
        models=['text_only','audio_only','vision_only','pooled_clean','fixed_clean','fixed_aug','mask_aug','gated_aug'],
        baseline_seeds=[17], main_models_seeds=[17,29,43],
        inference='select among four main model 3-seed ensembles using validation only; mean probabilities and intensities',
        test_protocol='lock selected architecture before any test model metrics; no test-driven tuning',
        rates=[.1,.3,.5,.7],subsets=['T','A','V','TA','TV','AV','TAV'], positions=['start','middle','end','random'],
        corruption_repeats=[0,1,2], rate_definition='deleted word-slot count / valid word-slot length; rounded half up, minimum 1',
        unk_policy='UNK is semantically unavailable, not evidence that artificial deletion occurred',
        text_corruption='delete token and attention before recomputing frozen context',
        time_unit='aligned token-slot fraction, no seconds inferred', clip_z=8,
        frozen_encoder=dict(repo='distilbert/distilbert-base-uncased',revision='12040accade4e8a0f71eabdb258fecc2e7e948be',
            weights_sha256=sha256(ENCODER/'model.safetensors')),
        source_sha256=sha256(DATA/'附件2-数据集特征文件/aligned_50.pkl'),
        packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','scikit-learn','pandas','matplotlib']})
    save_json(ROOT/'protocol.json',protocol)
    encoder = TextEncoder()
    train = adapt(raw['train']); clean_text = encoder.encode(train,train['o'])
    stats = fit_stats(train,clean_text)
    np.savez(ROOT/'checkpoints/normalization.npz',**stats)
    for split in ['train','valid']:
        b = train if split=='train' else adapt(raw[split])
        text = clean_text if split=='train' else encoder.encode(b,b['o'])
        profiles = [VALID_PROFILES[0]] + [dict(name=f'aug{j}',random_mix=True) for j in range(6)] if split=='train' else VALID_PROFILES
        for j,profile in enumerate(profiles):
            seed = (7000+j if split=='train' else protocol['validation_corruption_seed'])
            view = make_view(b,encoder,stats,profile,seed,clean_text=text)
            np.savez(ROOT/f'cache/{split}_{j}.npz',**view)
            print('prepared',split,j,profile['name'],flush=True)
        np.savez(ROOT/f'cache/{split}_labels.npz',y=raw[split]['regression_labels'].astype(np.float32),c=raw[split]['classification_labels'].astype(np.int64))
    print('Preparation complete; test labels not used for fitting or selection.',flush=True)


if __name__ == '__main__':
    main()
