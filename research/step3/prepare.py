"""只用训练/验证确定问题三协议；专项仅进行格式核查，不评价预测。"""
from common import *
from engine import load_view
from data import load_attachment2,adapt,TextEncoder,make_view
import shutil
import numpy as np

def main():
    for d in ['results','checkpoints','cache','results/figures','results/mapping','results/cards']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    shutil.copy2(P2/'checkpoints/normalization.npz',ROOT/'checkpoints/normalization.npz')
    # 交付包不含大型可重建缓存；需要时在step3内重建，绝不改写step2。
    missing=[s for s in ['train','valid'] if not (P2/f'cache/{s}_0.npz').exists() and not (ROOT/f'cache/{s}_clean.npz').exists()]
    if missing:
        data=load_attachment2();encoder=TextEncoder();stats=dict(np.load(ROOT/'checkpoints/normalization.npz'))
        for split in missing:
            b=adapt(data[split]);v=make_view(b,encoder,stats,dict(subset='',rate=0.,position='random'),0)
            np.savez(ROOT/f'cache/{split}_clean.npz',**v)
            np.savez(ROOT/f'cache/{split}_labels.npz',y=data[split]['regression_labels'].astype(np.float32),c=data[split]['classification_labels'].astype(np.int64))
    train=load_view('train')
    reference={name:np.median(train[name][train['o'][:,j]].astype(np.float32),axis=0) for j,name in enumerate(['text','audio','vision'])}
    np.savez(ROOT/'checkpoints/training_reference.npz',**reference)
    protocol=dict(seeds=[17,29,43],initialization='step2/gated_aug checkpoints, each trained on attachment2 train only',
        epochs=20,patience=5,lr=.0003,weight_decay=.001,batch_size=64,dropout=.2,hidden=64,
        coalition_probability=.2,coalition_sampling='uniform proper subsets 0..6; otherwise complete inputs',
        loss='train-frequency weighted CE + SmoothL1',checkpoint_selection='clean validation MAE + 1 - MacroF1',
        prediction_candidates=['step2_fixed_clean','step2_fixed_aug','step2_mask_aug','step2_gated_aug','step3_coalition_aug'],
        explanation_target='fixed original predicted class probability (ensemble mean), plus separate continuous intensity Shapley',
        primary_baseline='modality absent: features zero, observation mask false, recomputed availability statistics',
        sensitivity_baseline='training-only feature medians with original observation masks; a different representation-replacement game',
        ig=dict(nodes=32,reference='standardized zero, P/O/quality fixed; representation-level ranking only',convergence_nodes=64,convergence_samples=48),
        local=dict(fraction=.2,ranking='largest signed sum of IG over contiguous equal-length token windows',
            validation='delete raw text tokens before frozen encoding; audio/vision rows zero with observation false',
            random_controls=5,keep_control='retain selected window within the same modality, other modalities unchanged',
            special_confirmation='evaluate every same-length sliding window; report IG-selected and strongest observed deletion separately',
            probability_drop_warning='negative deletion effect is counter-evidence, never relabeled supporting'),
        faithfulness_validation_samples='all 728 validation samples; primary main-modality and all-modality summaries',
        stability=dict(seeds='all three predictors, target fixed to ensemble original class',noise_sd=.02,samples=96),
        weak_absolute_phi_threshold=.001,main_share_threshold=.45,
        timing=dict(method='CTC forced alignment of original full transcript; fast tokenizer offsets to token slots',
            quality=dict(mean_confidence=.2,low_word_confidence=.1,max_low_fraction=.5,max_diagnostic_wer=.8),
            source='step1 fixed wav2vec2 assets; no proportional t/50 video timestamps',
            feature_correspondence='task-provided aligned slot correspondence + within-word repeat audit; physical times estimated, not official ground truth'),
        stage_boundary='all model/baseline/window rules locked before special predictions',
        input_sha256=json.loads((P2/'protocol.json').read_text())['source_sha256'],
        code_reuse={name:sha256(PROJECT/'step1'/name) for name in ['alignment.py']},
        model_architecture_sha256=sha256(ROOT/'model.py'))
    save_json(ROOT/'protocol.json',protocol)
    save_json(ROOT/'numerical_protocol.json',dict(method='Gauss-Legendre IG',initial_nodes=32,refinement_nodes=[64,128,256],
        absolute_completeness_tolerance=.001,criterion='IG sum equals fixed-mask representation endpoint score difference; no labels used',
        reason='numerical integration refinement, not prediction-model or explanation-budget tuning'))
    d=load_special()
    save_json(ROOT/'results/interface_audit.json',dict(n=20,ids=d['id'],version='aligned',
        fields={k:dict(shape=list(d[k].shape),dtype=str(d[k].dtype)) for k in ['text_bert','audio','vision']},
        numpy_compatibility='NumPy2 numpy._core -> installed NumPy1 numpy.core via custom Unpickler only',
        no_environment_or_input_modification=True,
        input_sha256={p.name:sha256(p) for p in sorted(SPECIAL.glob('*.pkl'))}))
    save_json(ROOT/'reuse_manifest.json',dict(step2_model_source_sha256=sha256(P2/'model.py'),
        step3_model_copy_sha256=sha256(ROOT/'model.py'),step1_alignment_source_sha256=sha256(PROJECT/'step1/alignment.py')))
    print('Problem 3 protocol and train-only references prepared.',flush=True)

if __name__=='__main__':main()
