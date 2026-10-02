"""整理用户提供的21条人工审核记录，与既有自动筛选结果核对，不改写特征或掩码。"""
from pathlib import Path
from collections import Counter
import csv
import json
import math

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'results/samples'
# 原始说明逐字保存；首条说明以“因此”结束，正文不补写未提供的后续结论。
SOURCE = [
    ('-HwX2H8Z4hY$*$9',22,.7237673401832581,'音频卡顿、局部对应','语音识别正确，原视频音频存在卡顿，与原始文本不符合，但是大体词语正确因此'),
    ('-NFrJFQijFE$*$1',16,.006235090550035238,'无对应朗读','无朗读，音乐'),
    ('-NFrJFQijFE$*$2',18,.005170820746570826,'分段内容不对应','前半段无朗读，后半段音轨无对应'),
    ('-hnBHBN8p5A$*$6',13,.07967864722013474,'音轨与文本不对应','音轨无对应'),
    ('-hnBHBN8p5A$*$7',13,.003141609486192465,'音轨与文本不对应','音轨无对应'),
    ('-iRBcNs9oI8$*$3',8,.12826672196388245,'音轨与文本不对应','音轨无对应'),
    ('-iRBcNs9oI8$*$6',5,.2011333405971527,'音轨与文本不对应','音轨无对应'),
    ('-iRBcNs9oI8$*$7',9,.004080040846019983,'音轨与文本不对应','音轨无对应'),
    ('-iRBcNs9oI8$*$8',21,.06751171499490738,'音轨与文本不对应','音轨无对应'),
    ('-iRBcNs9oI8$*$9',5,.0003849636996164918,'音轨与文本不对应','音轨无对应'),
    ('-mJ2ud6oKI8$*$1',13,.01248920802026987,'无对应朗读','无朗读'),
    ('-mJ2ud6oKI8$*$2',11,.009920411743223667,'无对应朗读','无朗读'),
    ('-mJ2ud6oKI8$*$6',5,.015795908868312836,'音轨与文本不对应','音轨无对应'),
    ('-mJ2ud6oKI8$*$8',11,.005572875961661339,'音轨与文本不对应','音轨无对应'),
    ('-mJ2ud6oKI8$*$9',22,.05434906482696533,'音轨与文本不对应','音轨无对应'),
    ('-ri04Z7vwnc$*$0',8,.014759980142116547,'无对应朗读','无朗读'),
    ('-ri04Z7vwnc$*$2',16,.0025197379291057587,'无对应朗读','无朗读'),
    ('-ri04Z7vwnc$*$5',8,.0049331169575452805,'音轨与文本不对应','音轨无对应'),
    ('-yRb-Jum7EQ$*$1',49,.0748576894402504,'非英语语音','并非英语'),
    ('-yRb-Jum7EQ$*$5',25,.0630141869187355,'非英语语音','并非英语'),
    ('-yRb-Jum7EQ$\\_$6',19,.03225075453519821,'非英语语音','并非英语'),
]
CATEGORIES = [
    ('音轨与文本不对应','实际语音内容与给定转写不对应。'),
    ('无对应朗读','未听到对应朗读，其中1条为音乐。'),
    ('非英语语音','实际语音并非英语。'),
    ('分段内容不对应','前半段无朗读，后半段语音与转写不对应。'),
    ('音频卡顿、局部对应','音频存在卡顿，部分词语与文本一致，但未形成完整对应。'),
]


def main():
    rows=[]
    for supplied_id,words,confidence,category,note in SOURCE:
        sid=supplied_id.replace('$*$','$_$').replace('$\\_$','$_$')
        q=json.loads((DATA/sid/'quality.json').read_text())
        assert q['sample_id']==sid and q['requires_transcript_review']
        assert q['words']==words and math.isclose(q['mean_confidence'],confidence,rel_tol=1e-12)
        assert q['usable_word_times']==0
        rows.append({'sample_id':sid,'supplied_sample_id':supplied_id,
                     'requires_transcript_review':True,'reasons':';'.join(q['reasons']),
                     'mean_confidence':q['mean_confidence'],'low_confidence_fraction':q['low_confidence_fraction'],
                     'diagnostic_wer':q['diagnostic_wer'],'usable_word_times':0,'words':words,
                     'raw_text':q['raw_text'],'diagnostic_asr':q['diagnostic_asr'],
                     'manual_review_status':'已审核','manual_category':category,'original_review_note':note,
                     'review_source':'用户提供的人工审核记录','word_boundaries_annotated':False})
    flagged={p.parent.name for p in DATA.glob('*/quality.json') if json.loads(p.read_text())['requires_transcript_review']}
    assert len(rows)==len({r['sample_id'] for r in rows})==21
    assert flagged=={r['sample_id'] for r in rows}
    counts=Counter(r['manual_category'] for r in rows)
    words=Counter()
    for r in rows: words[r['manual_category']]+=r['words']
    assert [counts[k] for k,_ in CATEGORIES]==[11,5,3,1,1]
    assert [words[k] for k,_ in CATEGORIES]==[120,64,93,18,22]
    with (HERE/'人工审核_21条样本.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary={'status':'passed','source':'用户提供的21条人工审核记录；自动指标由既有quality.json匹配核对',
             'id_normalization':'将粘贴文本中的$*$及转义下划线还原为项目样本编号分隔符$_$；已按21条风险样本集合核对',
             'reviewed_samples':21,'reviewed_original_words':sum(words.values()),
             'categories':[dict(category=k,samples=counts[k],words=words[k],description=d) for k,d in CATEGORIES],
             'scope':'只审核自动筛出的21条；未提供其他79条的人工审核或逐词时间边界标注',
             'quality_masks_changed':False,'records':rows}
    (HERE/'manual_review_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=[r'''% 用户提供的人工审核结果；由build_manual_review.py汇总。
\begin{table}[H]
    \centering\small
    \caption{21条复核样本的人工审核结果}
    \label{tab:p1_manual_review}
    \setlength{\tabcolsep}{5pt}
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{@{}lrr>{\raggedright\arraybackslash}p{5.0cm}@{}}
        \toprule
        审核发现 & 样本数 & 原文词数 & 主要表现\\ \midrule''']
    for k,d in CATEGORIES:lines.append(f'        {k} & {counts[k]} & {words[k]} & {d}'+r'\\')
    lines += [r'        \midrule',r'        合计 & 21 & 317 & \\',r'        \bottomrule',r'    \end{tabular}',r'\end{table}']
    (HERE/'manual_review_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
