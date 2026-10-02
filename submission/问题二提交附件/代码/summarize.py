# 本程序及代码是在人工智能工具辅助下完成的。

# 人工智能工具名称：GPT-6 Astra

# 版本/型号：gpt-6-astra

# 开发机构/公司：OpenAI

# 版本颁布日期：2026年9月3日

"""从实验记录复算分组统计，并分别导出验证与缺失分析图。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--records', type=Path, default=Path(__file__).resolve().parents[1] / '实验记录/主实验')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(args.records / 'test_metrics.csv')
    valid = pd.read_csv(args.records / 'validation_prediction_review.csv')
    ens = test[test.seed == 'ensemble']
    primary = ens[(ens.pattern == 'single') & (ens['sync'] == True) & (ens.rate > 0)]
    # 先平均同一条件内的随机重复，再对不同条件等权平均。
    columns = ['accuracy', 'macro_f1', 'mae', 'pearson']
    conditions = primary.groupby(['model','subset','rate','position'])[columns].mean().reset_index()
    conditions.to_csv(args.output / '逐条件指标.csv', index=False)
    conditions.groupby('model')[columns].mean().to_csv(args.output / '主缺失条件等权均值.csv')
    ens[ens.name == 'clean'][['model'] + columns].to_csv(args.output / '完整输入模型比较.csv', index=False)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(args.output / (name + '.pdf'), bbox_inches='tight', pad_inches=.03)
        plt.close(fig)
    clean = valid[valid.scenario == 'clean']
    cm = confusion_matrix(clean.true_class, clean.predicted_class, labels=[0,1,2])
    pd.DataFrame(cm, index=['negative','neutral','positive'], columns=['negative','neutral','positive']).to_csv(args.output / '验证集混淆矩阵.csv')
    fig, ax = plt.subplots(figsize=(4.5,3.6))
    ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=['Negative','Neutral','Positive'],
           yticklabels=['Negative','Neutral','Positive'], xlabel='Predicted class', ylabel='True class')
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>.55*cm.max() else 'black')
    save(fig, '验证集极性混淆矩阵')
    fig, ax = plt.subplots(figsize=(4.5,3.6))
    ax.scatter(clean.true_intensity, clean.predicted_intensity, s=10, alpha=.4)
    ax.plot([-3,3],[-3,3], 'k--', lw=1)
    ax.set(xlabel='True intensity', ylabel='Predicted intensity', xlim=(-3,3), ylim=(-3,3))
    save(fig, '验证集强度预测')
    chosen = conditions[(conditions.model == 'gated_aug') & (conditions.position == 'random')]
    for metric in ['macro_f1','mae']:
        fig, ax = plt.subplots(figsize=(4.8,3.6))
        for subset in ['T','A','V','TA','TV','AV','TAV']:
            d = chosen[chosen.subset == subset].sort_values('rate')
            ax.plot(d.rate, d[metric], marker='o', ms=3, label=subset)
        ax.set(xlabel='Missing fraction', ylabel={'macro_f1':'Macro-F1','mae':'MAE'}[metric], xticks=[.1,.3,.5,.7])
        ax.legend(ncol=4, fontsize=8)
        ax.grid(alpha=.2)
        save(fig, '缺失类型与比例_' + metric)


if __name__ == '__main__':
    main()
