# -*- coding: utf-8 -*-
import re, os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (r2_score, mean_squared_error, roc_auc_score,
                             roc_curve, confusion_matrix, classification_report, accuracy_score)

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('output', exist_ok=True)

CSV = Path("E:/typora homework/netflix_titles.csv")
assert CSV.exists(), f"文件不存在：{CSV}"

#1. 读取与清洗
df = pd.read_csv(CSV).dropna(how='all').drop_duplicates(subset=['show_id'])

def parse_dur(d):
    if pd.isna(d):
        return np.nan, np.nan
    d = str(d)
    m = re.match(r'(\d+)\s*min', d)
    if m:
        return int(m.group(1)), np.nan
    s = re.match(r'(\d+)\s*Season', d)
    if s:
        return np.nan, int(s.group(1))
    return np.nan, np.nan

dd = df['duration'].apply(parse_dur)
df['dur_min'] = [x[0] for x in dd]
df['n_seasons'] = [x[1] for x in dd]
df['added_month'] = pd.to_datetime(df['date_added'], errors='coerce').dt.month
df['is_movie'] = (df['type'] == 'Movie').astype(int)

print(f"样本量：{len(df)}  列数：{df.shape[1]}")

#2. 特征工程
def base_feats(fr):
    f = pd.DataFrame(index=fr.index)
    f['dur_min'] = fr['dur_min'].fillna(fr['dur_min'].median())
    f['n_seasons'] = fr['n_seasons'].fillna(0)
    f['is_tvshow_flag'] = (fr['type'] == 'TV Show').astype(int)
    for col, nm in [('director', 'has_director'), ('cast', 'has_cast'),
                    ('country', 'has_country'), ('date_added', 'has_date')]:
        f[nm] = fr[col].notna().astype(int)
    f['n_cast'] = fr['cast'].fillna('').apply(
        lambda x: len([p for p in str(x).split(',') if p.strip()]))
    f['added_month'] = fr['added_month'].fillna(0)
    return f

def mk_name(prefix, t):
    return prefix + '_' + t.replace(' ', '_').replace('&', '').replace('/', '_').replace('.', '')

def oh_from_multi(fr, src, top_list, prefix, primary_only=False):
    cols = [mk_name(prefix, t) for t in top_list]
    oh = pd.DataFrame(0, index=fr.index, columns=cols)
    for idx, x in fr[src].items():
        if pd.isna(x):
            continue
        parts = [p.strip() for p in str(x).split(',') if p.strip()]
        if primary_only and parts:
            parts = [parts[0]]
        for p in parts:
            c = mk_name(prefix, p)
            if c in oh.columns:
                oh.loc[idx, c] = 1
    return oh

def top_k(fr, src, k):
    vals = []
    for x in fr[src].dropna():
        vals += [p.strip() for p in str(x).split(',') if p.strip()]
    return pd.Series(vals).value_counts().head(k).index.tolist()

TOPS = {
    'listed_in': top_k(df, 'listed_in', 12),
    'country':   top_k(df, 'country', 10),
    'director':  top_k(df, 'director', 12),
    'cast':      top_k(df, 'cast', 12),
}

def assemble(fr):
    f = base_feats(fr)
    g  = oh_from_multi(fr, 'listed_in', TOPS['listed_in'], 'genre')
    ct = oh_from_multi(fr, 'country',   TOPS['country'],   'ctry', primary_only=True)
    di = oh_from_multi(fr, 'director',  TOPS['director'],  'dir',  primary_only=True)
    ca = oh_from_multi(fr, 'cast',      TOPS['cast'],      'actor', primary_only=True)
    return pd.concat([f, g, ct, di, ca], axis=1).astype(float)

reg_df = df.dropna(subset=['release_year']).copy()
clf_df = df.dropna(subset=['type']).copy()
X_reg = assemble(reg_df); y_reg = reg_df['release_year'].astype(float).values
X_clf = assemble(clf_df); y_clf = clf_df['is_movie'].values

#3A. 回归 Elastic Net
Xtr, Xte, ytr, yte = train_test_split(X_reg.values, y_reg, test_size=0.25, random_state=42)
gs_reg = GridSearchCV(
    Pipeline([('sc', StandardScaler()), ('en', ElasticNet(max_iter=20000))]),
    {'en__alpha': [0.01, 0.05, 0.1, 0.5, 1, 2, 5],
     'en__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
    cv=KFold(5, shuffle=True, random_state=42),
    scoring='neg_mean_squared_error', n_jobs=-1
).fit(Xtr, ytr)

best_reg = gs_reg.best_estimator_
pred = best_reg.predict(Xte)
print('回归 best:', gs_reg.best_params_)
print('  RMSE=%.3f  R2=%.3f  (target_std=%.2f)  非零系数=%d/%d' % (
    np.sqrt(mean_squared_error(yte, pred)), r2_score(yte, pred), np.std(y_reg),
    (best_reg.named_steps['en'].coef_ != 0).sum(), X_reg.shape[1]))

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(yte, pred, s=12, alpha=.5, edgecolors='none')
lo, hi = yte.min() - 2, yte.max() + 2
ax.plot([lo, hi], [lo, hi], 'r--')
ax.set_title('回归: 预测 vs 真实发行年份')
ax.set_xlabel('真实 release_year'); ax.set_ylabel('预测 release_year')
fig.savefig('output/fig_reg_scatter.png', dpi=150, bbox_inches='tight'); plt.close(fig)

#3B. 分类 Elastic Net Logistic（剔除定义性泄露特征）
Xc = X_clf.drop(columns=['is_tvshow_flag', 'dur_min', 'n_seasons'])
Xtr, Xte, ytr, yte = train_test_split(Xc.values, y_clf, test_size=0.25,
                                     random_state=42, stratify=y_clf)
gs_clf = GridSearchCV(
    Pipeline([('sc', StandardScaler()),
              ('lr', LogisticRegression(penalty='elasticnet', solver='saga', max_iter=8000))]),
    {'lr__C': [0.05, 0.1, 0.5, 1, 5], 'lr__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=-1
).fit(Xtr, ytr)

best = gs_clf.best_estimator_
proba = best.predict_proba(Xte)[:, 1]
predc = (proba >= 0.5).astype(int)
print('分类 best:', gs_clf.best_params_)
print('  AUC=%.4f  Acc=%.4f  非零系数=%d/%d  混淆=%s' % (
    roc_auc_score(yte, proba), accuracy_score(yte, predc),
    (best.named_steps['lr'].coef_[0] != 0).sum(), Xc.shape[1],
    confusion_matrix(yte, predc).tolist()))
print(classification_report(yte, predc, target_names=['TV Show', 'Movie']))

fpr, tpr, _ = roc_curve(yte, proba)
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(fpr, tpr, lw=2, label='ElasticNet')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend()
ax.set_title('ROC: Movie vs TV Show')
fig.savefig('output/fig_roc.png', dpi=150, bbox_inches='tight'); plt.close(fig)

#4. 稀疏系数图
def coef_plot(coef, cols, title, path):
    s = pd.Series(coef, index=cols)
    nz = s[s != 0]
    keep = nz.reindex(nz.abs().sort_values().tail(20).index).sort_values() if len(nz) > 20 else nz.sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(keep)), keep.values,
            color=['#E15759' if v < 0 else '#4E79A7' for v in keep.values])
    ax.axvline(0, color='k', lw=.8)
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep.index, fontsize=9)
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)

coef_plot(best_reg.named_steps['en'].coef_, X_reg.columns, '回归稀疏系数 Top20', 'output/fig_coef_reg.png')
coef_plot(best.named_steps['lr'].coef_[0], Xc.columns, '分类稀疏系数 Top20', 'output/fig_coef_clf.png')

print('完成: 图表与指标已输出到 output/')