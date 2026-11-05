# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (RidgeCV, LassoCV, ElasticNetCV,
                                  Ridge, Lasso, ElasticNet)
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing, make_regression


def load_data():
    # 1) 优先 Hitters
    try:
        df = pd.read_csv("Hitters.csv").dropna()
        X = pd.get_dummies(df.drop(["Salary"], axis=1), drop_first=True)
        y = df["Salary"].astype(float)
        X = X.astype(float)
        print("[数据] 使用 Hitters，样本=%d, 特征=%d" % X.shape)
        return X, y, "Hitters"
    except Exception as e:
        print("[数据] Hitters 不可用(%s)，尝试 California Housing。" % type(e).__name__)

    # 2) California Housing
    try:
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = pd.Series(housing.target, name="MedHouseVal")
        print("[数据] 使用 California Housing，样本=%d, 特征=%d" % X.shape)
        return X, y, "CaliforniaHousing"
    except Exception as e:
        print("[数据] California Housing 不可用(%s)，回退合成数据。" % type(e).__name__)

    # 3) 合成数据兜底
    X, y = make_regression(
        n_samples=200,
        n_features=15,
        n_informative=6,
        n_redundant=4,
        noise=15,
        random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(15)])
    y = pd.Series(y, name="y")
    print("[数据] 使用合成数据，样本=%d, 特征=%d" % X.shape)
    return X, y, "Synthetic"


X, y, dataset_name = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xte = scaler.transform(X_test)
feature_names = list(X.columns)
print("[切分] 训练=%d, 测试=%d" % (Xtr.shape[0], Xte.shape[0]))

alphas = np.logspace(-3, 3, 100)   # 对数网格 lambda

ridge_cv = RidgeCV(alphas=alphas, cv=10).fit(Xtr, y_train)
lasso_cv = LassoCV(
    alphas=alphas, cv=10, max_iter=50000, random_state=42
).fit(Xtr, y_train)
enet_cv = ElasticNetCV(
    alphas=alphas, l1_ratio=0.5, cv=10, max_iter=50000, random_state=42
).fit(Xtr, y_train)

models = {"Ridge": ridge_cv, "Lasso": lasso_cv, "ElasticNet": enet_cv}


def coef_path(model_cls, Xtr, ytr, alphas, title, fname, chosen_alpha=None, **kw):
    paths = np.zeros((len(alphas), Xtr.shape[1]))
    for i, a in enumerate(alphas):
        m = model_cls(alpha=a, **kw).fit(Xtr, ytr)
        paths[i] = m.coef_

    plt.figure(figsize=(9, 5.5))
    plt.plot(np.log10(alphas), paths)
    plt.axhline(0, color="gray", lw=0.8)

    if chosen_alpha is not None and chosen_alpha > 0:
        plt.axvline(
            np.log10(chosen_alpha),
            color="red",
            ls="--",
            lw=1,
            label="CV 选中的 alpha",
        )

    plt.xlabel("log10(alpha)")
    plt.ylabel("系数 beta_j")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print("[图] 已保存", fname)


coef_path(
    Ridge,
    Xtr,
    y_train,
    alphas,
    "Ridge 系数路径（同步收缩，永不触零）",
    "coef_path_ridge.png",
    chosen_alpha=float(ridge_cv.alpha_),
)
coef_path(
    Lasso,
    Xtr,
    y_train,
    alphas,
    "Lasso 系数路径（依次撞线归零）",
    "coef_path_lasso.png",
    chosen_alpha=float(lasso_cv.alpha_),
    max_iter=50000,
)
coef_path(
    ElasticNet,
    Xtr,
    y_train,
    alphas,
    "Elastic Net 系数路径（l1_ratio=0.5）",
    "coef_path_enet.png",
    chosen_alpha=float(enet_cv.alpha_),
    l1_ratio=0.5,
    max_iter=50000,
)

# ===== 测试集评估 =====
rows = []
for name, m in models.items():
    pred = m.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    nz = int(np.sum(np.abs(m.coef_) > 1e-8))
    rows.append(
        {
            "模型": name,
            "CV选中的alpha": float(m.alpha_),
            "测试RMSE": rmse,
            "非零变量数": nz,
            "总变量数": len(feature_names),
        }
    )

res = pd.DataFrame(rows)
print("\n==== 测试集结果 ====")
print(res.to_string(index=False))
res.to_csv("作业4_结果汇总.csv", index=False, encoding="utf-8-sig")


# ===== 1-SE 法则（以 Lasso 为例）=====
def one_se_rule(Xtr, ytr, alphas, cv=10, seed=42, **model_kw):
    """返回 (alpha_min, alpha_1se, mse_min, se_at_min)。"""
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    mse_folds = np.zeros((cv, len(alphas)))

    for k, (tr, va) in enumerate(kf.split(Xtr)):
        for i, a in enumerate(alphas):
            m = Lasso(alpha=a, **model_kw).fit(Xtr[tr], np.asarray(ytr)[tr])
            mse_folds[k, i] = mean_squared_error(
                np.asarray(ytr)[va], m.predict(Xtr[va])
            )

    mse_mean = mse_folds.mean(axis=0)
    se = mse_folds.std(axis=0, ddof=1) / np.sqrt(cv)

    i_min = int(np.argmin(mse_mean))
    thresh = mse_mean[i_min] + se[i_min]

    ok = np.where(mse_mean <= thresh)[0]
    if len(ok) > 0:
        i_1se = int(ok[np.argmax(alphas[ok])])   # 满足阈值中取最大 alpha（最稀疏）
    else:
        i_1se = i_min

    return alphas[i_min], alphas[i_1se], mse_mean[i_min], se[i_min]


a_min, a_1se, mse_min, se_min = one_se_rule(
    Xtr, np.asarray(y_train), alphas, max_iter=50000
)

l_min = Lasso(alpha=a_min, max_iter=50000).fit(Xtr, y_train)
l_1se = Lasso(alpha=a_1se, max_iter=50000).fit(Xtr, y_train)

nz_min = int(np.sum(np.abs(l_min.coef_) > 1e-8))
nz_1se = int(np.sum(np.abs(l_1se.coef_) > 1e-8))
rmse_min = float(np.sqrt(mean_squared_error(y_test, l_min.predict(Xte))))
rmse_1se = float(np.sqrt(mean_squared_error(y_test, l_1se.predict(Xte))))

print("\n1-SE 法则(以 Lasso 为例)")
print("最小CV-MSE : alpha=%.4g, 非零变量=%d, 测试RMSE=%.4f" % (a_min, nz_min, rmse_min))
print("1-SE 法则  : alpha=%.4g, 非零变量=%d, 测试RMSE=%.4f" % (a_1se, nz_1se, rmse_1se))
print(
    "结论：从最小MSE模型切到1-SE模型，非零变量 %d -> %d（更稀疏），"
    "RMSE 由 %.4f -> %.4f。若差异可接受，应选更稀疏的 1-SE 模型。"
    % (nz_min, nz_1se, rmse_min, rmse_1se)
)

print("\n完成。图片：coef_path_ridge.png / coef_path_lasso.png / coef_path_enet.png")
print("结果表：作业4_结果汇总.csv")