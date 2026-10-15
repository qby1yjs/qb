# **支持向量机 (SVM) 数据分类实验报告 —— 德国信用数据**

> 算法主线：超平面 → 最大间隔 → 拉格朗日对偶 → KKT 稀疏性 → RBF 核
> 数据集：课件《04_SVM_Detail》第 04 节真实案例② —— 德国信用数据 (German Credit / Statlog)

## **一、算法原理回顾**

支持向量机的目标是找到一个能把两类样本分开、且**间隔最大**的超平面。

**1. 超平面与几何间隔**
线性判别函数为 f(x)=w·x+b，决策边界为 w·x+b=0。点到超平面的几何间隔为 γ = (w·x+b)/‖w‖。最大化最小几何间隔等价于在约束 yᵢ(w·xᵢ+b)≥1 下最小化 ½‖w‖²。

**2. 软间隔**
真实数据（如信用数据）线性不可分且存在噪声，引入松弛变量 ξᵢ 与惩罚系数 C：min ½‖w‖² + C·Σξᵢ，使"间隔大"与"少分错"折中。C 越大越不容忍误分，越易过拟合。

**3. 拉格朗日对偶**
把约束优化转成对偶问题，只需求解每个样本的乘子 αᵢ：max Σαᵢ − ½ΣΣ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)，约束 0≤αᵢ≤C、Σαᵢyᵢ=0。决策函数变为只依赖样本内积：f(x)=Σαᵢyᵢ(xᵢ·x)+b。

**4. KKT 条件与稀疏性**
由 KKT 互补条件 αᵢ·[yᵢ·f(xᵢ)−1+ξᵢ]=0 可知：只有落在间隔边界上或被误分的样本 αᵢ>0，即**支持向量**；绝大多数样本 αᵢ=0，对超平面没有贡献。因此分类器是**稀疏**的，支持向量占比也是衡量数据可分性/重叠程度的直观指标。

**5. RBF 核**
对线性不可分数据，用核函数把样本隐式映射到高维空间再求最大间隔。本实验用高斯核 K(xᵢ,xⱼ)=exp(−γ‖xᵢ−xⱼ‖²)，γ 越大边界越"弯"越易过拟合。

## **二、数据集说明**

表格

| 项目     | 内容                                                         |
| :------- | :----------------------------------------------------------- |
| 名称     | German Credit / Statlog German Credit                        |
| 出处     | UCI / Statlog 机器学习库（课件案例②的原始来源）              |
| 样本     | 1000 条信用贷款记录                                          |
| 标签     | `default`：0=信用良好(700)，1=信用违约(300)，**违约率 30%（类别不均衡）** |
| 特征     | Age、Sex、Job、Housing、Saving accounts、Checking account、Credit amount、Duration、Purpose |
| 特征工程 | 数值列(Age/Credit amount/Duration/Job)标准化；类别列 Sex/Housing/Saving/Checking/Purpose One-Hot 编码；缺失值分别用中位数/`unknown` 填补 |

**关于标签的关键处理：** Kaggle 上流传的 `german_credit_data.csv` 版本被**剥离了风险标签列**，无法直接做有监督分类。本实验用 UCI 原始 `german.data`（含真实风险标签）与 Kaggle 特征文件做**逐行对齐**——amount、duration、age 三列已验证逐行 1000/1000 完全一致，Sex 列按 UCI A9 属性解码后亦一致，据此把真实风险标签按行号挂回。代码内置自动重建逻辑，无标签文件时会自动从 `german.data` 重新挂标签。

## **三、实验设计**

- **模型**：RBF 核软间隔 SVC，`class_weight="balanced"` 应对 30% 违约的不均衡。
- **预处理流水线**：`ColumnTransformer`（数值标准化 + 类别 One-Hot）→ `SVC`，整体 `Pipeline` 防止数据泄漏。
- **调参**：`GridSearchCV` 网格搜索 C∈{0.1,1,10,100}、gamma∈{scale,0.01,0.1,0.5}，5 折分层交叉验证，以 **F1** 选参。
- **评估**：75%/25% 分层留出测试集，报告 Accuracy / Precision / Recall / F1 / ROC-AUC / 混淆矩阵 / 支持向量占比。

## **四、实验结果（实测，可复现）**

**最优超参数**：C = 1，gamma = 0.1（交叉验证 F1 = 0.604）

表格

| 指标                    | 数值                          |
| :---------------------- | :---------------------------- |
| Accuracy（准确率）      | 0.688                         |
| Precision（违约精确率） | 0.486                         |
| Recall（违约召回率）    | **0.720**                     |
| F1（违约）              | 0.581                         |
| ROC-AUC                 | **0.795**                     |
| 支持向量数              | 547 / 750（训练集 **72.9%**） |

**混淆矩阵（测试集，行为真实、列为预测）：**

表格

|              | 预测 good | 预测 default |
| :----------- | :-------- | :----------- |
| 真实 good    | 118 (TN)  | 57 (FP)      |
| 真实 default | 21 (FN)   | 54 (TP)      |

## **五、结果分析**

**1. 类别不均衡下的评估取向。** 数据 70% 良好 / 30% 违约，单看 Accuracy(0.688) 会高估性能。信用风险核心诉求是尽量不漏掉违约客户，因此以 Recall 与 ROC-AUC 为主：本模型违约召回 0.72（100 个违约抓出 72 个），漏报仅 21 个，AUC 0.795 明显优于随机基线。

**2. 支持向量稀疏性未充分显现（重要观察）。** 课件强调"只有支持向量决定超平面"，理论上支持向量应是少数，但本实验占比高达 72.9%——原因是信用数据两类高度**重叠**、线性不可分，多数样本落入间隔内或被误分（αᵢ>0）。这恰好反证 KKT 结论：间隔内/误分点才会 αᵢ>0，数据可分性差时支持向量自然趋多。

**3. 调参解读。** 网格搜索选到适中 C=1、gamma=0.1：C 没有选到 100 这样的大值，说明在重叠数据上加大惩罚只会把噪声点硬拉进来造成过拟合；gamma=0.1 让边界保留平滑度，是 F1/泛化的折中。

**4. 精确率偏低的代价。** 为追求高召回，模型把一部分好客户也判为违约（FP=57），精确率仅 0.486。这是"宁可错杀、不可漏放"的业务取舍。

**5. 总体水平判断。** 德国信用是公认**较难**的分类集，主流模型准确率通常 70%~76%，本结果与之量级一致，说明实现正确。

## **六、局限与改进方向**

- 类别重叠严重，RBF 核已接近该类数据天花板；可对比 LinearSVC，或对金额/期数做对数变换。
- 不均衡可进一步用 SMOTE 过采样或调决策阈值。
- 可加交叉特征（如"金额/期数"月供比）提升可分性。
- 用 PR-AUC 评估不均衡类别更稳健。

## **七、完整代码（直接复制到** `svm_german_credit.py` **即可运行）**

依赖：`pandas numpy scikit-learn matplotlib`。运行：`python svm_german_credit.py`

python



```
# -*- coding: utf-8 -*-
"""支持向量机 SVM 数据分类实验 —— 德国信用数据 (German Credit)
算法主线：超平面 -> 最大间隔 -> 拉格朗日对偶 -> KKT 稀疏性 -> RBF 核
数据集：课件 04_SVM_Detail 第04节真实案例②——German Credit / Statlog。
标签说明：Kaggle 版 german_credit_data.csv 被剥离标签，故提供两条等价路径：
  路径A：读已补齐标签的 german_credit_labeled.csv；
  路径B：用 german_credit_data.csv(无标签) + UCI 原始 german.data(含标签)，
         逐行对齐 + 关键列(amount/duration/age)校验后挂回真实标签。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay, classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data():
    labeled = os.path.join(HERE, "german_credit_labeled.csv")
    if os.path.exists(labeled):
        print("[数据] 读取已补齐标签的 german_credit_labeled.csv")
        return pd.read_csv(labeled)

    kag = os.path.join(HERE, "german_credit_data.csv")
    uci = os.path.join(HERE, "german.data")
    if not (os.path.exists(kag) and os.path.exists(uci)):
        raise FileNotFoundError("缺少数据文件：请提供 german_credit_labeled.csv，"
                                "或同时提供 german_credit_data.csv 与 UCI 的 german.data")
    rows = [ln.split() for ln in open(uci) if ln.strip()]
    u = pd.DataFrame(rows)
    df = pd.read_csv(kag)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    ok = ((u.iloc[:, 4].astype(int).values == df["Credit amount"].values) &
          (u.iloc[:, 1].astype(int).values == df["Duration"].values) &
          (u.iloc[:, 12].astype(int).values == df["Age"].values))
    if ok.sum() < len(df):
        raise RuntimeError(f"UCI 与 Kaggle 行序对齐失败：仅 {ok.sum()}/{len(df)} 行一致")
    df["Sex"] = np.where(u.iloc[:, 8].values == "A92", "female", "male")
    df["default"] = u.iloc[:, -1].astype(int).values - 1   # 1良好->0, 2违约->1
    print("[数据] 已用 UCI 原始标签挂回，违约率 %.1f%%" % (df["default"].mean() * 100))
    return df


def main():
    df = load_data()
    X = df.drop(columns=["default"])
    y = df["default"].values

    num_feat = ["Age", "Credit amount", "Duration", "Job"]
    cat_feat = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                         ("oh", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num_feat),
                             ("cat", cat_pipe, cat_feat)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=42)
    pipe = Pipeline([("pre", pre),
                     ("svm", SVC(kernel="rbf", class_weight="balanced",
                                 probability=True, random_state=42))])
    grid = GridSearchCV(pipe,
                        {"svm__C": [0.1, 1, 10, 100],
                         "svm__gamma": ["scale", 0.01, 0.1, 0.5]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        scoring="f1", n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    svm = best.named_steps["svm"]
    pred = best.predict(Xte)
    proba = best.predict_proba(Xte)[:, 1]
    n_sv = int(sum(svm.n_support_))

    print("\n=== 最优超参数 ===")
    print("C = %s, gamma = %s, 最优CV F1 = %.4f" % (svm.C, svm.gamma, grid.best_score_))
    print("支持向量数 = %d / 训练样本 %d (占比 %.1f%%)"
          % (n_sv, len(ytr), n_sv / len(ytr) * 100))
    print("\n=== 测试集评估 ===")
    print("Accuracy  = %.4f" % accuracy_score(yte, pred))
    print("Precision = %.4f" % precision_score(yte, pred))
    print("Recall    = %.4f" % recall_score(yte, pred))
    print("F1        = %.4f" % f1_score(yte, pred))
    print("ROC-AUC   = %.4f" % roc_auc_score(yte, proba))
    print(classification_report(yte, pred, target_names=["good", "default"]))

    cm = confusion_matrix(yte, pred)
    print("混淆矩阵 [[TN FP][FN TP]]:\n", cm)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["good", "default"]).plot(ax=ax[0], cmap="Blues")
    ax[0].set_title("混淆矩阵 (测试集)")
    fpr, tpr, _ = roc_curve(yte, proba)
    auc = roc_auc_score(yte, proba)
    ax[1].plot(fpr, tpr, lw=2, label="SVM RBF  AUC=%.3f" % auc)
    ax[1].plot([0, 1], [0, 1], "--", color="gray", label="随机基线")
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    ax[1].legend(); ax[1].set_title("ROC 曲线")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "svm_result_plots.png"), dpi=130)
    print("\n图表已保存：svm_result_plots.png")


if __name__ == "__main__":
    main()
```

## **八、运行与推送到仓库**

bash



```
git init
git add README.md svm_german_credit.py svm_result_plots.png german_credit_labeled.csv
git commit -m "SVM 德国信用数据分类实验"
git remote add origin <你的仓库地址>
git push -u origin main
```