import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "KaiTi"]  # 适配Windows中文
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

# 定义 UCI 官方数据集的列名和标签列
# 前20列是特征，第21列是标签(1=Good, 2=Bad)
COLUMN_NAMES = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_commitment', 'personal_status',
    'other_parties', 'residence_since', 'property_magnitude', 'age',
    'other_payment_plans', 'housing', 'existing_credits', 'job', 'num_maintaining',
    'telephone', 'foreign_worker', 'default'
]

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data():
    # 尝试读取 UCI 原始数据，兼容 .data 和 .doc 后缀
    uci_paths = [os.path.join(HERE, "german.data"), os.path.join(HERE, "german.doc")]
    uci_file = None
    for path in uci_paths:
        if os.path.exists(path):
            uci_file = path
            break

    if uci_file is None:
        raise FileNotFoundError("未找到数据文件。请确保 E:\\pyc\\ 文件夹下存在 'german.data' 或 'german.doc'")

    print(f"[数据] 正在读取原始数据: {os.path.basename(uci_file)}...")
    # 读取空格分隔的无表头文件
    df = pd.read_csv(uci_file, sep='\s+', header=None, names=COLUMN_NAMES)

    # 标签处理：1 代表信用好(0)，2 代表违约(1)
    df['default'] = (df['default'].astype(int) - 1)

    print(f"[数据] 加载成功。样本数: {len(df)}, 违约率: {df['default'].mean() * 100:.1f}%")
    return df


def main():
    try:
        df = load_data()
    except Exception as e:
        print(f"\n错误: {e}")
        print("请按 F2 重命名你的 'german.doc' 或 'german.data' 确保其位于 E:\\pyc\\ 目录下")
        return

    X = df.drop(columns=["default"])
    y = df["default"].values

    # 区分数值列和类别列
    # 根据 UCI 文档，数值列为: duration, credit_amount, installment_commitment, residence_since, age, existing_credits, num_maintaining
    num_feat = ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits",
                "num_maintaining"]
    cat_feat = [col for col in COLUMN_NAMES[:-1] if col not in num_feat]

    # 构建预处理管道
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                         ("oh", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", num_pipe, num_feat), ("cat", cat_pipe, cat_feat)])

    # 分割训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # 构建 SVM 模型管道
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))
    ])

    # 网格搜索最佳参数
    print("\n[训练] 正在进行网格搜索调参 (5折交叉验证)...")
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.1, 0.5]
    }

    grid_search = GridSearchCV(
        model_pipeline, param_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="f1", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    svm_clf = best_model.named_steps["svm"]
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算支持向量数量
    n_support_vectors = int(sum(svm_clf.n_support_))

    # 打印结果
    print("\n" + "=" * 40)
    print("=== 实验结果 (实测) ===")
    print("=" * 40)
    print(f"最优超参数: C={svm_clf.C}, gamma={svm_clf.gamma}")
    print(f"交叉验证 F1: {grid_search.best_score_:.4f}")
    print(
        f"支持向量数量: {n_support_vectors} / 训练样本 {len(y_train)} (占比 {n_support_vectors / len(y_train) * 100:.1f}%)")
    print(f"\n--- 测试集指标 ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=["信用良好", "违约"]))

    # 绘制图表
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 混淆矩阵
    disp1 = ConfusionMatrixDisplay(cm, display_labels=["信用良好", "违约"])
    disp1.plot(ax=ax[0], cmap="Blues")
    ax[0].set_title("混淆矩阵 (测试集)")

    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax[1].plot(fpr, tpr, lw=2, label=f"SVM RBF (AUC={auc:.3f})")
    ax[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="随机基线")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("ROC 曲线")
    ax[1].legend(loc="lower right")

    plt.tight_layout()
    save_path = os.path.join(HERE, "svm_result_plots.png")
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    print(f"\n[图表] 结果图表已保存至: {save_path}")


if __name__ == "__main__":
    main()