import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 准备数据 (模拟 Carseats 数据集)
np.random.seed(42) # 设置随机种子，保证每次生成的随机数一样
n = 100

data = {
    'Sales': np.random.uniform(0, 20, n),       # 销售额
    'Price': np.random.uniform(60, 150, n),     # 价格
    'Income': np.random.uniform(10, 100, n),    # 收入
    'Advertising': np.random.uniform(0, 20, n), # 广告费
    # 货架位置 (ShelveLoc)
    'ShelveLoc': np.random.choice(['Bad', 'Medium', 'Good'], n)
}

df = pd.DataFrame(data)

print("数据预览")
print(df.head())
print("\n")
# 建立多元线性回归模型
# 1. 提取自变量 X 和 响应变量 y
X = df[['Price', 'Income', 'Advertising', 'ShelveLoc']]
y = df['Sales']
# 2. 处理定性变量 ShelveLoc
X_with_dummies = pd.get_dummies(X, columns=['ShelveLoc'], drop_first=True)
# 3. 添加常数项 (截距)
X_model = sm.add_constant(X_with_dummies)
# 4. 拟合模型 (OLS = 普通最小二乘法)
model = sm.OLS(y, X_model).fit()

# 第三步：输出结果并解答题目问题
print("模型拟合报告")
print(model.summary())
print("\n")
#问题 1：指出 ShelveLoc 的基准组是什么？
print("问题1解答")
print("代码中使用了drop_first=True，这意味着它丢弃了字母排序的第一个类别作为参照。")
print("因为Bad<Good<Medium，所以被丢弃的'Bad'就是基准组。")
print("\n")
#问题 2：解读 ShelveLoc[Good] 系数的商业含义
print("问题2解答：ShelveLoc[Good] 系数的含义")
# 获取 Good 的系数
coef_good = model.params['ShelveLoc_Good']
print(f"ShelveLoc[Good] 的系数约为：{coef_good:.4f}")
print(f"含义解读：")
print(f"在保持价格(Price)、收入(Income)和广告投入(Advertising)不变的情况下，")
print(f"相比于基准组（货架位置为 'Bad'），")
print(f"将货架位置调整为 'Good'，平均会使销售额(Sales) { '增加' if coef_good > 0 else '减少' } {abs(coef_good):.4f} 个单位。")
print("\n")

#问题3：计算VIF(方差膨胀因子)
print("问题解答3：多重共线性评估(VIF)")
#VIF 用于检测自变量之间是否存在高度相关（比如价格和收入是不是同步变化的）
#VIF > 5 或 10 表示存在严重共线性；VIF 接近 1 表示没有共线性。
vif_data = pd.DataFrame()
vif_data["Variable"] = X_model.columns # 变量名
vif_data["VIF"] = [variance_inflation_factor(X_model.values, i) for i in range(X_model.shape[1])]
print(vif_data)
print("\n结论：")
for index, row in vif_data.iterrows():
    if row['VIF'] > 5:
        print(f"- 变量 {row['Variable']} 的 VIF 为 {row['VIF']:.2f}，可能存在多重共线性风险。")
    else:
        print(f"- 变量 {row['Variable']} 的 VIF 为 {row['VIF']:.2f}，共线性风险较低。")