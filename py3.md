## 作业 1（推导题）：证明正交设计下 $\hat{\boldsymbol\beta}*{\text{Ridge}}=\dfrac{1}{1+\lambda}\hat{\boldsymbol\beta}*{\text{OLS}}$

### 一、Ridge 的闭合解怎么来

Ridge 回归最小化的目标函数为

$$L(\boldsymbol\beta)=|Y-X\boldsymbol\beta|_2{2}+\lambda|\boldsymbol\beta|_2{2}=(Y-X\boldsymbol\beta){T}(Y-X\boldsymbol\beta)+\lambda,\boldsymbol\beta{T}\boldsymbol\beta$$

对 $\boldsymbol\beta$ 求梯度并令其为 0：

$$\nabla_{\boldsymbol\beta}L=-2X^{T}(Y-X\boldsymbol\beta)+2\lambda\boldsymbol\beta=\mathbf 0$$

整理得正规方程

$$X{T}X\boldsymbol\beta+\lambda\boldsymbol\beta=X{T}Y\quad\Longrightarrow\quad\big(X^{T}X+\lambda I\big)\boldsymbol\beta=X^{T}Y$$

当 $\lambda>0$ 时 $X^{T}X+\lambda I$ 严格正定、必可逆，故闭合解为

$$\boxed{\ \hat{\boldsymbol\beta}_{\text{Ridge}}=\big(X^{T}X+\lambda I\big){-1}X{T}Y\ }$$

普通最小二乘闭合解为 $\hat{\boldsymbol\beta}_{\text{OLS}}=(X{T}X){-1}X^{T}Y$。

### 二、代入正交条件 $X^{T}X=I$

正交设计（特征两两正交且已标准化为单位范数）意味着 $X^{T}X=I$，代入闭合解：

$$\hat{\boldsymbol\beta}_{\text{Ridge}}=(I+\lambda I){-1}X{T}Y=\big((1+\lambda)I\big){-1}X{T}Y=\frac{1}{1+\lambda}X^{T}Y$$

又在 $X^{T}X=I$ 下 OLS 解退化为

$$\hat{\boldsymbol\beta}_{\text{OLS}}=(X{T}X){-1}X{T}Y=I{-1}X{T}Y=X{T}Y$$

两式合并即得

$$\boxed{\ \hat{\boldsymbol\beta}*{\text{Ridge}}=\frac{1}{1+\lambda},\hat{\boldsymbol\beta}*{\text{OLS}}\ ,\qquad \hat\beta_{j,\text{Ridge}}=\frac{1}{1+\lambda},\hat\beta_{j,\text{OLS}}\ (\forall j)\ }$$

证毕。

### 三、结论的含义

1. **等比整体收缩**：所有系数乘同一缩放因子 $\dfrac{1}{1+\lambda}\in(0,1]$，方向不变、长度按同一比例缩短，没有任何系数被单独放大或淘汰。
2. **永不为 0**：只要 $\hat\beta_{j,\text{OLS}}\neq0$ 且 $\lambda$ 有限，就有 $\hat\beta_{j,\text{Ridge}}\neq0$。这正是 Ridge 不做特征选择（解稠密）的代数根源。
3. **偏差–方差权衡**：$\lambda\to0$ 因子 $\to1$ 退化为 OLS（无偏高方差）；$\lambda\to\infty$ 因子 $\to0$ 解趋于零向量（高偏差低方差）；中间存在使泛化误差最小的 $\lambda$。
4. **与 Lasso 对照**：同样正交场景下 Lasso 解为软门限 $\hat\beta_{j,\text{Lasso}}=S_\lambda(\hat\beta_{j,\text{OLS}})=\operatorname{sign}(\hat\beta_j)\max(|\hat\beta_j|-\lambda/2,,0)$，小系数被精确压到 0 产生稀疏解——正是圆约束与菱形约束几何差异的代数体现。

------

## 作业 3（分析题）：为什么正则化前必须做 Z-score 标准化？不做会怎样？

### 一、问题根源：惩罚对各系数“一视同仁”，但系数大小取决于量纲

Ridge 的 $\lambda\sum_j\beta_j^{2}$、Lasso 的 $\lambda\sum_j|\beta_j|$，对每个 $\beta_j$ 用同一个 $\lambda$、同一把尺子。但**系数 $\beta_j$ 的数值大小并不是变量重要性的客观度量，它依赖 $x_j$ 的量纲（单位、尺度）**：

$$\text{若 }x_j\text{ 的单位被放大 }1000\text{ 倍，则同一物理效应下 }\hat\beta_j\text{ 会缩小约 }1000\text{ 倍}$$

于是惩罚强度被“量纲绑架”：

- **量纲（方差）大的变量**：回归系数天然偏小，惩罚贡献几乎可忽略 → **几乎不受罚，被保留**。
- **量纲（方差）小的变量**：要产生同样影响必须取较大的系数，于是被重点压制 → **被过度收缩甚至误淘汰**。

结果：谁被收缩、谁被保留不再由变量对响应的真实贡献决定，而由“单位写成米还是公里”这种任意选择决定。这会让正则化（尤其 Lasso 的变量筛选）方向严重失真。

**直观例子**：同一收入数据，变量 A 用“元”计（数值上万、系数很小），变量 B 用“万元”计（数值很小、系数很大）。在不标准化的情况下 Lasso 可能把 B 罚成 0 却放过 A——纯属单位造成的假象。

### 二、第二个坑：截距项

惩罚若作用在截距 $\beta_0$ 上，会把整个拟合曲面强行往原点拉，造成系统性偏移。规范做法是先对响应与特征**中心化**（使 $\bar x_j=0,\ \bar y=0$），此时 $\hat\beta_0=\bar y$ 不受惩罚影响；工程上即“只对特征标准化、截距不参与惩罚”。

### 三、正确做法：Z-score 标准化

把每个特征减均值、除标准差，统一到**均值 0、方差 1** 的尺度，使“系数大小”重新可与“变量重要性”挂钩，惩罚才公平：

$$x_{ij}^{\text{scaled}}=\frac{x_{ij}-\bar x_j}{s_j},\qquad \bar x_j=\frac1n\sum_i x_{ij},\quad s_j=\sqrt{\frac1n\sum_i(x_{ij}-\bar x_j)^2}$$

标准化后各变量对惩罚“同权”，收缩才反映真实贡献，Lasso 的稀疏选择、Ridge 的相对收缩才有意义。

### 四、工程要点（防数据泄漏）

标准化参数（$\bar x_j,\ s_j$）**只能在训练集上估计**，再原样套用到验证/测试集：

```
scaler = StandardScaler().fit(X_train)      # 仅在训练集 fit
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)         # 测试集只 transform，绝不重新 fit
```

若用全量数据 fit 后再切分，测试集信息会泄漏进缩放参数，导致评估过于乐观。