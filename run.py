import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bams.data import VectorData
from bams.learners import ActiveLearner
from bams.query_strategies import BALD, HyperCubePool
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 加载并预处理数据
df = pd.read_csv("dataset/BostonHousing.csv")

# 新增训练测试分割（80%训练，20%测试）
X = df.drop("medv", axis=1).values
y = df["medv"].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 修改归一化部分，仅用训练数据拟合
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_norm = scaler_x.fit_transform(X_train)
y_train_norm = scaler_y.fit_transform(y_train)

# 测试集归一化（使用训练集的scaler）
X_test_norm = scaler_x.transform(X_test)
y_test_norm = scaler_y.transform(y_test)

# 修改查询池创建为使用训练数据分布
pool = HyperCubePool(dim=X_train_norm.shape[1], num_points=1000)

# 修改oracle函数使用训练数据
def housing_oracle(x):
    """模拟数据查询接口"""
    distances = np.linalg.norm(X_train_norm - x, axis=1)
    idx = np.argmin(distances)
    return y_train_norm[idx][0]


# 初始化主动学习器
learner = ActiveLearner(
    query_strategy=BALD(pool=pool),
    budget=10,
    base_kernels=["SE", "LIN", "PER", "LG"],
    max_depth=3,
    ndim=X_train_norm.shape[1]
)

# 记录训练过程
posteriors_history = []
predictions = []
true_values = []

# 运行学习循环
while learner.budget > 0:
    x = learner.next_query()
    y_true = housing_oracle(x)

    # 查询并更新模型
    y_pred = learner.query(housing_oracle, x)
    learner.update(x, y_pred)

    # 记录结果
    posteriors_history.append(learner.posteriors / np.sum(learner.posteriors))
    predictions.append(scaler_y.inverse_transform([[y_pred]]).item())
    true_values.append(scaler_y.inverse_transform([[y_true]]).item())

# 运行学习循环后新增测试集预测部分
# ==================== 新增测试集预测 ====================
X_test_norm = scaler_x.transform(X_test)
test_predictions = []
for x in X_test_norm:
    y_pred = learner.query(housing_oracle, x)
    test_predictions.append(scaler_y.inverse_transform([[y_pred]]).item())
    
y_test_true = scaler_y.inverse_transform(y_test_norm).flatten()
# ========================================================

# 结果可视化
plt.figure(figsize=(15, 5))

# 1. 归一化模型证据（保持不变）
plt.subplot(131)
for i in range(posteriors_history[0].shape[0]):
    plt.plot([p[i] for p in posteriors_history], label=f'Model {i}')
plt.title("Normalized Model Evidence")
plt.xlabel("Iteration")
plt.ylabel("Evidence")

# 2. 修改RMSE曲线为测试集评估
plt.subplot(132)
test_rmse = mean_squared_error(y_test_true, test_predictions, squared=False)
plt.axhline(test_rmse, color='r', linestyle='-', label=f'Test RMSE: {test_rmse:.2f}')
plt.title("Final Test RMSE")
plt.legend()

# 3. 修改预测值对比为测试集数据
plt.subplot(133)
plt.scatter(y_test_true, test_predictions, alpha=0.6)
plt.plot([min(y_test_true), max(y_test_true)],
         [min(y_test_true), max(y_test_true)], 'k--')
plt.title("Test Set: True vs Predicted Values")
plt.xlabel("True Values")
plt.ylabel("Predictions")

plt.tight_layout()
plt.savefig("results.png")
plt.show()