import numpy as np

def calculate_mae(y_true, y_pred):
    """
    计算平均绝对误差 (Mean Absolute Error, MAE)。

    MAE 是衡量预测值与真实值之间差异的平均大小。
    MAE = (1/n) * Σ|y_true[i] - y_pred[i]|

    Args:
        y_true (list or np.ndarray): 真实值列表或数组。
        y_pred (list or np.ndarray): 预测值列表或数组。

    Returns:
        float: 平均绝对误差，如果输入无效则返回 NaN。
    """
    if not isinstance(y_true, (list, np.ndarray)) or not isinstance(y_pred, (list, np.ndarray)):
        print("输入必须是列表或 NumPy 数组。")
        return np.nan

    if len(y_true) == 0 or len(y_pred) == 0:
        print("输入数组不能为空。")
        return np.nan

    if len(y_true) != len(y_pred):
        print("输入数组长度必须相等。")
        return np.nan

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def calculate_pearson_correlation(x, y):
    """
    计算两个数据集之间的皮尔逊相关系数 (Pearson Correlation Coefficient)。

    皮尔逊相关系数衡量两个变量之间的线性相关强度和方向，取值范围在 -1 到 1 之间。
    1 表示完全正相关，-1 表示完全负相关，0 表示没有线性相关。

    Args:
        x (list or np.ndarray): 第一个数据集。
        y (list or np.ndarray): 第二个数据集。

    Returns:
        float: 皮尔逊相关系数，如果输入无效或标准差为零则返回 NaN。
    """
    if not isinstance(x, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
        print("输入必须是列表或 NumPy 数组。")
        return np.nan

    if len(x) < 2 or len(y) < 2: # 至少需要两个数据点才能计算相关系数
        print("数据集长度至少需要为 2。")
        return np.nan

    if len(x) != len(y):
        print("输入数据集长度必须相等。")
        return np.nan

    x = np.array(x, dtype=float) # 确保数据类型为浮点数
    y = np.array(y, dtype=float)

    # 检查标准差是否为零，避免除以零的错误
    if np.std(x) == 0 or np.std(y) == 0:
        print("数据集中存在方差为零的情况，无法计算皮尔逊相关系数。")
        return np.nan

    # np.corrcoef 返回相关系数矩阵，我们需要 (0, 1) 位置的值
    correlation_matrix = np.corrcoef(x, y)
    pearson_r = correlation_matrix[0, 1]
    return pearson_r

# 示例用法：
if __name__ == '__main__':
    # 示例数据
    true_heart_rates = [62.28255941,
158.6712429,
59.65831944,
64.4480828,
57.54352966,
58.00181141,
68.02629749,
52.92178109,
50.22662524,
54.64787487,
53.22030119,
53.08722761,
94.69862325,
99.66649051,
171.5392273,
84.22918936,
120.9737664,
91.89786172,
94.8406194,
81.02538917,
95.75293647,
51.5139422,
57.77668322
]
    measured_heart_rates = [96,
97,
101,
87,
92,
89,
91,
92,
91,
93,
93,
93,
93,
91,
89,
92,
94,
96,
97,
95,
94,
94,
90
]

    # 计算平均绝对误差
    mae = calculate_mae(true_heart_rates, measured_heart_rates)
    if not np.isnan(mae):
        print(f"平均绝对误差 (MAE): {mae:.2f}")

    # 计算皮尔逊相关系数
    pearson_r = calculate_pearson_correlation(true_heart_rates, measured_heart_rates)
    if not np.isnan(pearson_r):
        print(f"皮尔逊相关系数: {pearson_r:.2f}")