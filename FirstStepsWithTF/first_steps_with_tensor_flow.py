from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# import tensorflow.contrib.estimator as es

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
# pd.set_option('display.max_rows', 10)
# pd.set_option('display.max_columns', 3)
# pd.set_option('precision', 1)


california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")

# 对数据进行随机化处理，以确保不会出现任何病态排序结果（可能会损害随机梯度下降法的效果）。
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)

# 此外，我们会将 median_house_value 调整为以千为单位，
california_housing_dataframe["median_house_value"] /= 1000.0


# 输出关于各列的一些实用统计信息快速摘要：样本数、均值、标准偏差、最大值、最小值和各种分位数。
print(california_housing_dataframe.describe())

# 第1步：定义特征并配置特征列

# 定义输入特征: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]

# 为 total_rooms 配置数值化特征
# total_rooms 数据的形状是一维数组（每个街区的房间总数列表）。这是 numeric_column 的默认形状，
# 因此我们不必将其作为参数传递。

feature_columns = [tf.feature_column.numeric_column("total_rooms")]


# 第2步：定义目标

# 下来，我们将定义目标，也就是 median_house_value。同样，我们可以从 california_housing_dataframe 中提取它：

# Define the label.
targets = california_housing_dataframe["median_house_value"]


# 第3步：配置LinearRegressor

# 设置梯度下降函数为优化器，设置学习速率为0.0000001。
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# 为了安全起见，通过 clip_gradients_by_norm 将梯度裁剪应用到优化器。梯度裁剪可确保梯度大小
# 在训练期间不会变得过大，而梯度过大会导致梯度下降法失败。设置最大下降范围为5，防止梯度爆炸
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# 利用特征列和优化器定义线性回归器
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)


# 第4步：定义输入函数
# 告诉 TensorFlow 如何对数据进行预处理，以及在模型训练期间如何批处理、随机处理和重复数据。

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """训练含有一个特征的线性回归模型

    Args:
        features: 特征的 pandas DataFrame
        targets: 目标的 pandas DataFrame
        batch_size: 传递给模型的每批数据的大小
        shuffle: True 或 False。 数据是否在训练期间以随机方式传递到模型。
        num_epochs: 输入数据的重复次数。如果将默认值 num_epochs=None 传递到 repeat()，输入数据会无限期重复。
    Returns:
        形如(features, labels)的元组，用于下次处理
        """
    # 将 Pandas 特征数据转换成 NumPy 数组字典。
    features = {key: np.array(value) for key, value in dict(features).items()}

    # 利用TensorFlow Dataset API 构造一个dataset，并且配置每批数据的大小和重复次数
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # 如果 shuffle == True 则打乱数据
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # 返回下一批数据
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# 第5步：训练模型
# 现在，我们可以在 linear_regressor 上调用 train() 来训练模型。我们会将 my_input_fn 封装在 lambda 中，
# 以便可以将 my_feature 和 target 作为参数传入

# 首先，我们会训练100步
linear_regressor.train(
    input_fn=lambda: my_input_fn(my_feature, targets),
    steps=100
)


# 第6步：评估模型
# 我们基于该训练数据做一次预测，看看我们的模型在训练期间与这些数据的拟合情况。
# 注意：训练误差可以衡量您的模型与训练数据的拟合情况，但并不能衡量模型泛化到新数据的效果。

# 为预测创建一个输入函数
# 注意：由于我们对每个例子只做一个预测，所以我们不需要重复或者打乱数据
prediction_input_fn = lambda : my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# 在线性回归模型上调用predict()方法来进行预测
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# 将预测格式化为NumPy数组，以便计算误差指标
predictions = np.array([item['predictions'][0] for item in predictions])

# 打印均方误差错误以及根均误差错误
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)


min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print(calibration_data.describe())

sample = california_housing_dataframe.sample(n=300)

# 获取最大最小值
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# 获取权重和偏置项
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# 获取有关的数据.
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# 追至直线（ Plot our regression line from (x_0, y_0) to (x_1, y_1).）
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# 坐标的名称
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# 画出图的分散点
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()
