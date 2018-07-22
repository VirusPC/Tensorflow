import tensorflow as tf
import  matplotlib as plt  # 数据集可视化
import  numpy as np  # 低级数字Python库
import pandas as pd  # 较高级别的数字Python库

# TensorFlow 提供了一个默认图。不过，我们建议您明确创建自己的 Graph，以便跟踪状态

# Create a graph
g = tf.Graph()

# Establish the graph as the "default" graph
with g.as_default():
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    sum = tf.add(x, y, name="x_y_sum")

    z = tf.constant(4, name="z_const")
    new_sum = tf.add(sum, z, name="x_y_z_sum")
    # Now create as session
    # The session will run the default graph
    with tf.Session() as sess:
        print(new_sum.eval())

# TensorFlow 编程本质上是一个两步流程：
#   ·将常量、变量和指令整合到一个图中。
#   ·在一个会话中评估这些常量、变量和指令。
