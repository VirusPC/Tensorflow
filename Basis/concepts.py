import tensorflow as tf

# 直接创建变量
x = tf.constant([5, 2])
y = tf.Variable([5])

# 先创建变量，后分配值，需要指定默认值
y2 = tf.Variable([0])
y2 = y2.assign([5])

# 定义一些常量或变量后，您可以将它们与其他指令（如 tf.add）结合使用

# 图必须在 TensorFlow 会话中运行，会话存储了它所运行的图的状态
# 在使用 tf.Variable 时，您必须在会话开始时调用 tf.global_variables_initializer，
# 以明确初始化这些变量
initialization = tf.global_variables_initializer()
with tf.Session() as sess:
    print(x.eval())
