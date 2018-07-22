import tensorflow as tf

# 在线性代数中，当两个矩阵相乘时，第一个矩阵的列数必须等于第二个矩阵的行数。 不存在广播
with tf.Graph().as_default():
    # Create a martix (2-d tensor) with 3 rows and 4 columns.
    x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]],
                    dtype=tf.int32)

    # Create a martrix with 4 rows and 4 columns
    y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

    # Multiply 'x' by 'y'
    # The resulting matrix will have 3 rows and 2 columns.
    matrix_multiply_result = tf.matmul(x, y)

    with tf.Session() as sess:
        print(matrix_multiply_result.eval())
