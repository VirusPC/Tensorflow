import tensorflow as tf

with tf.Graph().as_default():
    # Create an 8*2 matrix (2-D tensor)
    matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8],
                          [9, 10], [11, 12], [13, 14], [15, 16]],
                         dtype=tf.int32)

    # Reshape the 8#2 matrix into a 2*8 matrix
    reshaped_2_8_matrix = tf.reshape(matrix, [2, 8])

    # Reshape the 8*2 matrix into a 4*4 matrix
    reshaped_4_4_matrix = tf.reshape(matrix, [4, 4])

    # Reshape the 8*2 matrix into a 2*2**4 matrix
    reshaped_2_2_4_matrix = tf.reshape(matrix, [2, 2, 4])

    with tf.Session() as sess:
        print('Original matrix (8*2)')
        print(matrix.eval())
        print("Reshape matrix (2*8)")
        print(reshaped_2_8_matrix.eval())
        print("Reshape matrix (4*4)")
        print(reshaped_4_4_matrix.eval())
        print("Reshape matrix (2*2*4)")
        print(reshaped_2_2_4_matrix.eval())
