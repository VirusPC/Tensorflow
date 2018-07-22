import numpy as np
import logging

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 2.0, 2.0])
print(a*b)  # [2. 4. 6.]

# 效率更高
c = 2.0
print(a*c)  # [2. 4. 6.]

# Two dimensions are compatible when
#        1.they are equal, or
#        2.one of them is 1
# 即若对应维度大小不同，则其中一个大小必须为1

# Arrays do not need to have the same number of dimensions.
A = np.ones([8, 1, 6, 1])
B = np.ones([7, 1, 5])
C = A*B
print(C.shape)  # (8, 7, 6, 5)

A = np.ones([15, 3, 5])
B = np.ones([15, 1, 5])
C = A*B
print(C.shape)  # (15, 3, 5)

print("*************************************")
x = np.arange(4)  # [0 1 2 3]
xx = x.reshape(4, 1)  # [[0] [1] [2] [3]]
y = np.ones(5)  # [1. 1. 1. 1. 1. 1.]
z = np.ones((3, 4))
print(x.shape)
print(y.shape)
try:
    print(x + y)
except Exception as e:
    logging.exception(e)
    print("exception occured")
print(xx.shape)
print(y.shape)
print((xx+y).shape)
print(xx+y)
print(x.shape)
print(z.shape)
print((x+z).shape)
print(x+z)

print('***************')
x = np.arange(4)  # [0 1 2 3]
xx = x.reshape(2, 2)

y = np.arange(6)  # [0 1 2 3]
yy = y.reshape(2,3)

print(xx*yy)