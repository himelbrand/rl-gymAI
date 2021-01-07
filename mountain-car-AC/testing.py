import numpy as np

x = np.array([1, 2, 3, 4, 5])
w = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]])

Y = x * w

print(Y)
maxis = Y.max(axis=1).reshape(3, 1)
print(maxis)
pi = (np.exp(Y-maxis) / np.sum(np.exp(Y-maxis)))

print(pi)


