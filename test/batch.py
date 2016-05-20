import numpy as np

a = np.matrix([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12]
            ])
print a
print a[range(1,3),:]

b = np.array([1,2,3,4,5,6,7,8])
print b
print b[2:5]

