import numpy as np

mat = np.matrix([
                [5,2,3],
                [27,3,16],
                [-2,6,3],
                [1,7,4]
                ])

max_num = np.amax(mat, axis=1)

max_idx = np.argmax(mat, axis=1)

print 'mat'
print mat
print 'max_num'
print max_num
print 'max_idx'
print max_idx
