import numpy as np

mat = np.matrix([
                [17,22,3,4],
                [5,8,7,15],
                [9,10,4,12],
                [22,18,14,46],
                [15,2,63,6]
  ])

print 'mat'
print mat
print 'mat.shape'
print mat.shape[0], ' ', mat.shape[1]

#pick elments from mat
y = [2,0,3,1,0]
dim1 = range(mat.shape[0])

print 'range(mat.shape[0])'
print dim1
print 'y'
print y

pick = mat[dim1, y]
pick = np.reshape(pick, (mat.shape[0],1))

print 'pick'
print pick
print 'pick.shape'
print pick.shape

mat_mi_pick = mat-pick

print 'mat - pick'
print mat_mi_pick

thresh = np.maximum(0, mat_mi_pick)
print 'thresh'
print thresh

binary = thresh
binary[thresh>0] = 1
print 'binary'
print binary

col_sum = np.sum(binary, axis=0)
print 'col_sum'
print col_sum



"""
mat[dim1, y] = -1000
print 'modified mat'
print mat
print 'max(0,mat)'
print np.maximum(0, mat)
"""
