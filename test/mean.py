import numpy as np
import matplotlib.pyplot as plt

mat = np.matrix([
                [1,2,3,4],
                [5,8,7,1],
                [9,10,4,12],
                [12,18,14,46]
  ])

print mat.shape
print mat

prod = mat * mat
print 'prod'
print prod

prod_sum = np.sum(prod)
print 'prod_sum'
print prod_sum

mat_mean = np.mean(mat, axis=0)
print mat_mean

mat = mat - mat_mean
print 'mat after mean subtraction'
print mat

mat = np.hstack([mat, np.ones((mat.shape[0],1))])
print mat

#plt.figure(figsize=(4,4))
#plt.imshow(mat_mean.astype('uint8'))
#plt.show()
