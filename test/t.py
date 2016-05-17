import numpy as np
from collections import Counter

train = np.matrix([
              [1,2,3,4],
              [5,6,7,8],
              [6,2,6,4],
              [1,5,4,1]
              ])

test = np.matrix([
              [2,2,3,4],
              [5,8,7,8],
              [6,2,6,4],
              [1,5,4,1]
              ])

print 'train:'
print train
print 'test:'
print test

dist = np.zeros((4,4))
#for i in xrange(4):
#  for j in xrange(4):
#    dist[i,j] = np.linalg.norm(train[i,:]-test[j,:])

diff1 = train-test[0,:]
print 'diff1'
print diff1

for i in xrange(4):
  dist[i,:] = np.linalg.norm(train-test[i,:], axis=1)
  
print 'dist:'
print dist

##### split
sp = np.array([1,2,3,4,5,6,7,8,9])
sp = np.array_split(sp,3)
print 'sp'
print sp

##### concatenate
con_1 = np.concatenate((sp[0:1],sp[2:]), axis=0)
con_3 = np.vstack(sp[0:1]+sp[2:])
con_2 = sp[1]
print 'con_1'
print con_1
print 'con_3'
print con_3
print 'con_2'
print con_2

