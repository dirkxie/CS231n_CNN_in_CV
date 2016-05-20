import numpy as np

num_train = 10
num_validation = 5
#mask = range (num_train, num_train+num_validation)
#mask = range (num_train)
mask = np.random.choice (num_train, 3, replace=False)

print mask
