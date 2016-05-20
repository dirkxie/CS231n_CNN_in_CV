import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)   #X[i].shape = (1,D), W.shape = (D,C), scores.shape = (1,C)
    correct_class_score = scores[y[i]]
    gradient_count = 0
    for j in xrange(num_classes):
      if j == y[i]: # if j = correct label, continue
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:]    #add X[i,:].T from the y[i] column of dW (all incorrect labels)
        gradient_count += 1
    #outside for loop
    dW[:,y[i]] -= gradient_count * X[i,:] #subtract X[i,:].T from the y[i] column of dW (correct label)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
        
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_feature = X.shape[1]
    
  """
  #one loop
  # compute the loss
  for i in xrange(num_train):
    scores = X[i].dot(W)   #X[i].shape = (1,D), W.shape = (D,C), scores.shape = (1,C)
    #correct_class_score = scores[y[i]]
    margins = scores-scores[y[i]]+1
    margins[y[i]] = 0
    margins = np.maximum(0, margins)
    loss += np.sum(margins)
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  """
  
  #no loop
  scores = X.dot(W) #X.shape = (N,D), W.shape = (D,C), scores.shape = (N,C)
  correct_scores = scores[range(scores.shape[0]), y]
  correct_scores = np.reshape(correct_scores, (num_train,1))
  scores = scores - correct_scores + 1
  scores[range(scores.shape[0]), y] = 0
  scores = np.maximum(0, scores)
  loss = np.sum(scores)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
    
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # compute the loss and the gradient
  binary = scores
  binary[scores>0] = 1
  row_sum = np.sum(binary, axis=1)
  binary[range(num_train), y] = -row_sum[range(num_train)]
  dW = X.T.dot(binary)
  #print dW.shape
  dW /= num_train
  dW += reg*W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
