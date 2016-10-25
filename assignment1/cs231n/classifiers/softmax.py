import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  num_feature = W.shape[0]
  
  for i in range(num_train):
      scores = X[i,:].dot(W)
      
      #for numeric stability, do normalization first by subtracting max(score)
      """
      scores -= np.max(scores)
      correct_score = scores[y[i]]
      #cross-entropy loss: L_i = -f_yi + log*sum(e^fj)
      exp_fj = np.exp(scores)
      sum_exp_fj = np.sum(exp_fj)
      loss_i = -correct_score + np.log(sum_exp_fj)
      loss += loss_i
      """
        
      #original form of cross-entropy loss:
      #subtract max for stability
      scores -= np.max(scores)
      norm_scores = np.exp(scores)/np.sum(np.exp(scores))
      loss_i = -np.log(norm_scores[y[i]])
      loss += loss_i

      #gradient of loss with respect to W: 
      P = norm_scores # 1*C row
      P[y[i]] -= 1    # 1*C row
      Pmat = np.asmatrix(P)
      Xmat = np.asmatrix(X[i,:])
      #print(Pmat, type(Pmat), np.size(Pmat))
      #print(Xmat, type(Xmat), np.size(Xmat))
      dW += Xmat.T * Pmat # (1*D).T * 1*C = D*C, size of W
  loss = loss/num_train + 0.5*reg*np.sum(W*W)
  dW /= num_train
  dW += reg*W
    
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
    
  num_train = X.shape[0]
  num_class = W.shape[1]
  num_feature = W.shape[0]

  scores = np.asmatrix(X.dot(W))
  scores_max = np.amax(scores, axis=1)
  scores -= scores_max
  scores_P = np.exp(scores)/np.sum(np.exp(scores), axis=1)
  # loss
  scores_loss = -np.log(scores_P)
  loss_along_row = np.sum(scores_loss, axis=1)
  loss = np.sum(scores_loss)
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
  # gradient
  minus_1_mat = np.zeros(scores_P.shape)
  minus_1_mat[range(num_train), y] = 1
  scores_P -= minus_1_mat
  Xmat = np.asmatrix(X)
  dW = Xmat.T*scores_P
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

