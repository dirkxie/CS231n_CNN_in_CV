import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim) # 3*32*32 by 100
    self.params['b1'] = np.asmatrix(np.zeros(hidden_dim))
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes) # 100 by 10
    self.params['b2'] = np.asmatrix(np.zeros(num_classes))
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    
    # 0. get params
    W1 = self.params['W1']
    b1 = self.params['b1']
    W2 = self.params['W2']
    b2 = self.params['b2']
    # 1. forward pass
    FC1, cache_FC1 = affine_forward(X, W1, b1)
    ReLU1, cache_ReLU1 = relu_forward(FC1)
    FC2, cache_FC2 = affine_forward(ReLU1, W2, b2)
    scores = np.asarray(FC2)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # 2. normalize scores, softmax function, cross-entropy, loss
    N = X.shape[0]
    reg = self.reg
    scores_exp = np.exp(scores)
    scores_norm = scores_exp / np.sum(scores_exp, axis=1).reshape(scores_exp.shape[0],1)
    scores_y = scores_norm[range(N), list(y)]
    
    loss = -np.log(scores_y)
    loss = np.sum(loss)/N
    loss += 0.5*reg*(np.sum(np.multiply(W1,W1)) + np.sum(np.multiply(W2,W2)))
    
    # 3. backward pass
    dscores = scores_norm.copy()
    dscores[range(N), list(y)] -= 1
    
    dReLU1, dW2, db2 = affine_backward(dscores, cache_FC2)
    grads['W2'] = dW2/N + reg*W2
    grads['b2'] = db2/N
    
    dFC1 = relu_backward(dReLU1, cache_ReLU1)
    dX, dW1, db1 = affine_backward(dFC1, cache_FC1) 
    grads['W1'] = dW1/N + reg*W1
    grads['b1'] = db1/N
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    for i in range(1, self.num_layers+1): #number of hidden layers + 1 = number of W, b
        layer_input_dim = input_dim if i == 1 else hidden_dims[i-2]
        layer_output_dim = num_classes if i == self.num_layers else hidden_dims[i-1]
        
        self.params['W'+str(i)] = np.random.normal(loc=0.0, scale=weight_scale, size=(layer_input_dim, layer_output_dim))
        self.params['b'+str(i)] = np.zeros(layer_output_dim)
        
        if use_batchnorm and i!=self.num_layers:
            self.params['gamma'+str(i)] = np.ones(layer_output_dim)
            self.params['beta'+str(i)] = np.zeros(layer_output_dim)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    crnt_input = X
    affine_relu_cache = {}
    affine_bn_relu_cache = {}
    dropout_cache = {}
        
    # affine layers with ReLU before last layer
    for i in range(1, self.num_layers):
        crnt_W = 'W' + str(i)
        crnt_b = 'b' + str(i)
        
        # forward pass
        if not self.use_batchnorm: #if no batch_norm
            crnt_input, affine_relu_cache[i] = affine_relu_forward(crnt_input, self.params[crnt_W], self.params[crnt_b])
        else: #batch_norm
            crnt_gamma = 'gamma' + str(i)
            crnt_beta = 'beta' + str(i)
            crnt_input, affine_bn_relu_cache[i] = affine_bn_relu_forward(crnt_input, self.params[crnt_W], self.params[crnt_b], self.params[crnt_gamma], self.params[crnt_beta], self.bn_params[i-1])
        
        # dropout
        if self.use_dropout:
            crnt_input, dropout_cache[i] = dropout_forward(crnt_input, self.dropout_param)
     
    # last affine layer without ReLU
    crnt_W = 'W' + str(self.num_layers)
    crnt_b = 'b' + str(self.num_layers)
    affine_out, affine_cache = affine_forward(crnt_input, self.params[crnt_W], self.params[crnt_b])
    scores = affine_out
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # loss of softmax layer
    loss, dscores = softmax_loss(scores, y)
    
    # loss and gradient of last affine layer
    affine_dx, affine_dw, affine_db = affine_backward(dscores, affine_cache)
    grads['W'+str(self.num_layers)] = affine_dw + self.reg * self.params['W'+str(self.num_layers)]
    grads['b'+str(self.num_layers)] = affine_db
    crnt_W_self_mult = np.multiply(self.params['W'+str(self.num_layers)], self.params['W'+str(self.num_layers)])
    loss += 0.5 * self.reg * np.sum(crnt_W_self_mult)
    
    # loss and gradient of other layers
    for i in range(self.num_layers-1, 0, -1):
        if self.use_dropout:
            affine_dx = dropout_backward(affine_dx, dropout_cache[i])
        
        if not self.use_batchnorm:
            affine_dx, affine_dw, affine_db = affine_relu_backward(affine_dx, affine_relu_cache[i])
        else:
            affine_dx, affine_dw, affine_db, dgamma, dbeta = affine_bn_relu_backward(affine_dx, affine_bn_relu_cache[i])
            grads['beta'+str(i)] = dbeta
            grads['gamma'+str(i)] = dgamma
        crnt_W = 'W' + str(i)
        crnt_b = 'b' + str(i)
        loss += 0.5 * self.reg * np.sum(np.multiply(self.params[crnt_W], self.params[crnt_W]))
        grads[crnt_W] = affine_dw + self.reg * self.params[crnt_W]
        grads[crnt_b] = affine_db
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  affine_out, fc_cache = affine_forward(x, w, b)
  bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
  relu_out, relu_cache = relu_forward(bn_out)
  cache = (fc_cache, bn_cache, relu_cache)
  return relu_out, cache

def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  drelu_out = relu_backward(dout, relu_cache)
  dbn_out, dgamma, dbeta = batchnorm_backward(drelu_out, bn_cache)
  dx, dw, db = affine_backward(dbn_out, fc_cache)
  return dx, dw, db, dgamma, dbeta
