import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class CifarNet(object):
    # initialization function
  def __init__(self, input_dim=(3, 32, 32), 
                 filter_dimension= [(5,5,3,64), (5,5,64,64)],
                 fc_dimension = [384, 192],
                 num_classes=10, 
                 weight_scale=1e-3, 
                 reg=0.0,
                 dtype=np.float32):
    
    # parameter initialization
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C, H, W = input_dim
    
    #print (filter_dimension[0][2])
    
    self.params['CONV1_W'] = np.random.normal(0, weight_scale, filter_dimension[0])
    self.params['CONV1_b'] = np.zeros(filter_dimension[0][3])
    self.params['CONV2_W'] = np.random.normal(0, weight_scale, filter_dimension[1])
    self.params['CONV2_b'] = np.zeros(filter_dimension[1][3])
    self.params['FC3_W'] = np.random.normal(0, weight_scale, (filter_dimension[1][3]*H/4*W/4, fc_dimension[0]))
    self.params['FC3_b'] = np.zeros(fc_dimension[0])
    self.params['FC4_W'] = np.random.normal(0, weight_scale, (fc_dimension[0], fc_dimension[1]))
    self.params['FC4_b'] = np.zeros(fc_dimension[1])
    self.params['FC5_W'] = np.random.normal(0, weight_scale, (fc_dimension[1], num_classes))
    self.params['FC5_b'] = np.zeros(num_classes)
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
  def loss(self, X, y=None):
    CONV1_W, CONV1_b = self.params['CONV1_W'], self.params['CONV1_b']
    CONV2_W, CONV2_b = self.params['CONV2_W'], self.params['CONV2_b']
    FC3_W, FC3_b = self.params['FC3_W'], self.params['FC3_b']
    FC4_W, FC4_b = self.params['FC4_W'], self.params['FC4_b']
    FC5_W, FC5_b = self.params['FC5_W'], self.params['FC5_b']
    
    filter_size = 
     
pass