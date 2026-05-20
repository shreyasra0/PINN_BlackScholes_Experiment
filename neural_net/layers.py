# neural_net/layers.py
import numpy as np
class DenseLayer:
    def __init__(self, in_dim, out_dim):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros((out_dim,))
        
        self.x_input = None
        self.z_output = None
class SwishActivation:
    def forward(self, x):
        return x / (1.0 + np.exp(-x))
        
    def backward(self, x, upstream_grad):
        sig = 1.0 / (1.0 + np.exp(-x))
        swish_deriv = sig * (1.0 + x * (1.0 - sig))
        return upstream_grad * swish_deriv