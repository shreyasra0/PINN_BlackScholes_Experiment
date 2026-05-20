# neural_net/layers.py
import numpy as np

class DenseLayer:
    def __init__(self, in_dim, out_dim):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros((out_dim,))
        
        self.x_input = None
        self.z_output = None