# neural_net/optimizers.py
import numpy as np

class AdamOptimizer:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
        self.m_W = [np.zeros_like(layer.W) for layer in self.layers]
        self.v_W = [np.zeros_like(layer.W) for layer in self.layers]
        self.m_b = [np.zeros_like(layer.b) for layer in self.layers]
        self.v_b = [np.zeros_like(layer.b) for layer in self.layers]

    def step(self, grads_W, rands_b):
        self.t += 1
        for i, layer in enumerate(self.layers):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1.0 - self.beta1) * grads_W[i]
            self.v_W[i] = self.beta2 * self.v_W[i] + (1.0 - self.beta2) * (grads_W[i] ** 2)
            
            m_W_hat = self.m_W[i] / (1.0 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1.0 - self.beta2 ** self.t)
            
            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            
            self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * rands_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * (rands_b[i] ** 2)
            
            m_b_hat = self.m_b[i] / (1.0 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1.0 - self.beta2 ** self.t)
            
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)