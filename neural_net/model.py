# neural_net/model.py
import jax.numpy as jnp
from neural_net.layers import DenseLayer, SwishActivation

class SequentialPINN:
    def __init__(self):
        self.layers = [
            DenseLayer(3, 16),
            SwishActivation(),
            DenseLayer(16, 16),
            SwishActivation(),
            DenseLayer(16, 16),
            SwishActivation(),
            DenseLayer(16, 1)
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer.x_input = out
                layer.z_output = out @ layer.W + layer.b
                out = layer.z_output
            else:
                out = layer.forward(out)
        return out

    def get_jax_params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'W'):
                params.append({
                    'W': jnp.array(layer.W),
                    'b': jnp.array(layer.b)
                })
        return params

    def update_from_jax(self, jax_params):
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer.W = jax_params[idx]['W'].copy()
                layer.b = jax_params[idx]['b'].copy()
                idx += 1

def jax_forward(params, x):
    out = x
    num_layers = len(params)
    for i, layer in enumerate(params):
        out = jnp.dot(out, layer['W']) + layer['b']
        if i < num_layers - 1:
            out = out / (1.0 + jnp.exp(-out))
    return out