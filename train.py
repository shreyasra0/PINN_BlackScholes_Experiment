import numpy as np
import jax
import jax.numpy as jnp
import pickle
from neural_net.model import SequentialPINN, jax_forward
from neural_net.losses import compute_pinn_loss

def train_and_validate():
    X_raw = np.load('data/bs_X_processed.npy')
    y_raw = np.load('data/bs_y_processed.npy')
    unique = np.unique(np.hstack([X_raw, y_raw.reshape(-1, 1)]), axis=0)
    subset_size = 50000 
    indices = np.random.choice(unique.shape[0], subset_size, replace=False)
    X_data, y_data = unique[indices, :3], unique[indices, 3]
    scales = np.max(np.abs(X_data), axis=0)
    y_max = max(float(np.max(y_data)), 50.0)
    model = SequentialPINN()
    params = jax.tree_util.tree_map(lambda p: jax.random.normal(jax.random.PRNGKey(42), p.shape) * 0.01, model.get_jax_params())
    lr = 0.05 
    print(f"Starting training on {subset_size} samples...")
    for epoch in range(1, 51): 
        if epoch == 30: lr *= 0.1 
        def loss_fn(p):
            preds = jax_forward(p, jnp.array(X_data/scales))
            return jnp.mean((preds.flatten() - y_data/y_max)**2)
            
        loss, grads = jax.value_and_grad(loss_fn)(params)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.6f}")

    print("Validating on full dataset...")
    X_full = jnp.array(unique[:, :3] / scales)
    y_full = jnp.array(unique[:, 3] / y_max)
    preds = jax_forward(params, X_full)
    mse = jnp.mean((preds.flatten() - y_full)**2)
    print(f"Final Validation MSE: {mse:.6f}")

    with open('model_weights.pkl', 'wb') as f:
        pickle.dump(params, f)
        print("Weights saved to model_weights.pkl")

if __name__ == "__main__":
    train_and_validate()