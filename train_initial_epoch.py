# train_initial_epoch.py
import numpy as np
import jax
import jax.numpy as jnp
from neural_net.model import SequentialPINN, jax_forward
from neural_net.losses import compute_pinn_loss
from neural_net.optimizers import AdamOptimizer

def generate_mock_data(num_samples=1000):
    np.random.seed(42)
    S = np.random.uniform(300.0, 500.0, (num_samples, 1))
    t = np.random.uniform(0.01, 1.0, (num_samples, 1))
    K = np.random.uniform(350.0, 450.0, (num_samples, 1))
    X = np.hstack([S, t, K])
    y = np.maximum(S - K, 0.0) + np.random.normal(0, 1.0, (num_samples, 1))
    return jnp.array(X), jnp.array(y)

def run_first_epoch():
    X_train, y_train = generate_mock_data()
    sigma = 0.20
    r = 0.05
    
    model = SequentialPINN()
    dense_layers = [layer for layer in model.layers if hasattr(layer, 'W')]
    optimizer = AdamOptimizer(dense_layers, lr=0.001)
    
    jax_params = model.get_jax_params()
    
    S = X_train[:, 0]
    t = X_train[:, 1]
    K = X_train[:, 2]
    
    from physics.engine import black_scholes_pde_operator
    V_pred = jax_forward(jax_params, X_train)
    
    raw_data_loss = jnp.mean((V_pred - y_train) ** 2)
    
    pde_residual = black_scholes_pde_operator(jax_forward, jax_params, S, t, K, sigma, r)
    raw_physics_loss = jnp.mean(pde_residual ** 2)
    
    X_exp = jnp.stack([S, jnp.zeros_like(t), K], axis=1)
    V_exp_pred = jax_forward(jax_params, X_exp).squeeze()
    V_exp_true = jnp.maximum(S - K, 0.0)
    raw_expiration_loss = jnp.mean((V_exp_pred - V_exp_true) ** 2)
    
    X_floor = jnp.stack([jnp.zeros_like(S), t, K], axis=1)
    V_floor_pred = jax_forward(jax_params, X_floor).squeeze()
    raw_floor_loss = jnp.mean(V_floor_pred ** 2)
    
    print("--- RAW UNWEIGHTED LOSS MAGNITUDES (STEP 0) ---")
    print(f"Data Loss (MSE):         {raw_data_loss:.6f}")
    print(f"Physics PDE Loss (MSE):  {raw_physics_loss:.6f}")
    print(f"Expiration Loss (MSE):   {raw_expiration_loss:.6f}")
    print(f"Floor Boundary Loss (MSE):{raw_floor_loss:.6f}")
    print("-----------------------------------------------")
    
    lambda_physics = float(raw_data_loss / (raw_physics_loss + 1e-8))
    lambda_boundary = float(raw_data_loss / ((raw_expiration_loss + raw_floor_loss) / 2.0 + 1e-8))
    
    print("\n--- CALCULATED BALANCED LAMBDAS ---")
    print(f"Suggested lambda_physics:  {lambda_physics:.4f}")
    print(f"Suggested lambda_boundary: {lambda_boundary:.4f}\n")
    
    loss_grad_fn = jax.value_and_grad(compute_pinn_loss, argnums=1)
    
    print("--- RUNNING FIRST TRAINING EPOCH STEPS ---")
    loss_val, jax_grads = loss_grad_fn(
        jax_forward, jax_params, X_train, y_train, sigma, lambda_physics, lambda_boundary, r
    )
    
    grads_W = [np.array(layer_grad['W']) for layer_grad in jax_grads]
    grads_b = [np.array(layer_grad['b']) for layer_grad in jax_grads]
    
    optimizer.step(grads_W, grads_b)
    
    model.update_from_jax(jax_params)
    print(f"Step 1 Complete. Weighted Composite Loss: {loss_val:.6f}")

if __name__ == "__main__":
    run_first_epoch()