import jax.numpy as jnp
from pde_solver.physics_engine import black_scholes_pde_operator

def compute_pinn_loss(forward_fn, params, X_batch, y_batch, sigma, lambda_physics=2.8104, lambda_boundary=1.3695, r=0.0):
    V_pred = forward_fn(params, X_batch)
    data_loss = jnp.mean((V_pred - y_batch) ** 2)
    
    S = X_batch[:, 0]
    t = X_batch[:, 1]
    K = X_batch[:, 2]
    
    pde_residual = black_scholes_pde_operator(forward_fn, params, S, t, K, sigma, r)
    
    # Clamp the PDE residual to prevent explosion
    pde_residual = jnp.clip(pde_residual, -1e3, 1e3)
    physics_loss = jnp.mean(pde_residual ** 2)
    
    X_exp = jnp.stack([S, jnp.zeros_like(t) + 1e-5, K], axis=1)
    V_exp_pred = forward_fn(params, X_exp).squeeze()
    V_exp_true = jnp.maximum(S - K, 0.0)
    loss_expiration = jnp.mean((V_exp_pred - V_exp_true) ** 2)
    
    X_floor = jnp.stack([jnp.zeros_like(S), t, K], axis=1)
    V_floor_pred = forward_fn(params, X_floor).squeeze()
    loss_floor = jnp.mean(V_floor_pred ** 2)
    
    total_loss = data_loss + (lambda_physics * physics_loss) + (lambda_boundary * (loss_expiration + loss_floor))
    
    # Final safety catch: if any NaN leaks through, return a large finite number
    return jnp.where(jnp.isnan(total_loss), 1e6, total_loss)