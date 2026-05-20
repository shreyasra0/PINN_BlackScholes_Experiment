# physics/engine.py
import jax
import jax.numpy as jnp

def black_scholes_pde_operator(forward_fn, params, S, t, K, sigma, r=0.0):
    def price_scalar(S_val, t_val, K_val):
        x = jnp.array([[S_val, t_val, K_val]])
        return forward_fn(params, x)[0, 0]

    df_dS = jax.grad(price_scalar, argnums=0)
    d2f_dS2 = jax.grad(df_dS, argnums=0)
    df_dt = jax.grad(price_scalar, argnums=1)

    delta_vec = jax.vmap(df_dS)(S, t, K)
    gamma_vec = jax.vmap(d2f_dS2)(S, t, K)
    theta_vec = jax.vmap(df_dt)(S, t, K)
    
    X_batch = jnp.stack([S, t, K], axis=1)
    V = forward_fn(params, X_batch)

    pde_residual = theta_vec + 0.5 * (sigma**2) * (S**2) * gamma_vec + r * S * delta_vec - r * V.squeeze()
    
    return pde_residual