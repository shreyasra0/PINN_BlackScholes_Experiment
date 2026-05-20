import jax
import jax.numpy as jnp

def black_scholes_pde_operator(forward_fn, params, S, t, K, sigma, r=0.0):
    def batch_forward(S_arr, t_arr, K_arr):
        S_arr = jnp.atleast_1d(S_arr)
        t_arr = jnp.atleast_1d(t_arr)
        K_arr = jnp.atleast_1d(K_arr)
        X = jnp.stack([S_arr, t_arr, K_arr], axis=1)
        return forward_fn(params, X).flatten()

    grad_fn = jax.jacfwd(batch_forward, argnums=(0, 1))
    delta_vec, theta_vec = grad_fn(S, t, K)
    
    def scalar_forward(s, tt, kk):
        # Ensure input is 1D for the forward function
        s, tt, kk = jnp.atleast_1d(s), jnp.atleast_1d(tt), jnp.atleast_1d(kk)
        return batch_forward(s, tt, kk)[0]

    gamma_vec = jax.vmap(
        jax.grad(
            jax.grad(scalar_forward, argnums=0),
            argnums=0
        )
    )(S, t, K)

    V = batch_forward(S, t, K)
    
    pde_residual = theta_vec + 0.5 * (sigma**2) * (S**2) * gamma_vec + r * S * delta_vec - r * V
    
    return pde_residual