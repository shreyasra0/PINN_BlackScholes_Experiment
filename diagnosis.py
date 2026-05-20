# diagnose_nan.py
import numpy as np
import jax.numpy as jnp
from neural_net.model import SequentialPINN, jax_forward
from pde_solver.physics_engine import black_scholes_pde_operator

def run_diagnostics():
    np.random.seed(42)
    S = np.random.uniform(300.0, 500.0, (10,)).flatten()
    t = np.random.uniform(0.01, 1.0, (10,)).flatten()
    K = np.random.uniform(350.0, 450.0, (10,)).flatten()
    X_train = jnp.stack([S, t, K], axis=1)

    model = SequentialPINN()
    jax_params = model.get_jax_params()
    sigma = 0.20
    r = 0.05

    print("--- 1. INPUT MATRIX SHAPES ---")
    print(f"S shape: {S.shape}, t shape: {t.shape}, K shape: {K.shape}")
    print(f"X_train shape: {X_train.shape}")

    print("\n--- 2. FORWARD PASS SHAPE CHECK ---")
    try:
        V_pred = jax_forward(jax_params, X_train)
        print(f"V_pred shape: {V_pred.shape}")
        print(f"Contains NaN? {jnp.isnan(V_pred).any()}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

    print("\n--- 3. STEP-BY-STEP SCALAR PASS TRAVERSAL ---")
    S_val, t_val, K_val = S[0], t[0], K[0]
    x_single = jnp.array([[S_val, t_val, K_val]])
    print(f"Single row layout shape: {x_single.shape}")
    
    out = x_single
    for i, layer in enumerate(jax_params):
        z = jnp.dot(out, layer['W']) + layer['b']
        print(f"Layer {i} Affine output shape: {z.shape} | Max: {jnp.max(z):.4f} | Min: {jnp.min(z):.4f}")
        if i < len(jax_params) - 1:
            out = z / (1.0 + jnp.exp(-z))
            print(f"Layer {i} Swish output shape: {out.shape} | Max: {jnp.max(out):.4f} | Contains NaN? {jnp.isnan(out).any()}")
        else:
            out = z

    print("\n--- 4. ISOLATING INDIVIDUAL GREEKS ---")
    def price_scalar(S_v, t_v, K_v):
        x = jnp.array([[S_v, t_v, K_v]])
        return jax_forward(jax_params, x)[0, 0]

    import jax
    df_dS = jax.grad(price_scalar, argnums=0)
    d2f_dS2 = jax.grad(df_dS, argnums=0)
    df_dt = jax.grad(price_scalar, argnums=1)

    try:
        delta_sample = df_dS(S_val, t_val, K_val)
        gamma_sample = d2f_dS2(S_val, t_val, K_val)
        theta_sample = df_dt(S_val, t_val, K_val)
        print(f"Sample Delta: {delta_sample} (Is NaN? {jnp.isnan(delta_sample)})")
        print(f"Sample Gamma: {gamma_sample} (Is NaN? {jnp.isnan(gamma_sample)})")
        print(f"Sample Theta: {theta_sample} (Is NaN? {jnp.isnan(theta_sample)})")
    except Exception as e:
        print(f"Individual gradient calculation crashed: {e}")

if __name__ == "__main__":
    run_diagnostics()