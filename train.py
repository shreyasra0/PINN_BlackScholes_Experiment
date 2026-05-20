import numpy as np
import jax
import jax.numpy as jnp
from neural_net.model import SequentialPINN, jax_forward
from neural_net.losses import compute_pinn_loss
from pde_solver.physics_engine import black_scholes_pde_operator

def get_realtime_adjusted_sigma(S, t, K, base_vol=0.18, skew=-0.12, smile=0.25):
    eps = 1e-2
    S_safe = jnp.clip(S, eps, 10.0)
    t_safe = jnp.clip(t, eps, 10.0)
    K_safe = jnp.clip(K, eps, 10.0)
    
    moneyness = S_safe / K_safe
    log_moneyness = jnp.log(moneyness)
    time_factor = jnp.sqrt(t_safe)
    
    local_vol = base_vol + (skew * log_moneyness / time_factor) + (smile * (log_moneyness ** 2) / t_safe)
    return jnp.clip(local_vol, 0.05, 0.80)

def adam_update(step, params, grads, m, v, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    next_m = jax.tree_util.tree_map(lambda md, gd: b1 * md + (1.0 - b1) * gd, m, grads)
    next_v = jax.tree_util.tree_map(lambda vd, gd: b2 * vd + (1.0 - b2) * (gd ** 2), v, grads)
    
    m_hat = jax.tree_util.tree_map(lambda md: md / (1.0 - b1 ** step), next_m)
    v_hat = jax.tree_util.tree_map(lambda vd: vd / (1.0 - b2 ** step), next_v)
    
    next_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
    )
    return next_params, next_m, next_v

def train_model(epochs=100, lr=0.001, batch_size=1024):
    X_mmap = np.load('data/bs_X_processed.npy', mmap_mode='r')
    y_mmap = np.load('data/bs_y_processed.npy', mmap_mode='r')
    
    total_samples = X_mmap.shape[0]
    split_idx = int(total_samples * 0.8)
    num_batches = split_idx // batch_size
    
    r = 0.05
    model = SequentialPINN() # Keep your original robust layer size
    jax_params = model.get_jax_params()
    
    m_tree = jax.tree_util.tree_map(jnp.zeros_like, jax_params)
    v_tree = jax.tree_util.tree_map(jnp.zeros_like, jax_params)
    
    # Tiny 1024 sample calibration step
    X_init = jnp.array(X_mmap[:1024])
    y_init = jnp.array(y_mmap[:1024])
    S_t, t_t, K_t = X_init[:, 0], X_init[:, 1], X_init[:, 2]
    sig_init = get_realtime_adjusted_sigma(S_t, t_t, K_t)
    
    raw_data_loss = jnp.mean((jax_forward(jax_params, X_init) - y_init) ** 2)
    raw_phys_loss = jnp.mean(black_scholes_pde_operator(jax_forward, jax_params, S_t, t_t, K_t, sig_init, r) ** 2)
    
    lambda_physics = float(raw_data_loss / (raw_phys_loss + 1e-8))
    lambda_boundary = 2.0425 
    
    loss_grad_fn = jax.value_and_grad(compute_pinn_loss, argnums=1)
    
    # Step updates exactly one isolated batch
    @jax.jit
    def step_fn(carry, batch_data):
        params, m, v, step_idx = carry
        x_b, y_b = batch_data
        
        sig_b = get_realtime_adjusted_sigma(x_b[:, 0], x_b[:, 1], x_b[:, 2])
        loss_val, jax_grads = loss_grad_fn(
            jax_forward, params, x_b, y_b, sig_b, lambda_physics, lambda_boundary, r
        )
        
        next_params, next_m, next_v = adam_update(step_idx, params, jax_grads, m, v, lr=lr)
        return (next_params, next_m, next_v, step_idx + 1), loss_val

    # Lightweight validation slice
    X_val_jax = jnp.array(X_mmap[split_idx:split_idx + 2000])
    y_val_jax = jnp.array(y_mmap[split_idx:split_idx + 2000])
    S_v, t_v, K_v = X_val_jax[:, 0], X_val_jax[:, 1], X_val_jax[:, 2]
    sigma_val = get_realtime_adjusted_sigma(S_v, t_v, K_v)
    
    print("=" * 50)
    print(f"RUNNING STREAMING PINN PIPELINE (Batch Size: {batch_size})")
    print("=" * 50)
    
    global_step = 1
    batch_indices = np.arange(num_batches)
    
    for epoch in range(1, epochs + 1):
        # Shuffle only the order of the batch blocks, not individual rows
        np.random.shuffle(batch_indices)
        epoch_losses = []
        
        carry = (jax_params, m_tree, v_tree, global_step)
        
        for b_idx in batch_indices:
            start_i = b_idx * batch_size
            end_i = start_i + batch_size
            
            # Read exactly 1024 rows from disk directly into JAX
            X_batch = jnp.array(X_mmap[start_i:end_i])
            y_batch = jnp.array(y_mmap[start_i:end_i])
            
            carry, loss_val = step_fn(carry, (X_batch, y_batch))
            epoch_losses.append(float(loss_val))
            
        jax_params, m_tree, v_tree, global_step = carry
        
        if epoch == 1 or epoch % 10 == 0:
            avg_train_loss = np.mean(epoch_losses)
            val_data_loss = jnp.mean((jax_forward(jax_params, X_val_jax) - y_val_jax) ** 2)
            val_pde = jnp.mean(black_scholes_pde_operator(jax_forward, jax_params, S_v, t_v, K_v, sigma_val, r) ** 2)
            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Data MSE: {val_data_loss:.6f} | Val PDE MSE: {val_pde:.6f}")

    model.update_from_jax(jax_params)
    print("=" * 50)
    print("TRAINING COMPLETE.")
    print("=" * 50)

if __name__ == "__main__":
    train_model(epochs=100, lr=0.001, batch_size=1024)