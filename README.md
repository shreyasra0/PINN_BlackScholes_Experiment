# Physics-Informed Neural Network (PINN) for Option Pricing

This project implements a Physics-Informed Neural Network (PINN) to model option pricing surfaces based on the Black-Scholes PDE. By leveraging both market data and the underlying physics of option pricing, this model achieves high generalization with a compact architecture.

## Architecture
* **Model:** 3-layer Sequential Dense Network (32 neurons per layer).
* **Activation:** Swish activation for smooth gradient flow.
* **Physics Constraint:** Includes a PDE residual loss function to ensure market consistency.
* **Optimization:** Trained via JAX using custom loss balancing.

## Performance
The current model is trained on 50,000 samples and achieves a Validation MSE of **0.029119**. The training process is optimized to ensure low generalization error between the training subset and the full 1.7M dataset.

## Setup & Usage

### Prerequisites
Ensure you have the required environment with JAX and NumPy installed:
```bash
pip install -r requirements.txt