from argparse import Namespace
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import jax
from jax import random, numpy as jnp
import optax
import torch
from jaxid.datasets import SubsequenceDataset, NumpyLoader as DataLoader
from jaxid.common import MLP
from jaxid.statespace import StateUpdateMLP, BatchedSubNet
import nonlinear_benchmarks


# Configuration
cfg = {
    # Training
    "batch_size": 32,
    "seq_len_fit": 1000,
    "seq_len_est": 30,
    "lr": 2e-4,
    "epochs": 2,
    # Model
    "nx": 10,
    "ny": 1,
    "nu": 1,
    "hidden_f": [16],
    "hidden_g": [16],
    "hidden_est": [32, 16],
}
cfg = Namespace(**cfg)

# Random key
key = random.key(42)

# Load data
train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
sampling_time = train_val.sampling_time
u_train, y_train = train_val
u_train = u_train.reshape(-1, cfg.nu)
y_train = y_train.reshape(-1, cfg.ny)


# Rescale data
scaler_u = StandardScaler()
u = scaler_u.fit_transform(u_train).astype(np.float32)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y_train).astype(np.float32)

# Make Dataset and Dataloaders
train_data = SubsequenceDataset(u, y, subseq_len=cfg.seq_len_fit + cfg.seq_len_est)
train_loader = DataLoader(
    train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True
)


# Make model
f_xu = StateUpdateMLP(cfg.hidden_f + [cfg.nx])
g_x = MLP(cfg.hidden_g + [cfg.ny])
estimator = MLP(cfg.hidden_est + [cfg.nx])
batched_subnet = BatchedSubNet(f_xu, g_x, estimator)

# Initialize model
y_est = jnp.ones((cfg.batch_size, cfg.seq_len_est, cfg.ny))
u_est = jnp.ones((cfg.batch_size, cfg.seq_len_est, cfg.nu))
u_fit = jnp.ones((cfg.batch_size, cfg.seq_len_fit, cfg.nu))
_, params = batched_subnet.init_with_output(key, y_est, u_est, u_fit)


# Define loss function
def loss_fn(params, batch_u, batch_y):
    batch_y_est = batch_y[:, : cfg.seq_len_est, :]
    batch_u_est = batch_u[:, : cfg.seq_len_est, :]
    batch_u_fit = batch_u[:, cfg.seq_len_est :, :]
    batch_y_fit = batch_y[:, cfg.seq_len_est :, :]
    batch_y_hat = batched_subnet.apply(params, batch_y_est, batch_u_est, batch_u_fit)
    err = batch_y_fit - batch_y_hat
    loss = jnp.mean(err**2)
    return loss


# Setup optimizer
optimizer = optax.adam(learning_rate=cfg.lr)
opt_state = optimizer.init(params)
loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

# Training loop
LOSS = []
batch_x0 = jnp.zeros((cfg.batch_size, cfg.nx))
for epoch in range(cfg.epochs):
    for idx, (batch_u, batch_y) in tqdm(enumerate(train_loader)):
        loss_val, grads = loss_grad_fn(params, batch_u, batch_y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        LOSS.append(loss_val)
        if idx % 100 == 0:
            print("Loss step {}: ".format(idx), loss_val)


# Save a checkpoint (using torch utilities)
ckpt = {
    "params": params,
    "cfg": cfg,
    "LOSS": jnp.array(LOSS),
    "scaler_u": scaler_u,
    "scaler_y": scaler_y,
}

torch.save(ckpt, "ss_subnet.pt")
