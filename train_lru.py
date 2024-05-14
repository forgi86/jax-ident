from argparse import Namespace
import time
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import jax
from jax import random, numpy as jnp
import optax
import torch
import nonlinear_benchmarks
from jaxid.datasets import SubsequenceDataset, NumpyLoader as DataLoader
from jaxid.lru import BatchedDLRU


# Configuration
cfg = {
    # Training
    "batch_size": 32,
    "seq_len": 1000,
    "skip": 100,
    "lr": 2e-4,
    "epochs": 2,
    # Model
    "n_layers": 3,
    "d_model": 5,
    "d_state": 10,
}
cfg = Namespace(**cfg)


time_start = time.time()

# Random key
key = random.key(42)

# Load data
train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
ny = 1
nu = 1
sampling_time = train_val.sampling_time
u_train, y_train = train_val
u_train = u_train.reshape(-1, nu)
y_train = y_train.reshape(-1, ny)


# Rescale data
scaler_u = StandardScaler()
u = scaler_u.fit_transform(u_train).astype(np.float32)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y_train).astype(np.float32)

# Make Dataset and Dataloaders
train_data = SubsequenceDataset(u, y, subseq_len=cfg.seq_len)
train_loader = DataLoader(
    train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True
)

# Make model and initialize parameters
model = BatchedDLRU(n_layers=cfg.n_layers, d_model=cfg.d_model, d_state=cfg.d_state)
_, params = model.init_with_output(
    jax.random.key(0), jnp.ones((cfg.batch_size, cfg.seq_len, nu))
)

# Define loss function
def loss_fn(params, batch_u, batch_y):
    batch_y_hat = model.apply(params, batch_u)
    err = batch_y[:, cfg.skip :] - batch_y_hat[:, cfg.skip :]
    loss = jnp.mean(err**2)
    return loss

# Setup optimizer
optimizer = optax.adam(learning_rate=cfg.lr)
opt_state = optimizer.init(params)
loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

# Training loop
LOSS = []
for epoch in range(cfg.epochs):
    for idx, (batch_u, batch_y) in tqdm(enumerate(train_loader)):
        loss_val, grads = loss_grad_fn(params, batch_u, batch_y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        LOSS.append(loss_val)
        if idx % 100 == 0:
            print("Loss step {}: ".format(idx), loss_val)

train_time = time.time() - time_start
print(f"Training time: {train_time:.2f}")

# Save a checkpoint (using torch utilities)
ckpt = {
    "params": params,
    "cfg": cfg,
    "LOSS": jnp.array(LOSS),
    "scaler_u": scaler_u,
    "scaler_y": scaler_y,
    "train_time": train_time,
}

torch.save(ckpt, "lru.pt")
