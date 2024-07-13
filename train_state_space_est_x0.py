from argparse import Namespace
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import jax
from jax import random, numpy as jnp
import optax
import torch
from jaxid.datasets import SubsequenceDataset, NumpyLoader as DataLoader
from jaxid.common import MLP, ChannelSlicer
from jaxid.statespace import StateUpdateMLP, Simulator
import nonlinear_benchmarks


# Configuration
cfg = {
    # Training
    "batch_size": 32,
    "skip": 100,
    "lr": 2e-4,
    "epochs": 20_000,
    # Model
    "nx": 2,
    "ny": 1,
    "nu": 1,
    "hidden_f": [16, 8,],
    "hidden_g": [8,], # unused, output equation known
}
cfg = Namespace(**cfg)


# Random key
key = random.key(42)

# Load data
train_val, test = nonlinear_benchmarks.Cascaded_Tanks()
sampling_time = train_val.sampling_time
u_train, y_train = train_val
u_train = u_train.reshape(-1, cfg.nu)
y_train = y_train.reshape(-1, cfg.ny)
cfg.seq_len = y_train.shape[0]

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


# Make model
f_xu = StateUpdateMLP(cfg.hidden_f + [cfg.nx])
#g_x = MLP(cfg.hidden_g + [cfg.ny])
g_x = ChannelSlicer([0])
model = Simulator(f_xu, g_x)
x0 = jnp.ones(cfg.nx)
u_dummy = jnp.ones((cfg.seq_len, cfg.nu))
y_dummy, params = model.init_with_output(jax.random.key(0), x0, u_dummy)

opt_vars = (params, x0)
def loss_fn(opt_vars, u, y):
    params, x0 = opt_vars
    y_hat = model.apply(params, x0, u)
    err = y_hat - y
    loss = jnp.mean(err**2)
    return loss


# Setup optimizer
optimizer = optax.adam(learning_rate=cfg.lr)
opt_state = optimizer.init(opt_vars)
loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

# Training loop
LOSS = []
for epoch in (pbar := tqdm(range(cfg.epochs))):
        loss_val, grads = loss_grad_fn(opt_vars, u, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        opt_vars = optax.apply_updates(opt_vars, updates)
        LOSS.append(loss_val)
        if epoch % 100 == 0:
            pbar.set_postfix_str(f"Loss step {epoch}: {loss_val}")

# Save a checkpoint (using torch utilities)
params, x0 = opt_vars
ckpt = {
    "params": params,
    "x0": x0,
    "cfg": cfg,
    "LOSS": jnp.array(LOSS),
    "scaler_u": scaler_u,
    "scaler_y": scaler_y,
}

torch.save(ckpt, "ss_x0.pt")
