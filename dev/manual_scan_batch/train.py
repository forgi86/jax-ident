from argparse import Namespace
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import jax
from jax import random, numpy as jnp
import optax
from flax.training import orbax_utils
import orbax
from jaxid.datasets import SubsequenceDataset, NumpyLoader as DataLoader
from jaxid.statespace import StateUpdateAndOptput, MLP
import nonlinear_benchmarks


# Configuration
cfg = {
    # Training
    "batch_size": 32,
    "seq_len": 1000,
    "skip": 100,
    "lr": 2e-4,
    "epochs": 10,
    # Model
    "nx": 10,
    "ny": 1,
    "nu": 1,
    "hidden_f": [32, 16],
    "hidden_g": [32, 16],
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
train_data = SubsequenceDataset(u, y, subseq_len=cfg.seq_len)
train_loader = DataLoader(
    train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True
)


# Make model
f_xu = MLP(cfg.hidden_f + [cfg.nx])
g_x = MLP(cfg.hidden_g + [cfg.ny])
fg = StateUpdateAndOptput(f_xu, g_x)

# Initialize model parameters
x = jnp.ones((cfg.nx,))
u = jnp.ones((cfg.nu,))
key, subkey = random.split(key)
params = fg.init(subkey, x, u)


# Define loss
def simulate(params, x0, u_seq):
    fg_func = lambda x, u: fg.apply(params, x, u)
    return jax.lax.scan(fg_func, x0, u_seq)


def loss_fn(params, batch_x0, batch_u, batch_y):
    def sequence_mse(x0, u_seq, y_seq):
        _, y_hat = simulate(params, x0, u_seq)
        err = y_seq[cfg.skip :] - y_hat[cfg.skip :]
        loss = jnp.mean(err**2)
        return loss

    return jnp.mean(jax.vmap(sequence_mse)(batch_x0, batch_u, batch_y), axis=0)


# Setup optimizer
optimizer = optax.adam(learning_rate=cfg.lr)
opt_state = optimizer.init(params)
loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

# Training loop
LOSS = []
batch_x0 = jnp.zeros((cfg.batch_size, cfg.nx))
for epoch in range(cfg.epochs):
    for idx, (batch_u, batch_y) in tqdm(enumerate(train_loader)):
        loss_val, grads = loss_grad_fn(params, batch_x0, batch_u, batch_y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        LOSS.append(loss_val)
        if idx % 100 == 0:
            print("Loss step {}: ".format(idx), loss_val)


# Save a checkpoint
ckpt = {
    "params": params,
    "cfg": vars(cfg),
    "LOSS": jnp.array(LOSS),
    "scaler_u": scaler_u,
    "scaler_y": scaler_y,
}  # scalers are not correctly saved unfortunately...

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save(
    Path(".").resolve() / "models" / "model1", ckpt, save_args=save_args, force=True
)
