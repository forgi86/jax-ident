import nonlinear_benchmarks
from sklearn.preprocessing import StandardScaler
from argparse import Namespace
from jaxid.datasets import SubsequenceDataset, NumpyLoader
from jaxid.models import StateUpdateAndOptput, MLP
import jax
from jax import random, numpy as jnp
from jax.lax import scan
import optax
from tqdm import tqdm
from flax.training import orbax_utils
import orbax
from pathlib import Path


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
}
cfg = Namespace(**cfg)

key = random.key(42)

# Load data
train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
sampling_time = train_val.sampling_time
u_train, y_train = train_val   
u_train = u_train.reshape(-1, cfg.nu)
y_train = y_train.reshape(-1, cfg.ny)


# Rescale data
scaler_u = StandardScaler()
u = scaler_u.fit_transform(u_train)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y_train)

# Make Dataset and Dataloaders
train_data = SubsequenceDataset(u, y, subseq_len=cfg.seq_len)  
train_loader = NumpyLoader(train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True)


# Make model
f_xu = MLP([64, 32, cfg.nx])
g_x = MLP([64, 32, cfg.ny])
fg = StateUpdateAndOptput(f_xu, g_x)

# Initialize model parameters
x = jnp.ones((cfg.nx,))
u = jnp.ones((cfg.nu,))
key, subkey = random.split(key)
params = fg.init(subkey, x, u)


# Define loss
def simulate(params, x0, u_seq):
    fg_func = lambda x, u: fg.apply(params, x, u)
    return scan(fg_func, x0, u_seq)

def loss_fn(params, batch_x0, batch_u, batch_y):
    def sequence_mse(x0, u_seq, y_seq):
        _, y_hat = simulate(params, x0, u_seq)
        err = y_seq[cfg.skip:] - y_hat[cfg.skip:]
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
        if  idx % 100 == 0:
                print('Loss step {}: '.format(idx), loss_val)


# Save a checkpoint
ckpt = {'params': params,
        'cfg': vars(cfg),
        'LOSS': jnp.array(LOSS)
        }
        #'scaler_u': scaler_u,
        #'scaler_y': scaler_y} # cannot save these python objects unfortunately...

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save(Path(".").resolve()/ "models" / "model1", ckpt, save_args=save_args)

