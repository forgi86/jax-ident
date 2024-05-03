from argparse import Namespace
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchid.datasets import SubsequenceDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchid.models import MLP, StateSpaceSimulator
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

no_cuda = False

use_cuda = not no_cuda and torch.cuda.is_available()
device = "cuda:0" if use_cuda else "cpu"
torch.set_default_device(device)

# Pytorch-specific possible optimizations (activate no more than one!)
compile = True
trace = False
script = False

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
train_data = SubsequenceDataset(torch.tensor(u), torch.tensor(y), subseq_len=cfg.seq_len)
train_loader = DataLoader(
    train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    generator=torch.Generator(device=device)
)

# Make model
hidden_f = [(hidden, nn.Tanh()) for hidden in cfg.hidden_f]
f_xu = MLP(cfg.nx + cfg.nu, hidden_f + [(cfg.nx, None)])
hidden_g = [(hidden, nn.Tanh()) for hidden in cfg.hidden_g]
g_x = MLP(cfg.nx, hidden_g + [(cfg.ny, None)])
model = StateSpaceSimulator(f_xu, g_x)
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

dummy_input = (
    torch.zeros((cfg.batch_size, cfg.nx)),
    torch.zeros((cfg.batch_size, cfg.seq_len, cfg.ny)),
    #torch.zeros((cfg.seq_len, cfg.batch_size, cfg.ny)),  # time-first version
)

if trace:
    model = torch.jit.trace(model, dummy_input)

if compile:
    import time
    time_start = time.time()
    #model = torch.compile(model)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    model(*dummy_input)
    time_compile = time.time() - time_start
    print(f"Compile time: {time_compile:.2f}")
if script:
    model = torch.jit.script(model)

# Training loop
LOSS = []
batch_x0 = torch.zeros((cfg.batch_size, cfg.nx))
for epoch in range(cfg.epochs):
    for idx, (batch_u, batch_y) in tqdm(enumerate(train_loader)):

        # time-first version 
        # batch_u = batch_u.transpose(0, 1)
        # batch_y = batch_y.transpose(0, 1)

        optimizer.zero_grad()

        batch_y_pred = model(batch_x0, batch_u)
        loss = torch.nn.functional.mse_loss(batch_y_pred, batch_y)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print("Loss step {}: ".format(idx), loss)


# Save a checkpoint
ckpt = {
    "params": model.parameters(),
    "cfg": vars(cfg),
    "LOSS": np.array(LOSS),
    "scaler_u": scaler_u,
    "scaler_y": scaler_y,
}  # scalers are not correctly saved unfortunately...
