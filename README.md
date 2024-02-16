# System Identification in Jax

An experimental code base for system identification with Jax. 

## Features
* Implements training of state-space models as described e.g. in https://arxiv.org/abs/2206.12928.
* Borrows from [pytorch-ident](https://github.com/forgi86/pytorch-ident) for data loaders and metrics.
* Much faster than PyTorch implemetation, see experiments in [dev/pytorch_comparison](dev/pytorch_comparison). Reason: PyTorch currently lacks a native ``scan`` operation or an effective way to compile custom RNNs.
* Simple interface based on flax transformations. Lower-level implementations in [dev/manual_scan_batch](dev/manual_scan_batch).
