# System Identification with Jax

A code base for system identification with Jax. 

## Features
* Implements training of neural state-space models as described e.g. in https://arxiv.org/abs/2206.12928. Initial state handled either with:
    * ZERO initialization -> see [train_state_space.py](train_state_space.py)
    * FF initialization (aka SUBNET) -> see [train_state_space_subnet.py](train_state_space_subnet.py)
* Borrows from [pytorch-ident](https://github.com/forgi86/pytorch-ident) for data loaders and metrics.
* Much faster than PyTorch implemetation, see experiments in [dev/pytorch_comparison](dev/pytorch_comparison). Reason: PyTorch currently lacks a native ``scan`` operation and/or an effective way to compile custom RNNs.
* Simple interface based on flax transformations. Lower-level implementations in [dev/manual_scan_batch](dev/manual_scan_batch).
* New: experimental implementation of [dynoNet](https://arxiv.org/pdf/2006.02250) with direct back-propagation through the recurrence steps (no specialized formulas/derivation)
* New: experimental implementation of [LRU](https://proceedings.mlr.press/v202/orvieto23a.html). Implementation borrows heavily from https://github.com/NicolasZucchet/minimal-LRU