from flax import linen as nn
from jax import numpy as jnp
from typing import Sequence

class MLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x

class StateUpdateAndOptput(nn.Module):
    f_xu: nn.Module
    g_x: nn.Module

    def __call__(self, x, u):
        xu = jnp.concatenate([x, u])
        new_state = self.f_xu(xu)
        output = self.g_x(x)
        return new_state, output