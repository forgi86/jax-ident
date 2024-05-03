from flax import linen as nn
from jax import numpy as jnp
from .static import MLP


class StateUpdateAndOptput(nn.Module):
    f_xu: nn.Module
    g_x: nn.Module

    def __call__(self, x, u):
        xu = jnp.concatenate([x, u])
        new_state = self.f_xu(xu)
        output = self.g_x(x)
        return new_state, output


class Simulator(nn.Module):
    f_xu: nn.Module
    g_x: nn.Module

    @nn.compact
    def __call__(self, x0, u):
        ScanUpdate = nn.scan(
            StateUpdateAndOptput,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )

        rnn = ScanUpdate(self.f_xu, self.g_x)
        carry = x0
        carry, y = rnn(carry, u)
        return y


BatchedSimulator = nn.vmap(
    Simulator,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None},
    split_rngs={"params": False},
)
