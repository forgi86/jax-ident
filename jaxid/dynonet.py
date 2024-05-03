import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
from .static import MLP

# simulate one step with a function (for later use with scan)
def filter_step(params, carry, u_step):

    b_coeff, a_coeff = params
    u_carry, y_carry = carry
    u_carry = jnp.r_[u_step, u_carry]
    y_new = jnp.dot(b_coeff, u_carry) - jnp.dot(a_coeff, y_carry)

    u_carry = u_carry[:-1]
    y_carry = jnp.r_[y_new,  y_carry][:-1]
    carry = (u_carry, y_carry)
    return carry, y_new


filter_step_simo = jax.vmap(filter_step, in_axes=(0, 0, 0)) # params, carry, u_step
filter_step_mimo = jax.vmap(filter_step_simo, in_axes=(0, 0, None)) # params, carry, u_step

def mimo_filter(params, carry, u):
    _, y_all = jax.lax.scan(lambda carry, u: filter_step_mimo(params, carry, u), carry, u)
    return  y_all.mean(axis=-1)

batched_mimo_filter = jax.vmap(mimo_filter, in_axes=(None, 0, 0))


def fixed_std_initializer(std):
    """
    Returns a Flax initializer that initializes the weights with a fixed standard deviation.
    
    Args:
    variance (float): The desired variance of the weights.

    Returns:
    An initializer function.
    """
    def initializer(key, shape, dtype=jnp.float32):
        # Calculate standard deviation from the desired variance
        # Initialize weights from a normal distribution scaled by the std_dev
        return jax.random.normal(key, shape, dtype) * std
    return initializer


def fixed_uniform_initializer(range):
    """
    Returns a Flax initializer that initializes the weights with a fixed standard deviation.
    
    Args:
    variance (float): The desired variance of the weights.

    Returns:
    An initializer function.
    """
    def initializer(key, shape, dtype=jnp.float32):
        # Calculate standard deviation from the desired variance
        # Initialize weights from a normal distribution scaled by the std_dev
        return jax.random.uniform(key, shape, dtype, -range, range)
    return initializer


class MimoLTI(nn.Module):
    in_channels: int = 1
    out_channels: int = 1
    nb: int = 3
    na: int = 2

    coeff_init: Callable = fixed_uniform_initializer(1e-2)

    @nn.compact
    def __call__(self, inputs):

        b_coeff = self.param(
            "b_coeff",
            self.coeff_init,  # Initialization function
            (self.out_channels, self.in_channels, self.nb),
        )  # shape info.

        a_coeff = self.param(
            "a_coeff",
            self.coeff_init,  # Initialization function
            (self.out_channels, self.in_channels, self.na),
        )  # shape info.
        params = (b_coeff, a_coeff)

        u_carry = jnp.zeros((inputs.shape[0], self.out_channels, self.in_channels, self.nb - 1))
        y_carry = jnp.zeros((inputs.shape[0], self.out_channels, self.in_channels, self.na))
        carry = (u_carry, y_carry)
        y = batched_mimo_filter(params, carry, inputs)
        #y = inputs + 1
        return y


class DynoNet(nn.Module):
    in_channels: int = 1
    out_channels: int = 1
    nb: int = 10
    na: int = 10
    hidden_size: int = 20

    @nn.compact
    def __call__(self, u):
        y = nn.Sequential(
            [
                MimoLTI(self.in_channels, 4, self.nb, self.na),
                MLP([self.hidden_size, 3]),
                MimoLTI(3, self.out_channels, self.nb, self.na),
            ]
        )(u)
        return y + MimoLTI(self.in_channels, self.out_channels, self.nb, self.na)(u)