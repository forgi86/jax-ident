"""
MIT License

Copyright (c) 2023 Nicolas Zucchet, Robert Meier, Simon Schug.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# The LRU implementation borrows heavily from the MIT-licensed repository https://github.com/NicolasZucchet/minimal-LRU by Nicolas Zucchet, Robert Meier, Simon Schug.
# The original license is included above.


from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn

parallel_scan = jax.lax.associative_scan


# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


def nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(key, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class LRU(nn.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    d_model: int  # input and output dimensions
    d_state: int  # hidden state dimension
    r_min: float = 0.0  # smallest lambda norm
    r_max: float = 1.0  # largest lambda norm
    max_phase: float = 6.28  # max phase lambda

    def setup(self):
        self.theta_log = self.param(
            "theta_log", partial(theta_init, max_phase=self.max_phase), (self.d_state,)
        )
        self.nu_log = self.param(
            "nu_log",
            partial(nu_init, r_min=self.r_min, r_max=self.r_max),
            (self.d_state,),
        )
        self.gamma_log = self.param(
            "gamma_log", gamma_log_init, (self.nu_log, self.theta_log)
        )

        # Glorot initialized Input/Output projection matrices
        self.B_re = self.param(
            "B_re",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_state, self.d_model),
        )
        self.B_im = self.param(
            "B_im",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_state, self.d_model),
        )
        self.C_re = self.param(
            "C_re",
            partial(matrix_init, normalization=jnp.sqrt(self.d_state)),
            (self.d_model, self.d_state),
        )
        self.C_im = self.param(
            "C_im",
            partial(matrix_init, normalization=jnp.sqrt(self.d_state)),
            (self.d_model, self.d_state),
        )
        self.D = self.param("D", matrix_init, (self.d_model,))

    def __call__(self, inputs, state=None):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        C = self.C_re + 1j * self.C_im

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(inputs)
        if state is not None:
            Bu_elements = Bu_elements.at[0].set(Bu_elements[0] + diag_lambda * state)
        # Compute hidden states
        _, hidden_states = parallel_scan(
            binary_operator_diag, (Lambda_elements, Bu_elements)
        )
        # Use them to compute the output of the module
        outputs = jax.vmap(lambda h, x: (C @ h).real + self.D * x)(
            hidden_states, inputs
        )

        return outputs


class SequenceLayer(nn.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

    lru: LRU  # lru module
    d_model: int  # model size
    dropout: float = 0.0  # dropout probability
    norm: str = "layer"  # which normalization to use
    training: bool = True  # in training mode (dropout in training mode only)

    def setup(self):
        """Initializes the ssm, layer norm and dropout"""
        self.seq = self.lru()
        self.out1 = nn.Dense(self.d_model)
        self.out2 = nn.Dense(self.d_model)
        match self.norm:
            case "layer":
                self.normalization = nn.LayerNorm()
            case "batch":
                self.normalization = nn.BatchNorm(
                    use_running_average=not self.training, axis_name="batch"
                )
            case _:
                raise ValueError(f"Normalization {self.norm} not recognized")
    
        self.drop = nn.Dropout(
            self.dropout, broadcast_dims=[0], deterministic=not self.training
        )

    def __call__(self, inputs, state=None):
        x = self.normalization(inputs)  # pre normalization
        x = self.seq(x, state)  # call LRU
        x = self.drop(nn.gelu(x))  # gelu here?
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        x = self.drop(x)
        return inputs + x  # skip connection


class DLRU(nn.Module):
    """Encoder containing several SequenceLayer"""

    # Model parameters
    out_channels: int = 1
    n_layers: int = 3

    # LRU parameters
    d_model: int = 5  # input and output dimensions
    d_state: int = 10  # hidden state dimension
    r_min: float = 0.0  # smallest lambda norm
    r_max: float = 1.0  # largest lambda norm
    max_phase: float = 6.28  # max phase lambda
    dropout: float = 0.0
    training: bool = True
    norm: str = "layer"

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        lru = partial(
            LRU,
            d_model=self.d_model,
            d_state=self.d_state,
            r_min=self.r_min,
            r_max=self.r_max,
            max_phase=self.max_phase,
        )
        self.layers = [
            SequenceLayer(
                lru=lru,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                norm=self.norm,
            )
            for _ in range(self.n_layers)
        ]
        self.decoder = nn.Dense(self.out_channels)

    def __call__(self, u, state=None):
        x = self.encoder(u)  # embed input in latent space
        for layer_idx, layer in enumerate(self.layers):
            state_layer = state[layer_idx] if state is not None else None
            x = layer(x, state=state_layer)  # apply each layer
        y = self.decoder(x)  # decode to output space
        return y


# Here we call vmap to parallelize across a batch of input sequences
BatchedDLRU = nn.vmap(
    DLRU,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None},
    split_rngs={"params": False, "dropout": True},
    axis_name="batch",
)
