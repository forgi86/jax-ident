from flax import linen as nn
from typing import Sequence, Any, Dict


class MLP(nn.Module):
    features: Sequence[int]
    layer_kwargs: Dict[str, Any] = None

    def setup(self):
        layer_kwargs = self.layer_kwargs if self.layer_kwargs is not None else {}
        self.layers = [nn.Dense(feat, **layer_kwargs) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.tanh(x)
        return x

