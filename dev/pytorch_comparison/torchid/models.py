import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    def __init__(self, input_size, layers_data: list):
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            if activation is not None:
                assert isinstance(
                    activation, nn.Module
                ), "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data


class StateSpaceSimulator(nn.Module):

    def __init__(self, f_xu: nn.Module, g_x: nn.Module):
        super().__init__()
        self.f_xu = f_xu
        self.g_x = g_x


    def simulate_state(self, x_0: torch.Tensor, u: torch.Tensor):
        x: List[torch.Tensor] = []
        x_step = x_0
        dim_time: int = 1

        for u_step in u.split(1, dim=dim_time):  # split along the time axis
            u_step = u_step.squeeze(dim_time)
            x += [x_step]
            xu_step = torch.cat((x_step, u_step), dim=-1)
            dx = self.f_xu(xu_step)
            x_step = x_step + dx

        x = torch.stack(x, dim_time)
        return x

    # def simulate_state(self, x_0: torch.Tensor, u: torch.Tensor):
    #     x_step = x_0
    #     x = torch.empty((u.shape[0], u.shape[1], x_0.shape[1]))
    #     for idx in range(u.shape[0]):
    #         x[idx, :] = x_step
    #         u_step = u[idx, :]
    #         xu_step = torch.cat((x_step, u_step), dim=-1)
    #         dx = self.f_xu(xu_step)
    #         x_step = x_step + dx
    #     return x
    
    def forward(self, x_0, u): 
        x = self.simulate_state(x_0, u)
        y = self.g_x(x)
        return y
    

    # def forward(self, x_0, u):
    #     y: List[torch.Tensor] = []
    #     x_step = x_0
    #     dim_time: int = 0

    #     for u_step in u.split(1, dim=dim_time):  # split along the time axis
    #         y += [self.g_x(x_step)]
    #         u_step = u_step.squeeze(dim_time)
    #         xu_step = torch.cat((x_step, u_step), dim=-1)
    #         dx = self.f_xu(xu_step)
    #         x_step = x_step + dx

    #     y = torch.stack(y, dim_time)
    #     return y

