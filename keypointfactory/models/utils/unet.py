import torch
from .blocks import get_module


class Unet(torch.nn.Module):

    def __init__(self, in_features, conf):
        super().__init__()

        self.up = [int(u) for u in conf.arch.up]
        self.down = [int(d) for d in conf.arch.down]
        self.in_features = in_features
        
        size = conf.arch.kernel_size

        down_block = get_module(conf.arch.down_block)
        up_block = get_module(conf.arch.up_block)

        down_dims = [in_features] + self.down
        self.path_down = torch.nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(down_dims[:-1], down_dims[1:])):
            block = down_block(
                d_in, d_out, size=size, name=f"down_{i}", is_first=i == 0, conf=conf
            )
            self.path_down.append(block)

        bottom_dims = [self.down[-1]] + self.up
        horizontal_dims = down_dims[-2::-1]
        self.path_up = torch.nn.ModuleList()
        for i, (d_bot, d_hor, d_out) in enumerate(
            zip(bottom_dims, horizontal_dims, self.up)
        ):
            block = up_block(d_bot, d_hor, d_out, size=size, name=f"up_{i}", conf=conf)
            self.path_up.append(block)

        self.n_params = 0
        for params in self.parameters():
            self.n_params += params.numel()

    def forward(self, input):
        features = [input]
        for block in self.path_down:
            features.append(block(features[-1]))

        f_bot = features[-1]
        features_horizontal = features[-2::-1]
        for layer, f_hor in zip(self.path_up, features_horizontal):
            f_bot = layer(f_bot, f_hor)

        return f_bot
