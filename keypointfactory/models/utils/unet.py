import torch

from .blocks import get_module


class Unet(torch.nn.Module):

    def __init__(self, in_features, conf):
        super().__init__()

        self.up = [int(u) for u in conf.arch.up]
        self.down = [int(d) for d in conf.arch.down]
        self.in_features = in_features

        size = conf.arch.kernel_size

        if conf.arch.norm == "NoOp":
            self.first_norm = get_module("InstanceNorm2d")(in_features)
        else:
            self.first_norm = get_module("NoOp")()

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

    def forward(self, input1=None, input2=None):
        assert input1 is not None or input2 is not None

        if input1 is not None:
            features1 = [input1]
            features1[0] = self.first_norm(features1[0])
            for block in self.path_down:
                features1.append(block(features1[-1]))

        if input2 is not None:
            features2 = [input2]
            features2[0] = self.first_norm(features2[0])
            for block in self.path_down:
                features2.append(block(features2[-1]))

        if input1 is not None and input2 is not None:
            f_bot1 = out1 = features1[-1]
            f_bot2 = out2 = features2[-1]
            features_horizontal1 = features1[-2::-1]
            features_horizontal2 = features2[-2::-1]
            for layer, f_hor1, f_hor2 in zip(
                self.path_up,
                features_horizontal1,
                features_horizontal2,
            ):
                f_bot1 = layer(out1, f_hor1, out2)
                f_bot2 = layer(out2, f_hor2, out1)

                out1 = f_bot1
                out2 = f_bot2

            return out1, out2
        elif input1 is not None:
            f_bot = features1[-1]
            features_horizontal = features1[-2::-1]

            for layer, f_hor in zip(self.path_up, features_horizontal):
                f_bot = layer(f_bot, f_hor)

            return f_bot
        else:
            f_bot = features2[-1]
            features_horizontal = features2[-2::-1]

            for layer, f_hor in zip(self.path_up, features_horizontal):
                f_bot = layer(f_bot, f_hor)

            return f_bot
