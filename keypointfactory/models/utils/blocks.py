import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.misc import cut_to_match, size_is_pow2


def get_module(classname):
    import sys

    mod = sys.modules[__name__]
    cls = getattr(mod, classname, None)

    if cls is None:
        cls = getattr(nn, classname, None)
    if cls is None:
        raise AttributeError(f"Class {classname} not found...")
    else:
        return cls


class NoOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class TrivialUpsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialUpsample, self).__init__()

    def forward(self, x):
        r = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return r


class TrivialDownsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrivialDownsample, self).__init__()

    def forward(self, x):
        if not size_is_pow2(x):
            msg = f"Trying to downsample feature map of size {x.size()}"
            raise RuntimeError(msg)

        return F.avg_pool2d(x, 2)


class Downsample(nn.Module):
    def __init__(self, n_features, size, conf=None):
        super(Downsample, self).__init__()

        self.sequence = nn.Sequential(
            get_module(conf.arch.norm)(n_features),
            get_module(conf.arch.gate)(n_features),
            nn.Conv2d(
                n_features,
                n_features,
                size,
                stride=2,
                padding=size // 2,
                bias=conf.arch.bias,
            ),
        )

    def forward(self, x):
        return self.sequence(x)


class Upsample(nn.Module):
    def __init__(self, n_features, size, conf=None):
        super(Upsample, self).__init__()

        self.sequence = nn.Sequential(
            get_module(conf.arch.norm)(n_features),
            get_module(conf.arch.gate)(n_features),
            nn.ConvTranspose2d(
                n_features,
                n_features,
                size,
                stride=2,
                padding=2,
                output_padding=1,
            ),
        )

    def forward(self, x):
        return self.sequence(x)


class DownBlock(nn.Module):
    def __init__(self, in_, out_, size, name=None, is_first=False, conf=None):
        super(DownBlock, self).__init__()

        self.name = name
        if conf.arch.padding:
            padding_size = size // 2
        else:
            padding_size = 0

        sequence_list = []
        if not is_first:
            self.downsample = NoOp()
        else:
            self.downsample = get_module(conf.arch.downsample)(in_, size, conf=conf)
            sequence_list.append(get_module(conf.arch.norm)(in_))
            sequence_list.append(get_module(conf.arch.gate)(in_))

        sequence_list.append(
            nn.Conv2d(in_, out_, size, padding=padding_size, bias=conf.arch.bias)
        )
        sequence_list.append(get_module(conf.arch.norm)(out_))
        sequence_list.append(get_module(conf.arch.gate)(out_))
        sequence_list.append(
            nn.Conv2d(out_, out_, size, padding=padding_size, bias=conf.arch.bias)
        )
        self.sequence = nn.Sequential(*sequence_list)

    def forward(self, x):
        x = self.downsample(x)
        return self.sequence(x)


class ThinDownBlock(nn.Module):
    def __init__(self, in_, out_, size, name=None, is_first=False, conf=None):
        super(ThinDownBlock, self).__init__()

        self.name = name
        if conf.arch.padding:
            padding_size = size // 2
        else:
            padding_size = 0

        sequence_list = []
        if is_first:
            self.downsample = NoOp()
        else:
            self.downsample = get_module(conf.arch.downsample)(in_, size, conf=conf)
            sequence_list.append(get_module(conf.arch.norm)(in_))
            sequence_list.append(get_module(conf.arch.gate)(in_))

        sequence_list.append(
            nn.Conv2d(in_, out_, size, padding=padding_size, bias=conf.arch.bias)
        )
        self.sequence = nn.Sequential(*sequence_list)

    def forward(self, x):
        x = self.downsample(x)
        return self.sequence(x)


class UpBlock(nn.Module):
    def __init__(self, bottom_, horizontal_, out_, size, name=None, conf=None):
        super(UpBlock, self).__init__()

        self.name = name
        if conf.arch.padding:
            padding_size = size // 2
        else:
            padding_size = 0

        self.upsample = get_module(conf.arch.upsample)(bottom_, size, conf=conf)
        cat_ = bottom_ + horizontal_

        sequence_list = []
        sequence_list.append(get_module(conf.arch.norm)(cat_))
        sequence_list.append(get_module(conf.arch.gate)(cat_))
        sequence_list.append(
            nn.Conv2d(cat_, cat_, size, padding=padding_size, bias=conf.arch.bias)
        )
        sequence_list.append(get_module(conf.arch.norm)(cat_))
        sequence_list.append(get_module(conf.arch.gate)(cat_))
        sequence_list.append(
            nn.Conv2d(cat_, out_, size, padding=padding_size, bias=conf.arch.bias)
        )
        self.sequence = nn.Sequential(*sequence_list)

    def forward(self, bottom, horizontal):
        up_bot = self.upsample(bottom)
        horizontal = cut_to_match(up_bot, horizontal, n_pred=2)
        combined = torch.cat([up_bot, horizontal], dim=1)

        return self.sequence(combined)


class ThinUpBlock(torch.nn.Module):
    def __init__(self, bottom_, horizontal_, out_, size, name=None, conf=None):
        super(ThinUpBlock, self).__init__()

        self.name = name
        if conf.arch.padding:
            padding_size = size // 2
        else:
            padding_size = 0

        self.upsample = get_module(conf.arch.upsample)(bottom_, size, conf=conf)
        cat_ = bottom_ + horizontal_

        sequence_list = []
        sequence_list.append(get_module(conf.arch.norm)(cat_))
        sequence_list.append(get_module(conf.arch.gate)(cat_))
        sequence_list.append(
            nn.Conv2d(cat_, out_, size, padding=padding_size, bias=conf.arch.bias)
        )
        self.sequence = nn.Sequential(*sequence_list)

    def forward(self, bottom, horizontal):
        up_bot = self.upsample(bottom)
        horizontal = cut_to_match(up_bot, horizontal, n_pref=2)
        combined = torch.cat([up_bot, horizontal], dim=1)

        return self.sequence(combined)
