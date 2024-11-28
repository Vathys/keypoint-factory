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
        if is_first:
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


class ResDownBlock(nn.Module):
    def __init__(self, in_, out_, size, name=None, is_first=False, conf=None):
        super(ResDownBlock, self).__init__()

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
        sequence_list.append(get_module(conf.arch.norm)(out_))
        sequence_list.append(get_module(conf.arch.gate)(out_))
        sequence_list.append(
            nn.Conv2d(out_, out_, size, padding=padding_size, bias=conf.arch.bias)
        )
        self.sequence = nn.Sequential(*sequence_list)

        if in_ == out_:
            self.shortcut = NoOp()
        else:
            self.shortcut = nn.Conv2d(in_, out_, 1, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        return self.sequence(x) + self.shortcut(x)


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

    def forward(
        self, bottom, horizontal, attn_bottom=None
    ):  # attn_bottom is not used; included for compatibility
        up_bot = self.upsample(bottom)
        horizontal = cut_to_match(up_bot, horizontal, n_pref=2)
        combined = torch.cat([up_bot, horizontal], dim=1)

        return self.sequence(combined)


class ResUpBlock(nn.Module):
    def __init__(self, bottom_, horizontal_, out_, size, name=None, conf=None):
        super(ResUpBlock, self).__init__()

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

        if cat_ == out_:
            self.shortcut = NoOp()
        else:
            self.shortcut = nn.Conv2d(cat_, out_, 1, bias=False)

    def forward(
        self, bottom, horizontal, attn_bottom=None
    ):  # attn_bottom is not used; included for compatibility
        up_bot = self.upsample(bottom)
        horizontal = cut_to_match(up_bot, horizontal, n_pref=2)
        combined = torch.cat([up_bot, horizontal], dim=1)

        return self.sequence(combined) + self.shortcut(combined)


class ThinAttentionUpBlock(torch.nn.Module):
    def __init__(self, bottom_, horizontal_, out_, size, name=None, conf=None):
        super(ThinAttentionUpBlock, self).__init__()

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

        self.shortcut = AttentionBlock(bottom_, horizontal_, horizontal_, conf=conf)

    def forward(self, bottom, horizontal, attn_bottom=None):
        up_bot = self.upsample(bottom)
        attn_up_bot = self.upsample(attn_bottom) if attn_bottom is not None else None

        shortcut = (
            self.shortcut(attn_up_bot, horizontal)
            if attn_bottom is not None
            else self.shortcut(up_bot, horizontal)
        )
        horizontal = cut_to_match(up_bot, shortcut, n_pref=2)
        combined = torch.cat([up_bot, shortcut], dim=1)

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

    def forward(
        self, bottom, horizontal, attn_bottom=None
    ):  # attn_bottom is not used; included for compatibility
        up_bot = self.upsample(bottom)
        horizontal = cut_to_match(up_bot, horizontal, n_pref=2)
        combined = torch.cat([up_bot, horizontal], dim=1)

        return self.sequence(combined)


class AttentionUpBlock(nn.Module):
    def __init__(self, bottom_, horizontal_, out_, size, name=None, conf=None):
        super(AttentionUpBlock, self).__init__()

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

        self.shortcut = AttentionBlock(bottom_, horizontal_, horizontal_, conf=conf)

    def forward(self, bottom, horizontal, attn_bottom=None):
        up_bot = self.upsample(bottom)
        attn_up_bot = self.upsample(attn_bottom) if attn_bottom is not None else None

        shortcut = (
            self.shortcut(attn_up_bot, horizontal)
            if attn_bottom is not None
            else self.shortcut(up_bot, horizontal)
        )
        horizontal = cut_to_match(up_bot, shortcut, n_pref=2)
        combined = torch.cat([up_bot, shortcut], dim=1)

        return self.sequence(combined)


class AttentionBlock(torch.nn.Module):
    def __init__(self, F_g, F_l, F_int, conf=None):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            get_module(conf.arch.norm)(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            get_module(conf.arch.norm)(F_int),
        )
        self.W_psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            get_module(conf.arch.norm)(1),
            nn.Sigmoid(),
        )

        self.gate = get_module(conf.arch.gate)(F_int)

        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.gate(g1 + x1)
        psi = self.W_psi(psi)

        return x * psi
