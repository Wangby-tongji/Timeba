# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail

"""Historical neural-network blocks used by the canonical Timeba model."""

from math import gcd

import torch
from mamba_ssm.modules.mamba_simple import Mamba
from torch import nn


class Unet1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet1d, self).__init__()
        self.up = nn.ConvTranspose1d(512, 512, kernel_size=2, stride=2)
        self.bn1 = nn.GroupNorm(gcd(1, out_channels * 2), out_channels * 2)
        self.adptconv = nn.Conv1d(
            in_channels, 512, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.GroupNorm(gcd(1, out_channels), out_channels)
        self.conv1 = nn.Conv1d(
            out_channels * 2, out_channels, kernel_size=3, padding=1
        )

        self.mamba = Mamba(
            d_model=out_channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.bn3 = nn.GroupNorm(gcd(1, out_channels), out_channels)
        self.gelu = nn.GELU()

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.adptconv(skip)
        x = torch.cat((x, skip), dim=1)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.gelu(x)
        identity = x
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = self.bn3(x)
        x += identity
        return self.gelu(x)


class Conv1d(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        kernel_size=3,
        stride=1,
        norm="GN",
        ng=32,
        act=True,
    ):
        super(Conv1d, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.conv = nn.Conv1d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            padding=(int(kernel_size) - 1) // 2,
            stride=stride,
            bias=False,
        )

        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm="GN", ng=32, act=True):
        super(Linear, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.linear = nn.Linear(n_in, n_out, bias=False)

        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class MambaBlock(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        kernel_size=3,
        stride=1,
        norm="GN",
        ng=32,
        act=True,
    ):
        super(MambaBlock, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # Registered but unused in the historical forward path.  Retained for
        # strict checkpoint compatibility.
        self.conv2 = nn.Conv1d(
            n_out,
            n_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.mamba = Mamba(
            d_model=n_out,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.gelu = nn.GELU()

        if norm == "GN":
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        if stride != 1 or n_out != n_in:
            if norm == "GN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(
                        n_in, n_out, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.GroupNorm(gcd(ng, n_out), n_out),
                )
            elif norm == "BN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(
                        n_in, n_out, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm1d(n_out),
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out2 = self.bn1(out)
        out = self.gelu(out2)
        out = out.permute(0, 2, 1)
        out = self.mamba(out)
        out = out.permute(0, 2, 1)
        out = self.bn2(out)
        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.gelu(out)
        return out


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=1):
        super(GroupNorm, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.group_norm(x)
        return self.gelu(x)


class LinearRes2(nn.Module):
    def __init__(self, n_in, n_out):
        super(LinearRes2, self).__init__()
        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)

        if n_in != n_out:
            self.transform = nn.Linear(n_in, n_out, bias=False)
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)

        if self.transform is not None:
            x = self.transform(x)

        out += x
        return out


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm="GN", ng=32):
        super(LinearRes, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == "GN":
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        if n_in != n_out:
            if norm == "GN":
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out),
                )
            elif norm == "BN":
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out),
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out
