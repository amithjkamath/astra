# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.nets import UNet
from monai.networks.layers.factories import Act, Norm


__all__ = ["CascadedUNet"]


class CascadedUNet(UNet):
    """
    Cascaded UNet, where two UNets are placed one after the other in "series".
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels_first: Sequence[int],
        channels_second: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            channels_first,
            strides,
            kernel_size,
            up_kernel_size,
            num_res_units,
            act,
            norm,
            dropout,
            bias,
            adn_ordering,
        )

        self.channels_first = channels_first
        self.channels_second = channels_second

        def _create_block(
            inc: int,
            outc: int,
            channels: Sequence[int],
            strides: Sequence[int],
            is_top: bool,
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(
                    c, c, channels[1:], strides[1:], False
                )  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(
                inc, c, s, is_top
            )  # create layer in downsampling path
            up = self._get_up_layer(
                upc, outc, s, is_top
            )  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        # Create the two cascaded components of the UNet.
        self.first = _create_block(
            in_channels, out_channels, self.channels_first, self.strides, True
        )
        self.second = _create_block(
            in_channels + 1, out_channels, self.channels_second, self.strides, True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.first(x)
        z = self.second(torch.cat([y, x], dim=1))
        return z
