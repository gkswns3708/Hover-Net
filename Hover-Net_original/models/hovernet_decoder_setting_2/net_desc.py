# net_desc = Network Description 
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape
import os
####
class HoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast': # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list)) # -> 7 x 7 convolution 
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1) # -> Residual Unit * 3
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2) # -> Residual Unit * 4
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2) # -> Residual Unit * 6
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2) # -> Residual Unit * 3

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    # def forward(self, imgs):
    #     # print(type(imgs), f"- {os.path.relpath(__file__)}_type(imgs)")
        
    #     assert torch.max(imgs) >= 1.0, "Image should be in range 0-255"
    #     imgs = imgs / 255.0

    #     print(imgs.shape, "- input image shape", flush=True)
    #     if self.training:
    #         d0 = self.conv0(imgs)
    #         print(d0.shape, "- after 7 x 7 convolution", flush=True)
    #         d0 = self.d0(d0, self.freeze) # TODO: 왜 첫 부분에만 self.freeze를 넣는지 알아내기
    #         print(d0.shape, "- after Residual Unit x 3", flush=True)
    #         with torch.set_grad_enabled(not self.freeze):
    #             d1 = self.d1(d0)
    #             print(d1.shape, "- after Residual Unit x 4", flush=True)
    #             d2 = self.d2(d1)
    #             print(d2.shape, "- after Residual Unit x 6", flush=True)
    #             d3 = self.d3(d2)
    #             print(d3.shape, "- after Residual Unit x 3", flush=True)
    #         d3 = self.conv_bot(d3)
    #         print(d3.shape, "- after 1 x 1 convolution")
    #         d = [d0, d1, d2, d3]
    #     else:
    #         d0 = self.conv0(imgs)
    #         d0 = self.d0(d0)
    #         d1 = self.d1(d0)
    #         d2 = self.d2(d1)
    #         d3 = self.d3(d2)
    #         d3 = self.conv_bot(d3)
    #         d = [d0, d1, d2, d3]

    #     # TODO: switch to `crop_to_shape` ?
    #     if self.mode == 'original':
    #         d[0] = crop_op(d[0], [184, 184])
    #         d[1] = crop_op(d[1], [72, 72])
    #         print(d[0].shape, " - d[0], after crop_op([184, 184])", flush=True)
    #         print(d[1].shape, " - d[1], after crop_op([72, 72])", flush=True)
    #     else:
    #         d[0] = crop_op(d[0], [92, 92])
    #         d[1] = crop_op(d[1], [36, 36])
    #         print(d[0].shape, " - d[0], after crop_op([92, 92])", flush=True)
    #         print(d[1].shape, " - d[1], after crop_op([36, 36])", flush=True)
    #     out_dict = OrderedDict()
        
    #     for branch_name, branch_desc in self.decoder.items():
    #         print(branch_name, "- branch_name")
    #         print(d[-1].shape, "- upsampling input shape", flush=True)
    #         u3 = self.upsample2x(d[-1]) + d[-2]  # TODO: 왜 d[-2] 하는지 알아내기
    #         print(u3.shape, "- after upsampling and adding d[-2]", flush=True) 
            
    #         u3 = branch_desc[0](u3)
    #         print(u3.shape, "- after 5x5 Conv + Dense Decode Unit x 8 + 1x1 Conv", flush=True)

    #         u2 = self.upsample2x(u3) + d[-3]
    #         print(u2.shape, "- after upsampling and adding d[-3]", flush=True)
    #         u2 = branch_desc[1](u2)
    #         print(u2.shape, "- after 5x5 Conv + Dense Decode Unit x 4 + 1x1 Conv", flush=True)

    #         u1 = self.upsample2x(u2) + d[-4]
    #         print(u1.shape, "- after upsampling and adding d[-4]", flush=True)
    #         u1 = branch_desc[2](u1)
    #         print(u1.shape, "- after 5x5 Conv + 1x1 Conv", flush=True)

    #         u0 = branch_desc[3](u1)
    #         print(u0.shape, "- after 1x1 Conv", flush=True)
    #         out_dict[branch_name] = u0

    #     return out_dict

    def forward(self, imgs):
        # print(type(imgs), f"- {os.path.relpath(__file__)}_type(imgs)")
        
        assert torch.max(imgs) >= 1.0, "Image should be in range 0-255"
        imgs = imgs / 255.0

        if self.training:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        # TODO: switch to `crop_to_shape` ?
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict
####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoVerNet(mode=mode, **kwargs)

