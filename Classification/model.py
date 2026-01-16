from collections import OrderedDict
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.densenet import _Transition,_DenseBlock
from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer

class AttentionBlock(nn.Module):
    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                stride=1,
                padding=0,),
            nn.BatchNorm3d(num_features=f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                stride=1,
                padding=0,),
            nn.BatchNorm3d(num_features=f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,),
            nn.BatchNorm3d(num_features=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        g1 = self.W_g(x)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class DenseAttnNet(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        init_features = 64,
        growth_rate= 32,
        block_config = (6, 12, 24, 16),
        bn_size: int = 4,
        act = ("relu", {"inplace": True}),
        norm = "batch",
        dropout_prob= 0.0,
    ) -> None:
        super().__init__()

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels
                self_attention = AttentionBlock(spatial_dims=3,f_g=in_channels,f_int=in_channels,f_l=in_channels)
                self.features.add_module(f"SelfAttention{i + 1}", self_attention)
                
        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(in_channels, out_channels)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers(x)
        return x
