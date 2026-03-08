from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MultiStageModel(nn.Module):
    def __init__(self, mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv, is_train=True, dropout_prob: float = 0.0):
        self.num_stages = mstcn_stages
        self.num_layers = mstcn_layers
        self.num_f_maps = mstcn_f_maps
        self.dim = mstcn_f_dim
        self.num_classes = out_features
        self.causal_conv = mstcn_causal_conv
        self.is_train = is_train
        print(f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps: {self.num_f_maps}, dim: {self.dim}")
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv,
                                       is_train=is_train,
                                       dropout_prob=dropout_prob)
        self.stages = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.num_classes,
                                       self.num_classes,
                                       causal_conv=self.causal_conv,
                                       is_train=is_train,
                                       dropout_prob=dropout_prob)

        self.smoothing = False

    def forward(self, x):
        """
        If is_train is False (inference), return first-stage features [B, num_f_maps, T]
        so downstream Transformer receives 32-d features, matching the working pipeline.
        If is_train is True (training/classification), return stacked class logits.
        """
        out = self.stage1(x)
        if not self.is_train:
            # Inference path: return temporal features (num_f_maps channels)
            return out

        # Training path: run second stage on class probabilities
        outputs_classes = out.unsqueeze(0)
        out_classes = self.stages(F.softmax(out, dim=1))
        outputs_classes = torch.cat((outputs_classes, out_classes.unsqueeze(0)), dim=0)
        return outputs_classes

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        mstcn_reg_model_specific_args = parser.add_argument_group(title='mstcn reg specific args options')
        mstcn_reg_model_specific_args.add_argument("--mstcn_stages", default=4, type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_layers", default=10, type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_f_maps", default=64, type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_f_dim", default=2048, type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_causal_conv", action='store_true')
        return parser


class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_f_maps: int,
                 dim: int,
                 num_classes: int,
                 causal_conv: bool = False,
                 is_train: bool = True,
                 dropout_prob: float = 0.0):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.is_train = is_train
        self.layers = nn.ModuleList([
            copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, causal_conv=causal_conv, dropout_prob=dropout_prob))
            for i in range(num_layers)
        ])
        if self.is_train:
            self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        if self.is_train:
            out = self.conv_out_classes(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation: int,
                 in_channels: int,
                 out_channels: int,
                 causal_conv: bool = False,
                 kernel_size: int = 3,
                 dropout_prob: float = 0.0):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        padding = (dilation * (kernel_size - 1)) if self.causal_conv else dilation
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_prob)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.activation(self.conv_dilated(x))
        out = self.dropout(out)
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.activation(self.conv_1x1(out))
        out = self.dropout(out)
        return x + out


class SingleStageModel1(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 causal_conv=False):
        super(SingleStageModel1, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_classes = self.conv_out_classes(out)
        return out_classes, out

class MultiStageModel1(nn.Module):
    def __init__(self, mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv):
        self.num_stages = mstcn_stages  # 4 #2
        self.num_layers = mstcn_layers  # 10  #5
        self.num_f_maps = mstcn_f_maps  # 64 #64
        self.dim = mstcn_f_dim  #2048 # 2048
        self.num_classes = out_features  # 7
        self.causal_conv = mstcn_causal_conv
        print(
            f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
            f" {self.num_f_maps}, dim: {self.dim}")
        super(MultiStageModel1, self).__init__()
        self.stage1 = SingleStageModel1(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel1(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 causal_conv=self.causal_conv))
            for s in range(self.num_stages - 1)
        ])
        self.smoothing = False

    def forward(self, x):
        out_classes, _ = self.stage1(x)
        outputs_classes = out_classes.unsqueeze(0)
        for s in self.stages:
            out_classes, out = s(F.softmax(out_classes, dim=1))
            outputs_classes = torch.cat(
                (outputs_classes, out_classes.unsqueeze(0)), dim=0)
        return out