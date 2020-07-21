import torch
from torch import nn

class DimreduceModel(nn.Sequential):
    def __init__(self, input_size, output_size):
        super(DimreduceModel, self).__init__()
        self.add_module("linear_layer1",nn.Linear(in_features=input_size, out_features=output_size, bias=False))
        self.add_module("act1",nn.Tanh())

