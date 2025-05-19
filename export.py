import onnx

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

# import onnx

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

import random

use_bias = False
# bottleneck = 64
# bottleneck = 32
# bottleneck = 16
bottleneck1 = 20
bottleneck2 = 15
# bottleneck = 8
# bottleneck = 5
class SimpleNN(nn.Module):
    def __init__(self, res):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(res * res, bottleneck1, bias=use_bias)
        self.fc2 = nn.Linear(bottleneck1, bottleneck2, bias=use_bias)
        self.fc3 = nn.Linear(bottleneck2, 10, bias=use_bias)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN(5)
filename = 'BEST2_mnist_2x2_5_20_15_True_30e_False_False.model'
model.load_state_dict(torch.load(filename))

model.eval()

# quantized_model = torch.quantization.quantize_dynamic(
#     model, {nn.Linear}, dtype=torch.qint8
# )

torch.onnx.export(
    # quantized_model,
    model,
    torch.randn(1, 25),
    "model.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
quantized_model_path = "model_quantized.onnx"
quantize_dynamic(
    "model.onnx",
    quantized_model_path,
    weight_type=QuantType.QInt8
)

print(f"Quantized model saved to {quantized_model_path}")