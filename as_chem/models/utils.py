import torch.nn as nn

supported_activations = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid()
}