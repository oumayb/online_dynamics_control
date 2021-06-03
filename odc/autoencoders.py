"""
Contains AutoEncoder parent class
"""
import torch.nn as nn
from abc import abstractmethod


class AutoEncoder(nn.Module):
    """
    Has 3 class methods:
        - encode
        - decode
        - forward
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        @abstractmethod
        def encode(self, x):
            pass

        @abstractmethod
        def decode(self, z):
            pass

        @abstractmethod
        def forward(self, x):
            pass
