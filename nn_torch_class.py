# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:38:17 2024

@author: romas
"""

import torch.nn as nn

class NN_torch(nn.Module):
    
    def __init__(self, pool_size: tuple, kernel_size: tuple, classes_amount: int, input_channels: int):
        
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),
            nn.Conv2d(32, 64, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),
            nn.Conv2d(64, 128, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),            
            nn.Flatten(),
            
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, classes_amount),
            nn.Softmax(dim = 1)
            )
        
        self.double()
        
    def forward(self, x):
        
        logits = self.layers(x)
        
        return logits