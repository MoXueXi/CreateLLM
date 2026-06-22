import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    
    def forward(self, x):
        return self.linear(x)