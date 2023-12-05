import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer_size):
        super(Net, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("fc1", nn.Linear(input_size, hidden_size))
        self.model.add_module("norm1d", nn.BatchNorm1d(hidden_size))
        self.model.add_module("relu", nn.ReLU())
        # self.model.add_module("dropout", nn.Dropout(p=0.7))
        if hidden_layer_size > 0 :
            for i in range(hidden_layer_size-1):
                self.model.add_module("fc"+str(i+2), nn.Linear(hidden_size, hidden_size))
                self.model.add_module("norm1d", nn.BatchNorm1d(hidden_size))
                self.model.add_module("relu", nn.ReLU())
                # self.model.add_module("dropout", nn.Dropout(p=0.5))

        self.model.add_module("fc"+str(hidden_layer_size+2), nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.model(x)