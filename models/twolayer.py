import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=num_classes)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        out = F.softmax(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out