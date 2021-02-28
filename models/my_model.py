import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        # structure = [64, 'M', 128, 'M', 256, 256, "M", 512, 512, "M", 512, 512, 'M']
        structure = [32, 32, 'M', 64, 'M', 128, 128, 'M', 256, "D", 256, 'A', 512, "D", 512, 'M']
        self.structure = self._make_layers(structure)
        self.classifier = nn.Linear(512, 10)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        out = self.structure(x)
        out = out.view(out.size(0), -1)
        outs = self.classifier(out)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs

    def _make_layers(self, structure):
        layers = []
        in_channels = 3
        for x in structure:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == "A":
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == "D":
                layers += [nn.Dropout2d(0.05)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
