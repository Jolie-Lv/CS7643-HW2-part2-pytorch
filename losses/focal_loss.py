import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    E = (1 - np.power(beta, cls_num_list)) / (1 - beta)
    alpha = 1 / E
    C = len(cls_num_list)
    per_cls_weights = alpha / (np.sum(alpha) / C)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        n_classes = len(self.weight)
        target_one_hot = F.one_hot(target, n_classes).float()

        weights = self.weight
        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(target_one_hot.shape[0], 1) * target_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, n_classes)

        BCLoss = F.binary_cross_entropy_with_logits(input=input, target=target_one_hot, reduction="none")

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * target_one_hot * input - self.gamma * torch.log(1 +
                                                                               torch.exp(-1.0 * input)))

        loss = modulator * BCLoss

        weighted_loss = weights * loss

        focal_loss = torch.sum(weighted_loss)
        loss = focal_loss / torch.sum(target_one_hot)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
