"""
This module contains regression loss functions to be used in conjunction with Emb2Emb.
"""

import torch
from torch import nn
from torch.nn import CosineEmbeddingLoss
from torch.nn.modules.loss import BCELoss


class CosineLoss(CosineEmbeddingLoss):

    def __init__(self, **args):
        super(CosineLoss, self).__init__(**args)

    def forward(self, predicted, true):
        target = torch.ones((predicted.size(0))).to(predicted.device)
        return super(CosineLoss, self).forward(predicted, true, target)


class FlipLoss(nn.Module):

    def __init__(self, baseloss, classifier, lambda_clfloss=0.5,
                 increase_until=10000,
                 *args):
        super(FlipLoss, self).__init__(*args)
        self.baseloss = baseloss
        # assumed to return logit for binary classification (sigmoid)
        self.classifier = classifier
        self.sigmoid = nn.Sigmoid()
        self.bce = BCELoss()
        self.lambda_clfloss = lambda_clfloss
        self.increase_until = increase_until

        self.i = 0

        for p in self.classifier.parameters():
            p.requires_grad = False

        self.classifier.eval()

    def _get_lambda(self):

        if self.i >= self.increase_until:
            l = self.lambda_clfloss
        else:
            l = (float(self.i) / self.increase_until) * self.lambda_clfloss

        if self.training:
            self.i = self.i + 1

        return l

    def forward(self, predicted, true):
        baseloss = self.baseloss(predicted, true)

        predicted_label = self.classifier(predicted)
        predicted_label = self.sigmoid(predicted_label)

        # we are assuming that the "fake" example was trained to be
        # label '0'
        desired_label = torch.zeros_like(
            predicted_label, device=predicted_label.device)

        clf_loss = self.bce(predicted_label, desired_label)
        l = self._get_lambda()
        clf_loss = l * clf_loss
        baseloss = (1 - l) * baseloss
        loss = clf_loss + baseloss

        return loss
