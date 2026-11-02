"""
Gradient Reversal Layer (GRL).

This module is used for Domain-Adversarial Neural Networks (DANN).
During the forward pass, it acts as an identity function.
During the backward pass, it multiplies the gradient by a negative scalar (alpha).
This forces the feature extractor upstream to learn domain-invariant features.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Autograd function for gradient reversal.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for Domain Adversarial Training.
    
    Args:
        alpha: The scalar to multiply with the reversed gradient.
               Typically this should be annealed from 0 to 1 during training.
    """
    def __init__(self, alpha: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
