"""Residual velocity regression in standardized coefficient space."""
import torch


def velocity_loss(prediction,target,gradient_weight=0.):
    error=prediction-target
    loss=error.square().mean()
    if gradient_weight:
        delta=2*torch.pi/error.shape[1]
        derivative=(error.roll(-1,1)-error.roll(1,1))/(2*delta)
        loss=loss+gradient_weight*derivative.square().mean()
    return loss
