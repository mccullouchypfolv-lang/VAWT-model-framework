"""Residual velocity regression in standardized coefficient space."""
import torch
import math
import numbers


def validate_velocity_regression(regression,huber_delta=1.):
    if regression not in ('mse','huber'):raise ValueError('unknown velocity regression loss')
    if regression=='huber' and (isinstance(huber_delta,bool) or not isinstance(huber_delta,numbers.Real)
            or not math.isfinite(huber_delta) or huber_delta<=0):
        raise ValueError('Huber delta must be finite and positive')
    return regression


def validate_time_weighting(mode):
    if mode not in ('none','floral_quadratic'):raise ValueError('unknown flow time weighting')
    return mode


def complex_spectral_loss(prediction,target,*,time=None,time_weighting='none'):
    """Full-band complex L1: mean |FFT_ortho(v-u)|, including phase error."""
    validate_time_weighting(time_weighting)
    if prediction.ndim!=3 or prediction.shape!=target.shape:raise ValueError('matching [batch,theta,channel] required')
    error=prediction-target;n=error.shape[1]
    spectrum=torch.fft.rfft(error,dim=1,norm='ortho')
    multiplicity=error.new_full((spectrum.shape[1],),2.);multiplicity[0]=1.
    if n%2==0:multiplicity[-1]=1.
    per_sample=(spectrum.abs()*multiplicity[:,None]).sum((1,2))/(n*error.shape[2])
    if time_weighting=='floral_quadratic':
        if time is None:raise ValueError('time required for quadratic weighting')
        t=torch.as_tensor(time,dtype=error.dtype,device=error.device)
        if t.ndim<1 or t.shape[0]!=error.shape[0] or t.numel()!=error.shape[0] or not torch.isfinite(t).all() or ((t<0)|(t>1)).any():
            raise ValueError('one finite time in [0,1] per sample required')
        per_sample=per_sample*(1+2*t.reshape(-1).square())
    return per_sample.mean()


def velocity_loss(prediction,target,gradient_weight=0.,*,time=None,time_weighting='none',
                  sample_weights=None,angular_weights=None,regression='mse',huber_delta=1.):
    validate_time_weighting(time_weighting)
    validate_velocity_regression(regression,huber_delta)
    error=prediction-target
    if sample_weights is not None or angular_weights is not None or regression=='huber':
        element=torch.nn.functional.huber_loss(prediction,target,reduction='none',delta=huber_delta) if regression=='huber' else error.square()
        if gradient_weight:
            delta=2*torch.pi/error.shape[1]
            derivative=(error.roll(-1,1)-error.roll(1,1))/(2*delta)
            element=element+gradient_weight*derivative.square()
        if angular_weights is not None:
            angular=torch.as_tensor(angular_weights,device=error.device,dtype=error.dtype)
            if angular.shape!=error.shape[1:] or not torch.isfinite(angular).all() or (angular<=0).any():
                raise ValueError('angular weights must be positive finite [theta,channels]')
            element=element*angular
        per_sample=element.mean(dim=tuple(range(1,error.ndim)))
        if sample_weights is not None:
            sample=torch.as_tensor(sample_weights,device=error.device,dtype=error.dtype)
            if sample.shape!=(error.shape[0],) or not torch.isfinite(sample).all() or (sample<=0).any():
                raise ValueError('sample weights must be positive finite [batch]')
            per_sample=per_sample*sample
        if time_weighting=='floral_quadratic':
            if time is None:raise ValueError('time required for quadratic weighting')
            t=torch.as_tensor(time,device=error.device,dtype=error.dtype)
            if t.ndim<1 or t.shape[0]!=error.shape[0] or t.numel()!=error.shape[0] or not torch.isfinite(t).all() or ((t<0)|(t>1)).any():
                raise ValueError('time must contain one finite value in [0,1] per sample')
            per_sample=per_sample*(1+2*t.reshape(-1).square())
        return per_sample.mean()
    if time_weighting=='floral_quadratic':
        if time is None:raise ValueError('time required for quadratic weighting')
        t=torch.as_tensor(time,device=error.device,dtype=error.dtype)
        if t.ndim<1 or t.shape[0]!=error.shape[0] or t.numel()!=error.shape[0] or not torch.isfinite(t).all() or ((t<0)|(t>1)).any():
            raise ValueError('time must contain one finite value in [0,1] per sample')
        axes=tuple(range(1,error.ndim))
        per_sample=error.square().mean(dim=axes)
        if gradient_weight:
            delta=2*torch.pi/error.shape[1]
            derivative=(error.roll(-1,1)-error.roll(1,1))/(2*delta)
            per_sample=per_sample+gradient_weight*derivative.square().mean(dim=axes)
        return ((1+2*t.reshape(-1).square())*per_sample).mean()
    # Preserve historical uniform-MSE reduction and gradients exactly.
    loss=error.square().mean()
    if gradient_weight:
        delta=2*torch.pi/error.shape[1]
        derivative=(error.roll(-1,1)-error.roll(1,1))/(2*delta)
        loss=loss+gradient_weight*derivative.square().mean()
    return loss
