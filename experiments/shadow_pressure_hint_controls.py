"""Frozen RF/LF visual-hint weights and local gap loss for one Shadow ablation."""

import numpy as np


def carry_pressure_weights(expected_proxy,participation):
    expected=np.asarray(expected_proxy,dtype=float)
    aux=np.asarray(participation,dtype=float)
    if expected.shape!=(190,5) or aux.shape!=(190,5) or not np.isfinite(expected).all() or not np.isfinite(aux).all() or np.any(expected<0) or np.any((aux<0)|(aux>1)):
        raise ValueError('expected finite 190x5 uncalibrated hints')
    median=np.median(expected[42:126]*aux[42:126],axis=0)
    selected=median[3:5]
    if selected.sum()<=0:
        raise ValueError('visual RF/LF hints are uninformative')
    return median,1.+selected/selected.sum()


def gap_loss_mm(gaps_mm,weights,delta_from_source,parameter_ranges):
    gaps=np.asarray(gaps_mm,dtype=float);w=np.asarray(weights,dtype=float)
    delta=np.asarray(delta_from_source,dtype=float);span=np.asarray(parameter_ranges,dtype=float)
    if (gaps.shape!=(3,2) or w.shape!=(2,) or delta.shape!=(5,) or span.shape!=(5,)
            or not all(np.isfinite(x).all() for x in (gaps,w,delta,span))
            or np.any(w<=0) or np.any(span<=0)):
        raise ValueError('invalid RF/LF gap objective')
    outside=np.maximum(gaps-1.,0.)**2+np.maximum(-gaps,0.)**2
    deep=np.maximum(-gaps-2.,0.)**2
    return float(np.sum(w[None,:]*(outside+4.*deep))+2.*np.sum((delta/span)**2))
