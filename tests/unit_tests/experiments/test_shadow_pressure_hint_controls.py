import numpy as np
import pytest
from experiments.shadow_pressure_hint_controls import carry_pressure_weights,gap_loss_mm


def test_visual_weights_are_frozen_rf_lf_shares_not_forces():
    expected=np.ones((190,5));aux=np.full((190,5),.5)
    expected[42:126,3]=2.;expected[42:126,4]=1.
    median,weight=carry_pressure_weights(expected,aux)
    np.testing.assert_allclose(median[3:5],[1.,.5])
    np.testing.assert_allclose(weight,[1.+2/3,1.+1/3])
    with pytest.raises(ValueError):carry_pressure_weights(expected,np.zeros_like(aux))


def test_gap_loss_prefers_near_contact_and_penalizes_deep_penetration():
    w=np.ones(2);delta=np.zeros(5);span=np.ones(5)
    ideal=gap_loss_mm(np.full((3,2),.5),w,delta,span)
    far=gap_loss_mm(np.full((3,2),5.),w,delta,span)
    deep=gap_loss_mm(np.full((3,2),-5.),w,delta,span)
    assert ideal==0 and far>0 and deep>far
    assert gap_loss_mm(np.full((3,2),.5),w,np.ones(5),span)>ideal
