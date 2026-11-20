import numpy as np
import pytest
from experiments.shadow_timing import retime_commands


def test_retiming_includes_fingers_and_preserves_slow_segments():
    q=np.zeros((4,28));q[2:,27]=.4
    t=np.array([100,100100,200100,300100],dtype=np.int64)
    before=q.copy()
    actual,margin=retime_commands(t,q,np.full(28,2.))
    assert np.array_equal(q,before)
    assert actual[0]==t[0]
    assert np.diff(actual)[0]==100000
    assert np.diff(actual)[1]>=204000
    assert np.diff(actual)[2]==100000
    assert margin>=0
    assert np.all(np.abs(np.diff(q,axis=0))/(np.diff(actual)[:,None]/1e6)<=2)


def test_retiming_rejects_unusable_inputs():
    with pytest.raises(ValueError):
        retime_commands([0,0],np.zeros((2,28)),np.ones(28))
    with pytest.raises(ValueError):
        retime_commands([0,100],np.zeros((2,28)),np.zeros(28))
