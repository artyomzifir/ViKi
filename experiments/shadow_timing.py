"""Retiming of saved arm and hand commands without changing their poses."""
import numpy as np


def retime_commands(timestamps, commands, speed_limits):
    timestamps=np.asarray(timestamps,dtype=np.int64)
    commands=np.asarray(commands,dtype=float)
    limits=np.asarray(speed_limits,dtype=float)
    if (timestamps.ndim!=1 or len(timestamps)<2 or commands.ndim!=2
            or commands.shape[0]!=len(timestamps) or limits.shape!=(commands.shape[1],)
            or np.any(np.diff(timestamps)<=0) or not np.isfinite(commands).all()
            or not np.isfinite(limits).all() or np.any(limits<=0)):
        raise ValueError('retiming requires aligned finite commands and positive times/limits')
    required=np.max(np.abs(np.diff(commands,axis=0))/limits,axis=1)*1.02
    segment_us=np.maximum(np.diff(timestamps),np.ceil(required*1e6).astype(np.int64))
    times=np.r_[timestamps[0],timestamps[0]+np.cumsum(segment_us)]
    margin=float(np.min(limits-np.abs(np.diff(commands,axis=0))/(segment_us[:,None]/1e6)))
    return times,margin
