"""Fit Shadow's different anatomy to landmarks from the final articulated fit."""
import json
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from viki.perception.hand_model import HandParams, build, fk_landmark_positions

HUMAN_FROM_SHADOW = np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])

def fit_shadow(clean, model, collision):
    human = build(HandParams.from_dict(json.loads(str(clean['geometry_hand_params_json']))))
    fitted = np.asarray(clean['hand_fit_joint_angles'], float)
    names = [str(n) for n in model.names if str(n).startswith('rh_')]
    indices = [model.idx_qs[model.getJointId(n)] for n in names]
    data = model.createData()
    palm = model.getFrameId('rh_palm', pin.FrameType.BODY)
    codes = ['th','ff','mf','rf','lf']
    frames = [model.getFrameId('rh_'+c+s, pin.FrameType.BODY)
              for c in codes for s in ['proximal','middle','distal','tip']]
    # Shadow thumb has no proximal joint landmark equivalent to human CMC;
    # all four link origins nevertheless describe its articulated chain.
    lo = model.lowerPositionLimit[indices].copy()
    hi = model.upperPositionLimit[indices].copy()
    neutral = pin.neutral(model)
    def positions(x):
        q=neutral.copy();q[indices]=x
        pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
        p=data.oMf[palm]
        return np.stack([p.rotation.T@(data.oMf[f].translation-p.translation) for f in frames])
    rest=positions(np.zeros(len(names)))
    def clearances(x):
        p=positions(x).reshape(5,4,3)
        # Sampled phalanx separation proxy for the fitting objective;
        # the final plan is separately checked with every vendor collision body.
        samples=np.concatenate([p,(p[:,:-1]+p[:,1:])/2],axis=1)
        return np.array([np.linalg.norm(samples[a,:,None]-samples[b,None,:],axis=-1).min()-.018
                         for a in range(5) for b in range(a)])
    result=[]; errors=[]
    prev=np.clip(np.zeros(len(names)),lo+1e-6,hi-1e-6)
    for t,hq in enumerate(fitted):
        hq=hq.copy();hq[:3]=0;hq[3:7]=[0,0,0,1]
        pts=fk_landmark_positions(human,hq,list(range(1,21)))@HUMAN_FROM_SHADOW
        # Match directions along each phalanx, using Shadow segment lengths.
        target=rest.copy()
        for f in range(5):
            start=f*4
            for j in range(1,4):
                v=pts[start+j]-pts[start+j-1]
                length=np.linalg.norm(rest[start+j]-rest[start+j-1])
                target[start+j]=target[start+j-1]+v/(np.linalg.norm(v)+1e-12)*length
        def residual(x):
            p=positions(x)
            return np.r_[(p-target).ravel()*100, (x-prev)*.08, np.minimum(clearances(x)-.002,0)*2000]
        fit=least_squares(residual,prev,bounds=(lo+1e-7,hi-1e-7),max_nfev=40)
        prev=fit.x

        if t%20==0: print('finger fit',t,'minimum cross-finger gap',float(clearances(prev).min()),flush=True)
        result.append(prev.copy());errors.append(np.linalg.norm(positions(prev)-target,axis=1))
    print('Shadow anatomical fit mean/median mm',np.mean(errors)*1000,np.median(errors)*1000,flush=True)
    return {n:np.array(result)[:,i] for i,n in enumerate(names)}
