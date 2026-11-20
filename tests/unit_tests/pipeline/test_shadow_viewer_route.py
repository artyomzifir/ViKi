"""Shadow selection must never silently show the default gripper plan."""
import asyncio
import hashlib
import json
import pytest
from fastapi import HTTPException
from viki.contracts import Episode
from viki.server.routes import pipeline


def test_shadow_requires_matching_saved_plan(tmp_path, monkeypatch):
    ep=Episode(tmp_path)
    monkeypatch.setattr(pipeline,'_episode',lambda _:ep)
    (tmp_path/'plan.h5').write_bytes(b'old two finger plan')
    assert asyncio.run(pipeline.retarget_scene('demo',artifact='shadow_object'))['ready'] is False
    plan=tmp_path/'shadow_object_plan.npz';plan.write_bytes(b'saved shadow command artifact')
    scene=tmp_path/'shadow_object_viewer.json'
    scene.write_text(json.dumps({'ready':True,'gripper_model':'shadow_hand_right','plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest()}))
    assert asyncio.run(pipeline.retarget_scene('demo',artifact='shadow_object'))['gripper_model']=='shadow_hand_right'
    plan.write_bytes(b'updated trajectory')
    with pytest.raises(HTTPException) as error:
        asyncio.run(pipeline.retarget_scene('demo',artifact='shadow_object'))
    assert error.value.status_code==409


def test_shadow_mesh_stays_inside_asset_root(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline.config,'MODELS_DIR',str(tmp_path))
    root=tmp_path/'shadow_hand';root.mkdir()
    (tmp_path/'outside.obj').write_text('v 0 0 0')
    with pytest.raises(HTTPException) as error:
        asyncio.run(pipeline.shadow_mesh('../outside.obj'))
    assert error.value.status_code==404
