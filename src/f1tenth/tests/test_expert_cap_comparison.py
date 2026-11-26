"""Cap studies must unlock cached route speeds without moving the racing line."""
import numpy as np
import pytest
from compare_expert_speed_caps import rebuild_profile,prepare_profiles


@pytest.mark.parametrize('radius,expected_gain',[(1.,0.),(30.,2.)])
def test_rebuild_preserves_geometry_and_distinguishes_corner_limit(tmp_path,radius,expected_gain):
    theta=np.linspace(0,2*np.pi,256,endpoint=False)
    points=radius*np.column_stack([np.cos(theta),np.sin(theta)])
    work=dict(speed_cap=6.,vgain=1.,min_speed=1.,lateral_acceleration=4.,
              planning_acceleration=2.,braking_deceleration=2.,curvature_window=.75)
    speed=rebuild_profile(points,work)
    src=tmp_path/'source';src.mkdir()
    np.savez_compressed(src/'circle.npz',points=points,speed=speed,offset=np.zeros(len(points)))
    audit=prepare_profiles(src,tmp_path/'cap8',['circle'],work,8.)
    with np.load(tmp_path/'cap8/circle.npz') as data:
        assert np.array_equal(data['points'],points)
        assert np.max(data['speed']-speed)==pytest.approx(expected_gain,abs=.001)
    assert audit[0]['baseline_profile_max_error']==0.
    assert audit[0]['literal_cap_change_planner_profile_identical']
    with np.load(src/'circle.npz') as data:assert np.array_equal(data['speed'],speed)
