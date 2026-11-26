"""Racing-line objective, obstacle bounds and explicit planner opt-in."""
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from utils.racing_line import Corridor, envelope_time, geometry, load_racing_waypoints, optimize
from utils.rule_based import PurePursuitPlanner
from utils.speed_profile import speed_profile


def circle_map(tmp_path):
    name = 'circle'
    yy,xx = np.mgrid[:240,:240]
    radius = np.hypot(xx*.05-6.,yy*.05-6.)
    free = (radius>2.2)&(radius<3.8)
    Image.fromarray((free[::-1]*255).astype('uint8')).save(tmp_path/'circle_map.png')
    (tmp_path/'circle_map.yaml').write_text('image: circle_map.png\nresolution: 0.05\norigin: [-6.0, -6.0, 0.0]\n')
    angles = np.linspace(0,2*np.pi,100,endpoint=False)
    points = np.column_stack([3*np.cos(angles),3*np.sin(angles),np.full(100,.8),np.full(100,.8)])
    np.savetxt(tmp_path/'circle_centerline.csv',points,delimiter=',')
    return name


def work():
    return dict(vgain=1.,speed_cap=6.,lateral_acceleration=4.,planning_acceleration=2.,
                braking_deceleration=2.,curvature_window=.75,min_speed=1.)


def test_time_objective_matches_controller_speed_envelope():
    angle=np.linspace(0,2*np.pi,100,endpoint=False)
    points=np.column_stack([4*np.cos(angle),2*np.sin(angle)])
    _,curvature=geometry(points)
    limits=np.minimum(8.,np.sqrt(4./np.maximum(np.abs(curvature),1e-6)))
    arc,speed,_=speed_profile(points,limits,work())
    expected=np.sum(2*np.diff(arc)/(speed+np.roll(speed,-1)))
    assert envelope_time(points,4.,2.,2.,6.,.75)==pytest.approx(expected,rel=1e-9)


def test_optimizer_improves_circle_without_touching_wall(tmp_path):
    name=circle_map(tmp_path)
    arrays,report=optimize(tmp_path,name,work(),dict(width=.29,length=.57,max_steer=.48,length_f=.2,length_r=.15),iterations=35)
    assert report['estimated_lap_s'] < report['center_estimated_lap_s']*.99
    assert report['max_abs_offset_m'] > .05
    assert report['minimum_static_footprint_clearance_m'] >= .06
    assert report['max_dense_curvature_per_m']<=report['curvature_limit_per_m']*1.01
    assert np.isfinite(arrays['points']).all()
    assert (arrays['speed']<=6.).all() and (arrays['speed']>=1.).all()


def test_normal_bounds_do_not_jump_across_infield_to_other_lane(tmp_path):
    name=circle_map(tmp_path)
    corridor=Corridor(tmp_path,name)
    center,normal,widths=corridor.sample(.3)
    # Deliberately wrong CSV widths: occupancy constraints must stop at first wall.
    bounds=corridor.bounds(center,normal,np.full_like(widths,10.),.145,.12)
    assert np.max(np.abs(bounds))<.8
    for sign,index in [(-1,0),(1,1)]:
        candidate=center+normal*bounds[:,index,None]
        assert corridor.clearance(candidate).min()>=.265-1e-8


def test_loading_is_explicit_and_missing_racing_line_does_not_fallback(tmp_path):
    theta=np.linspace(0,2*np.pi,16,endpoint=False)
    points=np.column_stack([3*np.cos(theta),3*np.sin(theta)])
    line=SimpleNamespace(xs=points[:,0],ys=points[:,1],vxs=np.full(16,4.))
    track=SimpleNamespace(name='circle',centerline=line,raceline=line)
    planner=PurePursuitPlanner(track,.35,'centerline')
    initial=planner.waypoints.copy()
    obs=dict(poses_x=[3.],poses_y=[0.],poses_theta=[np.pi/2],linear_vels_x=[0.])
    params=dict(tlad=.7,vgain=1.,raceline_directory=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        planner.plan(obs,params)
    np.savez(tmp_path/'circle.npz',points=points*.95,speed=np.full(16,2.))
    planner.plan(obs,params)
    assert np.all(planner.waypoints[:,2]==2.)
    assert not np.array_equal(planner.waypoints,initial)
    planner.plan(obs,dict(tlad=.7,vgain=1.))
    assert np.array_equal(planner.waypoints,initial)


def test_invalid_racing_line_is_rejected(tmp_path):
    path=tmp_path/'bad.npz'
    np.savez(path,points=np.zeros((8,2)),speed=np.ones(8))
    with pytest.raises(ValueError,match='repeated'):
        load_racing_waypoints(path)


def test_replaced_reference_invalidates_collected_data_signature(tmp_path):
    from easydict import EasyDict
    from ruamel.yaml import YAML
    from utils.expert_config import ROOT
    from utils.expert_data import signature
    config=EasyDict()
    for name in ['f1tenth','dynamic']:
        config.update(YAML(typ='safe').load((ROOT/'configs/task'/(name+'.yaml')).read_text()))
    config.expert_parameters=dict(raceline_directory=str(tmp_path))
    path=tmp_path/'circle.npz'
    np.savez(path,points=np.zeros((8,2)),speed=np.ones(8))
    before=signature(config)
    np.savez(path,points=np.ones((8,2)),speed=np.ones(8))
    assert signature(config)!=before
