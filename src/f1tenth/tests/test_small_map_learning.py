"""Chronological sampling and full-progress scoring at compact-track finish zones."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from easydict import EasyDict
from utils.small_map_learning import chronological_indices, collect, DiskSampler


@pytest.mark.parametrize('lateral_acceleration',[5,6])
def test_model7_changes_teacher_corner_limit_without_changing_vehicle_response(lateral_acceleration):
    from train_small_map_policy import configuration
    from utils.expert_config import ROOT
    old=configuration()
    new=configuration(ROOT/f'configs/experiment_small_maps_raceline_lat{lateral_acceleration}_6mps.yaml')
    assert old.expert_parameters['lateral_acceleration']==4.
    assert new.expert_parameters['lateral_acceleration']==float(lateral_acceleration)
    assert old.longitudinal_response==new.longitudinal_response
    for config in (old,new):
        config.expert_parameters.pop('lateral_acceleration')
        config.expert_parameters.pop('raceline_directory')
        config.pop('experiment_path')
        experiment=config.pop('experiment_config')
        experiment.pop('description')
        experiment['expert'].pop('lateral_acceleration')
        experiment['expert'].pop('raceline_directory')
        config['experiment_config']=experiment
    assert old==new


def test_history_never_crosses_episode_boundary():
    starts=np.array([0,0,0,3,3,3])
    assert chronological_indices([0,2,3,5],starts).tolist()==[
        [0,0,0,0],[0,0,1,2],[3,3,3,3],[3,3,4,5]]


def test_heldout_collection_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError,match='Validation'):
        collect(None,['heldout'],['heldout'],tmp_path/'data')
    assert not (tmp_path/'data').exists()


def test_disk_sampler_excludes_masked_labels_and_pads_history(tmp_path):
    np.save(tmp_path/'scans.npy',np.repeat(np.arange(5,dtype=np.float32)[:,None],1080,axis=1))
    np.save(tmp_path/'actions.npy',np.column_stack([np.arange(5),np.zeros(5)]).astype(np.float32))
    np.save(tmp_path/'starts.npy',np.array([0,0,0,3,3]))
    np.save(tmp_path/'mask.npy',np.array([False,False,True,True,False]))
    (tmp_path/'metadata.json').write_text(json.dumps(dict(maps=['train'],episodes=[
        dict(map='train',start=0,stop=3),dict(map='train',start=3,stop=5)])))
    sampler=DiskSampler(tmp_path,2,['train'],['val'])
    x,y=sampler.batch(100)
    assert set(y[:,0])=={2,3}
    for frames,label in zip(x.reshape(-1,4,1080),y):
        assert frames[:,0].tolist()==([0,0,1,2] if label[0]==2 else [3,3,3,3])
    with pytest.raises(ValueError,match='forbidden'):
        DiskSampler(tmp_path,2,['train'],['train'])


@pytest.mark.parametrize('collision_step,expected',[(None,'complete'),(4,'collision')])
def test_strict_policy_completion_ignores_early_checkpoint(monkeypatch,collision_step,expected):
    import utils.policy_evaluation as runner
    class Env:
        def __init__(self):
            self.unwrapped=self;self.steps=0;self.current_time=0.;self.collisions=[False]
            self.track=SimpleNamespace(centerline=SimpleNamespace(xs=np.array([0.,1.,1.,0.]),ys=np.array([0.,0.,1.,1.])))
            self.sim=SimpleNamespace(agent_poses=np.zeros((1,3)),agents=[SimpleNamespace(state=np.zeros(7),response_parameters={})])
        def raw(self):
            x,y=[(0,0),(1,0),(1,1),(0,1)][self.steps%4]
            return dict(poses_x=[x],poses_y=[y])
        def reset(self,options):return np.ones(4320),dict(obs_dict=self.raw())
        def getObs(self,raw):return np.ones(4320)
        def step(self,command):
            self.steps+=1;self.current_time+=.025
            self.collisions=[self.steps==collision_step]
            return self.raw(),0,True,False,dict(checkpoint_done=[True])
        def close(self):pass
    monkeypatch.setattr(runner,'F1Wrapper',lambda *a,**k:Env())
    monkeypatch.setattr(runner,'policy_action',lambda obs:np.zeros(2))
    args=EasyDict(evaluation_laps=1,full_progress_completion=True,max_episode_steps=10,
        max_steer=.48,min_speed=1.,max_speed=6.,timestep=.025)
    row=runner.episode((args,'square','cl_grid_static',1,None))
    assert row['outcome']==expected,row
    assert row['steps']==4 and row['forward_progress_m']==4.
