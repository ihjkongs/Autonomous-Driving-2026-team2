"""Train-only, disk-backed chronological LiDAR examples for small-map BC/DAgger."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import random

import numpy as np

from utils.env import F1Wrapper, normalize_speed, unnormalize_speed
from utils.expert_config import ROOT
from utils.rule_based import PurePursuitPlanner, nearest_point_on_trajectory


def select_maps(manifest, split):
    names=[m['name'] for m in manifest['maps'] if m['split']==split]
    if not names or len(names)!=len(set(names)):
        raise ValueError('Missing or repeated map identifiers')
    return names


def chronological_indices(index, starts, frames=4):
    index=np.asarray(index,dtype=np.int64)
    return np.maximum(index[:,None]-np.arange(frames-1,-1,-1),starts[index,None])


def _collect(job):
    args,name,mode,seed,laps,directory,recovery,beta = job
    config=deepcopy(args)
    config.seed,config.reset_config=seed,{'type':mode}
    random.seed(seed);np.random.seed(seed)
    rng=np.random.RandomState(seed+377)
    env=None;scans=[];actions=[];masks=[]
    row=dict(map=name,mode=mode,seed=seed,required_laps=laps,outcome='error',steps=0,
             kind='learner_recovery' if recovery else 'verified_expert',teacher_steps=0)
    try:
        env=F1Wrapper(config,[name])
        observation,info=env.reset(options={'map':name})
        if recovery:
            poses=env.unwrapped.sim.agent_poses.copy()
            lateral=rng.uniform(-.12,.12)
            poses[0,:2]+=lateral*np.array([-np.sin(poses[0,2]),np.cos(poses[0,2])])
            poses[0,2]+=rng.uniform(-.08,.08)
            observation,info=env.reset(options={'map':name,'poses':poses})
        native=env.unwrapped
        planner=PurePursuitPlanner(native.track,config.length_f+config.length_r,
                                   config.expert_parameters['reference'])
        xy=np.column_stack([native.track.centerline.xs,native.track.centerline.ys]).astype(float)
        xy=np.vstack([xy,xy[0]])
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        length=float(arc[-1])
        def position(raw):
            point=np.array([raw['poses_x'][0],raw['poses_y'][0]],dtype=float)
            _,_,t,i=nearest_point_on_trajectory(point,xy)
            return float(arc[i]+t*(arc[i+1]-arc[i]))
        previous=position(info['obs_dict']);progress=0.
        row['initial_pose']=native.sim.agent_poses[0].tolist()
        row['response']=native.sim.agents[0].response_parameters
        for step in range(1,config.max_episode_steps+1):
            raw=info['obs_dict']
            # Loading the external racing reference happens before measuring error.
            speed,steer=planner.plan(raw,config.expert_parameters)
            point=np.array([raw['poses_x'][0],raw['poses_y'][0]],dtype=float)
            _,distance,_,nearest=nearest_point_on_trajectory(point,planner.waypoints[:,:2])
            segment=planner.waypoints[nearest+1,:2]-planner.waypoints[nearest,:2]
            heading=float(np.arctan2(segment[1],segment[0]))
            error=float((raw['poses_theta'][0]-heading+np.pi)%(2*np.pi)-np.pi)
            if recovery:
                if distance>.12 or abs(error)>.2:speed=min(speed,2.5)
                if distance>.35 or abs(error)>.45:speed=min(speed,1.5)
            label=np.clip([steer/config.max_steer,
                normalize_speed(speed,config.min_speed,config.max_speed)],-1.,1.).astype(np.float32)
            if not np.isfinite(observation).all() or not np.isfinite(label).all():
                raise ValueError('Non-finite input/teacher label')
            # Store every consecutive current scan; reconstruct history at sampling.
            scans.append(np.asarray(observation[-1080:],dtype=np.float32).copy())
            actions.append(label)
            if recovery:
                from utils.policy_evaluation import policy_action
                learner=policy_action(observation)
                hard=abs(learner[0]-label[0])>.035 or abs(learner[1]-label[1])>.12
                masks.append(bool(hard or step%6==0))
                use_teacher=step<=40 or distance>.20 or abs(error)>.30 or rng.rand()<beta
                applied=label if use_teacher else learner
                row['teacher_steps']+=int(use_teacher)
            else:
                applied=label;masks.append(True);row['teacher_steps']+=1
            command=np.array([applied[0]*config.max_steer,
                             unnormalize_speed(applied[1],config.min_speed,config.max_speed)])
            raw,_,_,truncated,info=native.step(command)
            observation=env.getObs(raw);info['obs_dict']=raw
            current=position(raw)
            progress+=(current-previous+length/2)%length-length/2;previous=current
            row['steps']=step
            if native.collisions[0]:
                row['outcome']='collision'
                if recovery:masks[max(0,len(masks)-10):]=[False]*min(10,len(masks))
                break
            if progress>=laps*length:
                row['outcome']='complete';break
            if truncated:
                row['outcome']='timeout';break
        else:row['outcome']='timeout'
        row.update(progress_fraction=progress/(laps*length),elapsed_s=float(native.current_time),
                   label_examples=int(sum(masks)),completed_laps=int(progress//length))
        if not recovery and row['outcome']!='complete':
            raise RuntimeError('Expert rollout did not complete; no labels accepted')
        if not scans:raise RuntimeError('Empty collection episode')
        destination=Path(directory)
        np.save(destination/'scans.npy',np.asarray(scans,dtype=np.float32))
        np.save(destination/'actions.npy',np.asarray(actions,dtype=np.float32))
        np.save(destination/'mask.npy',np.asarray(masks,dtype=bool))
        row['directory']=str(destination)
        (destination/'episode.json').write_text(json.dumps(row,indent=2)+'\n')
        return row
    finally:
        if env is not None:env.close()


def collect(args,maps,forbidden,output,workers=8,recovery_actor=None,recovery_scale=None,
            beta=.15,laps=2,seeds=(124,2026)):
    if set(maps)&set(forbidden):raise ValueError('Validation maps cannot enter label collection')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    recovery=recovery_actor is not None
    jobs=[]
    for name in maps:
        for mode,seed in [('cl_grid_static',seeds[0]),('cl_random_static',seeds[1])]:
            directory=output/f'episode{len(jobs):05d}';directory.mkdir()
            jobs.append((args,name,mode,seed,laps,str(directory),recovery,beta))
    kwargs=dict(max_workers=workers,mp_context=get_context('spawn'))
    if recovery:
        from utils.policy_evaluation import initialize_policy
        kwargs.update(initializer=initialize_policy,initargs=(str(recovery_actor),str(recovery_scale)))
    records=[]
    with ProcessPoolExecutor(**kwargs) as pool,(output/'episodes.jsonl').open('w') as stream:
        futures=[pool.submit(_collect,j) for j in jobs]
        for future in as_completed(futures):
            row=future.result();records.append(row)
            stream.write(json.dumps(row)+'\n');stream.flush()
            if len(records)%40==0 or row['outcome']!='complete':
                print(f"[Collect {len(records)}/{len(jobs)}] {row['map']} {row['outcome']} labels={row['label_examples']}",flush=True)
    records.sort(key=lambda r:r['directory'])
    total=sum(r['steps'] for r in records)
    scans=np.lib.format.open_memmap(output/'scans.npy',mode='w+',dtype=np.float32,shape=(total,1080))
    actions=np.lib.format.open_memmap(output/'actions.npy',mode='w+',dtype=np.float32,shape=(total,2))
    starts=np.lib.format.open_memmap(output/'starts.npy',mode='w+',dtype=np.int64,shape=(total,))
    mask=np.lib.format.open_memmap(output/'mask.npy',mode='w+',dtype=bool,shape=(total,))
    position=0
    for row in records:
        directory=Path(row['directory']);end=position+row['steps']
        scans[position:end]=np.load(directory/'scans.npy',mmap_mode='r')
        actions[position:end]=np.load(directory/'actions.npy',mmap_mode='r')
        mask[position:end]=np.load(directory/'mask.npy',mmap_mode='r')
        starts[position:end]=position
        row.update(start=position,stop=end)
        position=end
    for array in [scans,actions,starts,mask]:array.flush()
    del scans,actions,starts,mask
    from utils.expert_data import signature
    metadata=dict(maps=list(maps),forbidden_maps=list(forbidden),rows=total,episodes=records,
       label_examples=sum(r['label_examples'] for r in records),kind='recovery' if recovery else 'verified_expert',
       outcomes=dict(Counter(r['outcome'] for r in records)),signature=signature(args),
       history='Four consecutive scans, oldest first; repeat episode first scan at start; never cross episode boundaries.',
       storage='Current scan float32 once per step; actions normalized steering/speed; no map or pose in network input.')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


class DiskSampler:
    def __init__(self,directory,seed,allowed_maps,forbidden_maps):
        self.directory=Path(directory)
        self.metadata=json.loads((self.directory/'metadata.json').read_text())
        if set(self.metadata['maps'])-set(allowed_maps) or set(self.metadata['maps'])&set(forbidden_maps):
            raise ValueError('Dataset contains forbidden/unknown training maps')
        self.scans=np.load(self.directory/'scans.npy',mmap_mode='r')
        self.actions=np.load(self.directory/'actions.npy',mmap_mode='r')
        self.starts=np.load(self.directory/'starts.npy',mmap_mode='r')
        mask=np.load(self.directory/'mask.npy',mmap_mode='r')
        self.indices=np.flatnonzero(mask)
        if not len(self.indices):raise ValueError('No eligible examples')
        weights=np.zeros(len(mask),dtype=np.float64)
        counts=Counter()
        for ep in self.metadata['episodes']:
            counts[ep['map']]+=int(mask[ep['start']:ep['stop']].sum())
        for ep in self.metadata['episodes']:
            s,e=ep['start'],ep['stop'];y=self.actions[s:e]
            weights[s:e]=(1+2*np.abs(y[:,0])+2*(y[:,1]<-.2))/max(1,counts[ep['map']])
        self.cdf=np.cumsum(weights[self.indices]);self.rng=np.random.RandomState(seed)

    def batch(self,size):
        slots=np.searchsorted(self.cdf,self.rng.rand(size)*self.cdf[-1])
        index=self.indices[slots]
        history=chronological_indices(index,self.starts)
        return np.asarray(self.scans[history]).reshape(size,4320),np.asarray(self.actions[index])
