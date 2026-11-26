"""Bounded closed-path optimization with a speed-envelope lap-time objective.

This is a local offline optimization, not a globally optimal vehicle trajectory.
Occupancy-map clearance is checked independently of the CSV width estimates.
"""
from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize
from ruamel.yaml import YAML

from utils.speed_profile import speed_profile


@njit(cache=True)
def geometry(points):
    n = len(points)
    ds, curvature = np.empty(n), np.empty(n)
    for i in range(n):
        a = points[i] - points[(i-1) % n]
        b = points[(i+1) % n] - points[i]
        la, lb = np.sqrt((a*a).sum()), np.sqrt((b*b).sum())
        chord = np.sqrt(((a+b)**2).sum())
        ds[i] = lb
        curvature[i] = 2*(a[0]*b[1]-a[1]*b[0])/max(la*lb*chord, 1e-12)
    return ds, curvature


@njit(cache=True)
def envelope_time(points, lateral, acceleration, braking, speed_cap, window):
    """Controller envelope, additionally bounded by pointwise lateral acceleration."""
    n = len(points)
    ds, point_curvature = geometry(points)
    arc = np.empty(n+1)
    arc[0] = 0.
    for i in range(n):
        arc[i+1] = arc[i] + ds[i]
    speed = np.empty(n)
    for i in range(n):
        samples = np.empty((2,2))
        for direction in range(2):
            s = (arc[i] + (-window if direction == 0 else window)) % arc[-1]
            j = min(np.searchsorted(arc, s, side='right')-1, n-1)
            t = (s-arc[j])/max(ds[j],1e-12)
            samples[direction] = points[j]*(1-t) + points[(j+1)%n]*t
        a, b = points[i]-samples[0], samples[1]-points[i]
        angle = np.arctan2(a[0]*b[1]-a[1]*b[0], (a*b).sum())
        k = max(abs(angle)/window,abs(point_curvature[i]))
        speed[i] = max(1.,min(speed_cap,np.sqrt(lateral/max(k,1e-6))))
    for _ in range(2):
        for i in range(n-1,-1,-1):
            speed[i] = min(speed[i],np.sqrt(speed[(i+1)%n]**2+2*braking*ds[i]))
        for i in range(n):
            j = (i+1)%n
            speed[j] = min(speed[j],np.sqrt(speed[i]**2+2*acceleration*ds[i]))
    duration = 0.
    for i in range(n):
        duration += 2*ds[i]/max(speed[i]+speed[(i+1)%n],1e-8)
    return duration


@njit(cache=True)
def objective(offset, center, normal, mode, lateral, acceleration, braking, cap, window, max_curvature):
    points = center + normal*offset.reshape((-1,1))
    ds, k = geometry(points)
    bending = (k*k*ds).sum()
    excess = np.maximum(np.abs(k)-max_curvature,0.)
    penalty = 2000.*(excess*excess*ds).sum()
    smooth = .025*((k-np.roll(k,1))**2).sum()
    if mode == 0:
        return bending + .025*ds.sum() + penalty + smooth
    return envelope_time(points,lateral,acceleration,braking,cap,window) + .003*bending + penalty + smooth


class Corridor:
    def __init__(self, directory, name):
        self.directory = Path(directory)
        self.name = name
        self.spec = YAML(typ='safe').load((self.directory/(name+'_map.yaml')).read_text())
        if abs(float(self.spec['origin'][2])) > 1e-12:
            raise ValueError('Rotated occupancy maps are not supported')
        self.resolution = float(self.spec['resolution'])
        self.origin = np.array(self.spec['origin'][:2])
        with Image.open(self.directory/self.spec['image']) as image:
            free = np.asarray(image.convert('L'))[::-1] > 128
        # Zero border also treats the outside of an all-white map as occupied.
        padded = np.pad(free,1,constant_values=False)
        self.distance = distance_transform_edt(padded)[1:-1,1:-1]*self.resolution
        raw = np.loadtxt(self.directory/(name+'_centerline.csv'), delimiter=',')
        keep = np.r_[True,np.linalg.norm(np.diff(raw[:,:2],axis=0),axis=1)>1e-8]
        self.raw = raw[keep]
        if np.linalg.norm(self.raw[0,:2]-self.raw[-1,:2]) < 1e-8:
            self.raw = self.raw[:-1]
        if self.raw.shape[1] != 4 or len(self.raw) < 8 or not np.isfinite(self.raw).all():
            raise ValueError('Invalid centerline/width CSV')

    def clearance(self, points):
        pixels = (np.asarray(points)-self.origin)/self.resolution
        distances = map_coordinates(self.distance,[pixels[...,1].ravel(),pixels[...,0].ravel()],
                                    order=1,mode='constant',cval=0.,prefilter=False)
        # Conservative allowance for pixel-center rather than cell-boundary distances.
        return distances.reshape(pixels.shape[:-1]) - self.resolution*.5

    def sample(self, step):
        xy = self.raw[:,:2]
        closed = np.vstack([xy,xy[0]])
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]
        spline = CubicSpline(arc,closed,bc_type='periodic')
        positions = np.linspace(0,arc[-1],max(32,int(np.ceil(arc[-1]/step))),endpoint=False)
        center = spline(positions)
        tangent = spline(positions,1)
        tangent /= np.linalg.norm(tangent,axis=1,keepdims=True)
        normal = np.column_stack([-tangent[:,1],tangent[:,0]])
        widths = np.column_stack([np.interp(positions,arc,np.r_[self.raw[:,j],self.raw[0,j]]) for j in (2,3)])
        return center,normal,widths

    def bounds(self, center, normal, widths, half_width, margin):
        clearance = half_width + margin
        if np.min(self.clearance(center)) < clearance:
            raise ValueError('Center reference has insufficient requested clearance')
        limits = []
        for sign, width in [(-1.,widths[:,0]),(1.,widths[:,1])]:
            bound = np.zeros(len(center))
            allowed = np.maximum(0.,width-clearance)
            # Stop at the FIRST obstacle; do not jump to another nearby lane.
            for distance in np.arange(.025,float(allowed.max())+.025,.025):
                eligible = ((bound >= distance-.026) & (allowed >= distance) &
                            (self.clearance(center+normal*(sign*distance)) >= clearance))
                bound = np.where(eligible,distance,bound)
            limits.append(bound)
        return np.column_stack([-limits[0],limits[1]])

    def footprint_clearance(self, points, length, width):
        """Dense path and 15 samples across each oriented vehicle rectangle."""
        closed = np.vstack([points,points[0]])
        ds = np.linalg.norm(np.diff(closed,axis=0),axis=1)
        arc = np.r_[0.,np.cumsum(ds)]
        spline = CubicSpline(arc,closed,bc_type='periodic')
        positions = np.linspace(0,arc[-1],int(np.ceil(arc[-1]/.025)),endpoint=False)
        center = spline(positions)
        tangent = spline(positions,1)
        tangent /= np.linalg.norm(tangent,axis=1,keepdims=True)
        normal = np.column_stack([-tangent[:,1],tangent[:,0]])
        minimum = float('inf')
        for longitudinal in np.linspace(-length/2,length/2,5):
            for lateral in [-width/2,0.,width/2]:
                minimum = min(minimum,float(self.clearance(center+tangent*longitudinal+normal*lateral).min()))
        return minimum


def optimize(directory, name, work, vehicle, margin=.12, sample_step=.25, iterations=90):
    corridor = Corridor(directory,name)
    center,normal,widths = corridor.sample(sample_step)
    bounds = corridor.bounds(center,normal,widths,vehicle['width']/2,margin)
    zero = np.zeros(len(center))
    settings = (float(work['lateral_acceleration']),float(work['planning_acceleration']),
                float(work['braking_deceleration']),float(work['speed_cap']),float(work['curvature_window']))
    max_curvature = float(np.tan(vehicle['max_steer']*.96)/(vehicle['length_f']+vehicle['length_r']))
    history = []
    options = dict(maxiter=iterations,maxfun=150000,ftol=1e-10,gtol=1e-4,eps=1e-5,maxls=30)
    curve = minimize(objective,zero,args=(center,normal,0,*settings,max_curvature),method='L-BFGS-B',bounds=bounds,options=options)
    history.append(dict(stage='minimum_curvature',success=bool(curve.success),iterations=int(curve.nit),message=str(curve.message)))
    timed = minimize(objective,curve.x,args=(center,normal,1,*settings,max_curvature),method='L-BFGS-B',bounds=bounds,options=options)
    history.append(dict(stage='speed_envelope_time',success=bool(timed.success),iterations=int(timed.nit),message=str(timed.message)))
    choices = [('center',zero)]
    # Coupled shifts cross nonsmooth active-set barriers that coordinate finite
    # differences can miss (e.g. moving a complete circular path inward).
    for shift in [-.35,-.2,.2,.35]:
        choices.append((f'coupled_shift{shift:g}',np.clip(zero+shift,bounds[:,0],bounds[:,1])))
    for label,offset in [('curvature',curve.x),('time',timed.x)]:
        for blend in [1.,.75,.5,.25]:
            choices.append((label+f'_blend{blend:g}',offset*blend))
    accepted = []
    for label,offset in choices:
        points = center+normal*offset[:,None]
        closed = np.vstack([points,points[0]])
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]
        spline = CubicSpline(arc,closed,bc_type='periodic')
        dense = np.linspace(0,arc[-1],int(np.ceil(arc[-1]/.025)),endpoint=False)
        first,second = spline(dense,1),spline(dense,2)
        curvature = np.abs(first[:,0]*second[:,1]-first[:,1]*second[:,0])/np.maximum(np.linalg.norm(first,axis=1)**3,1e-12)
        peak = float(curvature.max())
        if peak > max_curvature*1.01:
            continue
        wall = corridor.footprint_clearance(points,vehicle['length'],vehicle['width'])
        if wall < .06:
            continue
        duration = float(envelope_time(points,*settings))
        accepted.append((duration,label,points,offset,wall,peak))
    if not accepted:
        raise ValueError('No path passed the dense footprint clearance check')
    duration,label,points,offset,wall,peak = min(accepted,key=lambda row:row[0])
    _,point_curvature = geometry(points)
    lateral_limit = np.sqrt(work['lateral_acceleration']/np.maximum(np.abs(point_curvature),1e-6))
    arc,speeds,kappa = speed_profile(points,np.minimum(8.,lateral_limit),work)
    return dict(points=points,speed=speeds,offset=offset,curvature=kappa),dict(
        map=name,chosen=label,estimated_lap_s=duration,
        center_estimated_lap_s=float(envelope_time(center,*settings)),
        minimum_static_footprint_clearance_m=wall,max_abs_offset_m=float(np.max(np.abs(offset))),
        max_dense_curvature_per_m=peak,curvature_limit_per_m=max_curvature,
        fraction_noncenter=float(np.mean(np.abs(offset)>.02)),optimization=history,
        candidates=[dict(estimated_lap_s=a[0],name=a[1],footprint_clearance_m=a[4],max_curvature_per_m=a[5]) for a in accepted])


def load_racing_waypoints(path):
    with np.load(path,allow_pickle=False) as data:
        points,speed = data['points'],data['speed']
    if points.ndim != 2 or points.shape[1] != 2 or speed.shape != (len(points),):
        raise ValueError('Invalid racing-line shape')
    if len(points)<8 or not np.isfinite(points).all() or not np.isfinite(speed).all() or (speed<=0).any():
        raise ValueError('Invalid racing-line values')
    if (np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)<1e-8).any():
        raise ValueError('Racing line contains repeated consecutive points')
    waypoints = np.column_stack([points,speed])
    return np.vstack([waypoints,waypoints[0]])
