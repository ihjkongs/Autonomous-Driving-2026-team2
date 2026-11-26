"""Small rectangular maps must keep image geometry and car coordinates aligned."""
from types import SimpleNamespace

import numpy as np
import pytest

from f1tenth_gym.envs.rendering.objects import Map
from f1tenth_gym.envs.rendering.renderer import RenderSpec
from f1tenth_gym.envs.rendering.rendering_pygame import PygameEnvRenderer


@pytest.mark.parametrize('shape', [(80,200), (200,80), (80,80)])
def test_map_scaling_preserves_width_and_height(shape):
    image = np.zeros(shape, dtype=np.float32)
    image[10:20,30:40] = 255.
    rendered = Map(image, zoom_level=2.)
    assert rendered.map_surface.get_size() == (shape[1]*2, shape[0]*2)
    assert rendered.track_map[70,30,0] == 255.


def test_car_and_camera_use_same_scale_as_rectangular_map(monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    track = SimpleNamespace(occupancy_map=np.zeros((80,200),dtype=np.float32),
                            spec=SimpleNamespace(origin=[0.,0.,0.],resolution=.05))
    renderer = PygameEnvRenderer(
        params=dict(length=.57,width=.29), track=track, agent_ids=['agent_0'],
        render_spec=RenderSpec(zoom_in_factor=2., focus_on='agent_0'),
        render_mode='rgb_array', render_fps=40)
    try:
        assert renderer.map_canvases['car'].get_size() == (400,160)
        assert renderer.ppus['car'] == .5
        # A car at (5 m, 2 m) must be centered in that map surface.
        renderer.update(dict(poses_x=[5.],poses_y=[2.],poses_theta=[0.],
            collisions=[0],steering_angles=[0.],velocity=[1.],action=[[0.,1.]],sim_time=0.))
        renderer.cars[0].render(renderer.map_canvases['car'])
        assert renderer.cars[0].rect.center == (200,80)
    finally:
        renderer.close()
