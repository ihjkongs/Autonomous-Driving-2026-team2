"""Exercise launch parameter merging without requiring a local ROS install."""
import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('flag, expected', [('false', False), ('true', True)])
@pytest.mark.parametrize('overrides', [{}, {'speed_limit': '1.0', 'max_acceleration': '1.5',
                                           'max_deceleration': '2.0'}])
def test_launch_flag_overrides_yaml_default(flag, expected, overrides, tmp_path):
    source = ast.parse((ROOT / 'launch/bringup_launch.py').read_text())
    function = next(n for n in source.body
                    if isinstance(n, ast.FunctionDef) and n.name == '_launch_agent')
    module = ast.Module(body=[function], type_ignores=[])

    class LaunchConfiguration:
        def __init__(self, key):
            self.key = key

        def perform(self, context):
            return context.get(self.key, '')

    namespace = {'yaml': yaml, 'LaunchConfiguration': LaunchConfiguration,
                 'Node': lambda **kwargs: kwargs}
    exec(compile(module, '<launch parameter test>', 'exec'), namespace)
    # Vehicle-specific YAML settings must survive unspecified launch overrides.
    config = tmp_path / 'agent.yaml'
    config.write_text(yaml.safe_dump({'agent_node': {'ros__parameters': {
        'name': 'imitation', 'algo_idx': 2, 'dry_run': True,
        'speed_limit': 1.7, 'max_acceleration': 2.0, 'max_deceleration': 3.0}}}))
    context = {'agent_config': str(config),
               'dry_run': flag, 'project_root': '/home/team2/project'}
    context.update(overrides)
    node = namespace['_launch_agent'](context)[0]
    assert len(node['parameters']) == 1
    parameters = node['parameters'][0]
    assert parameters['dry_run'] is expected
    assert parameters['project_root'] == context['project_root']
    assert parameters['name'] == 'imitation'
    assert parameters['algo_idx'] == 2
    assert parameters['speed_limit'] == float(overrides.get('speed_limit', 1.7))
    assert parameters['max_acceleration'] == float(overrides.get('max_acceleration', 2.0))
    assert parameters['max_deceleration'] == float(overrides.get('max_deceleration', 3.0))
