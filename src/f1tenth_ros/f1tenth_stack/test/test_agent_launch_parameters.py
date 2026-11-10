"""Exercise launch parameter merging without requiring a local ROS install."""
import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('flag, expected', [('false', False), ('true', True)])
def test_launch_flag_overrides_yaml_default(flag, expected):
    source = ast.parse((ROOT / 'launch/bringup_launch.py').read_text())
    function = next(n for n in source.body
                    if isinstance(n, ast.FunctionDef) and n.name == '_launch_agent')
    module = ast.Module(body=[function], type_ignores=[])

    class LaunchConfiguration:
        def __init__(self, key):
            self.key = key

        def perform(self, context):
            return context[self.key]

    namespace = {'yaml': yaml, 'LaunchConfiguration': LaunchConfiguration,
                 'Node': lambda **kwargs: kwargs}
    exec(compile(module, '<launch parameter test>', 'exec'), namespace)
    context = {'agent_config': str(ROOT / 'config/agent.yaml'),
               'dry_run': flag, 'project_root': '/home/team2/project'}
    node = namespace['_launch_agent'](context)[0]
    assert len(node['parameters']) == 1
    parameters = node['parameters'][0]
    assert parameters['dry_run'] is expected
    assert parameters['project_root'] == context['project_root']
    assert parameters['name'] == 'imitation'
    assert parameters['algo_idx'] == 2
    assert parameters['speed_limit'] == 1.0
