from pathlib import Path
import pytest
from utils.run_paths import experiment_path


def test_number_selects_descriptive_directory_without_creating_numeric_copy(tmp_path):
    expected = tmp_path/'imitation/2_bc_mlp_4mps_verified_1080pts_1frame_realtest_20261110'
    expected.mkdir(parents=True)
    assert experiment_path(tmp_path, 'imitation', 2) == expected
    assert not (tmp_path/'imitation/2').exists()


def test_legacy_numeric_directory_remains_supported(tmp_path):
    expected = tmp_path/'imitation/2'
    expected.mkdir(parents=True)
    assert experiment_path(tmp_path, 'imitation', 2) == expected


def test_ambiguous_experiment_is_rejected(tmp_path):
    for name in ['2', '2_new']:
        (tmp_path/'imitation'/name).mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match='Multiple'):
        experiment_path(tmp_path, 'imitation', 2)


def test_new_experiment_label_and_other_algorithms(tmp_path):
    assert experiment_path(tmp_path, 'imitation', 6, 'bc_mlp_6mps') == tmp_path/'imitation/6_bc_mlp_6mps'
    assert experiment_path(tmp_path, 'ppo', 1) == tmp_path/'ppo/1'
    with pytest.raises(ValueError, match='label'):
        experiment_path(tmp_path, 'imitation', 6, '../escape')
