"""Checkpoint selection must prefer completion over survival or speed."""
from utils.policy_evaluation import summarize, selection_key


def test_completion_wins_over_fast_partial_drive():
    success=summarize([dict(outcome='complete',elapsed_s=300.,progress_fraction=1.),
                       dict(outcome='collision',elapsed_s=1.,progress_fraction=.01)])
    failure=summarize([dict(outcome='timeout',elapsed_s=500.,progress_fraction=.99)]*2)
    assert selection_key(success)>selection_key(failure)


def test_same_completion_rate_prefers_shorter_lap():
    fast=summarize([dict(outcome='complete',elapsed_s=100.,progress_fraction=1.)])
    slow=summarize([dict(outcome='complete',elapsed_s=200.,progress_fraction=1.)])
    assert selection_key(fast)>selection_key(slow)


def test_no_completions_uses_progress_not_survival_time():
    near=summarize([dict(outcome='collision',elapsed_s=10.,progress_fraction=.8)])
    far=summarize([dict(outcome='timeout',elapsed_s=500.,progress_fraction=.1)])
    assert selection_key(near)>selection_key(far)
    assert near['mean_lap_s'] is None


def test_continuous_laps_report_lap_time_separately_from_episode_time():
    result = summarize([dict(outcome='complete', elapsed_s=270.,
                             required_laps=3, progress_fraction=1.)])
    assert result['mean_episode_s'] == 270.
    assert result['mean_lap_s'] == 90.
