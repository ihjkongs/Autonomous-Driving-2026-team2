"""Create a reproducible offline experiment report and figure from saved results."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from utils.expert_config import ROOT


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def expert_reports(validation, strict=False):
    experiments = {}
    for name in ('nominal', 'stress', 'randomized', 'no_preview'):
        directory = validation / ('sim6mps_' + ('strict_' if strict else '') + name + '_20261112')
        summary = read_json(directory/'summary.json')
        if summary is None:
            continue
        records = [json.loads(s) for s in (directory/'episodes.jsonl').read_text().splitlines()]
        completed = [r for r in records if r['outcome'] == 'complete']
        experiments[name] = dict(summary=summary, metadata=read_json(directory/'metadata.json'),
            mean_completed_lap_s=float(np.mean([r['elapsed_s'] for r in completed])) if completed else None,
            completed=len(completed),
            lowest_episode_peak_actual_speed_mps=min(r['max_actual_speed_mps'] for r in records),
            episodes=[{k: v for k, v in r.items() if k not in ('first_commands', 'at_collection_horizon')} for r in records])
    return experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=int, default=[3])
    parser.add_argument('--output', type=Path, default=ROOT.parents[1]/'reports/sim6mps_20261112.json')
    cli = parser.parse_args()
    validation = ROOT / 'results/expert_validation'
    legacy_experiments = expert_reports(validation)
    experiments = expert_reports(validation, strict=True)
    runs = {}
    for number in cli.runs:
        from utils.run_paths import experiment_path
        directory = experiment_path(ROOT/'results', 'imitation', number)
        candidates = []
        for path in sorted((directory/'selection').glob('*/summary.json'), key=lambda p:int(p.parent.name)):
            candidates.append(dict(update=int(path.parent.name), summary=read_json(path)))
        independent = {path.name: read_json(path/'summary.json')
                       for path in directory.glob('evaluation_*20261112*') if (path/'summary.json').exists()}
        runs[str(number)] = dict(summary=read_json(directory/'training_summary.json'),
            checkpoint_selection_collision_model='Legacy collision checker before the body/overlap fixes; use strict independent evaluation for current validity.',
            effective_experiment=read_json(directory/'backup/effective_experiment.json'),
            selection_candidates=candidates, independent_evaluations=independent,
            model_path=str(directory/'sim2real/actor.pt'),
            sha256=hashlib.sha256((directory/'sim2real/actor.pt').read_bytes()).hexdigest()
                   if (directory/'sim2real/actor.pt').exists() else None)
    deployed = {}
    repo = ROOT.parents[1]
    from utils.run_paths import experiment_path
    deployed_run = experiment_path(ROOT/'results', 'imitation', 2)
    for relative in [str((deployed_run/'sim2real/actor.pt').relative_to(repo)),
                     str((deployed_run/'sim2real/agent_obs_scale.pkl').relative_to(repo)),
                     'src/f1tenth_ros/f1tenth_stack/config/agent.yaml',
                     'src/f1tenth_ros/f1tenth_stack/config/vesc.yaml']:
        raw = (repo/relative).read_bytes()
        committed_path = relative.replace('/2_bc_mlp_4mps_verified_1080pts_1frame_realtest_20261110/', '/2/')
        exists_in_head = subprocess.run(['git','cat-file','-e','HEAD:'+relative], cwd=repo,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        committed = subprocess.check_output(['git','show','HEAD:'+(relative if exists_in_head else committed_path)], cwd=repo)
        deployed[relative] = dict(sha256=hashlib.sha256(raw).hexdigest(), unchanged_from_HEAD=raw==committed)
    matched = None
    if 'nominal' in experiments and 'no_preview' in experiments:
        nominal = {(r['map'], r['reset'], r['seed']): r for r in experiments['nominal']['episodes']}
        pairs = [(nominal[(r['map'], r['reset'], r['seed'])], r)
                 for r in experiments['no_preview']['episodes']]
        matched = dict(episodes=len(pairs),
            identical_initial_poses=all(a['initial_pose'] == b['initial_pose'] for a, b in pairs),
            preview_enabled_completed=sum(a['outcome'] == 'complete' for a, b in pairs),
            preview_disabled_completed=sum(b['outcome'] == 'complete' for a, b in pairs),
            preview_disabled_failures=[dict(map=b['map'], reset=b['reset'], outcome=b['outcome'])
                                      for a, b in pairs if b['outcome'] != 'complete'])
    result = dict(date='2026-11-12', generated_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        scope='Offline simulation and previously recorded data only; no new real-vehicle measurements or deployment.',
        assumptions_not_calibrated=True, electrical_faults_simulated=False,
        observation='4 chronological LiDAR frames, 4320 ranges, 25ms sampling, 75ms history span; reset repeats initial frame.',
        checks=dict(simulation_unit_and_integration_tests_passed=29,
                    test_command='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy python -m pytest tests installation/tests/test_dynamics.py -q',
                    scope='Simulation, data collection equivalence and resume validation; no new ROS/hardware integration test.'),
        expert_experiments=experiments, matched_preview_ablation=matched,
        expert_experiments_before_collision_fix=legacy_experiments,
        dataset=read_json(ROOT/'expert_data_6mps_randomized_strict_20261112.metadata.json'),
        dataset_requalification=read_json(repo/'reports/sim6mps_dataset_equivalence_20261112.json'),
        collision_fixes=[
            'Wall samples inside the vehicle footprint now count as collision, including at zero speed.',
            'Ray distances use physical body length 0.57m instead of wheelbase 0.35m, with per-car LiDAR offset.',
            'Approaching obstacles use a TTC horizon of at least one simulator timestep; receding obstacles are excluded.'
        ],
        strict_policy_diagnostics=read_json(repo/'reports/policy_failure_diagnostics_strict_20261112.json'),
        training_runs=runs, deployed_artifacts=deployed,
        limitations=[
            'Response limits are assumed test cases, not estimates of physical braking capability.',
            'No VESC electrical current loop, charge-current limit, fault trip or longitudinal tire slip is simulated.',
            'Old rosbag averages cannot identify instantaneous overcurrent peaks or independent ground speed.',
            'Expert and policy completion are separate results; all 23 evaluation tracks are training tracks.',
            'Four-frame models are incompatible with the unchanged 1080-input ROS deployment node.',
            'No obstacle-stop policy, sensor timing jitter robustness or physical 6m/s operation is established.'
        ])
    primary = runs.get('4', {}).get('independent_evaluations', {}).get('evaluation_strict_independent_20261112')
    result['current_result'] = dict(
        expert_completed=sum(experiments[name]['completed'] for name in ('nominal', 'stress', 'randomized')),
        expert_episodes=sum(experiments[name]['summary']['episodes'] for name in ('nominal', 'stress', 'randomized')),
        policy_strict_independent=primary,
        policy_passed_all_evaluation_episodes=bool(primary and primary['completed'] == primary['episodes']),
        physical_6mps_validated=False,
        next_work='Improve imitation on failed corners and collect recovery states; retain existing Jetson model.')
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    names = list(experiments)
    rates = [experiments[name]['completed']/experiments[name]['summary']['episodes'] for name in names]
    axes[0].bar(names, rates)
    for i, name in enumerate(names):
        e = experiments[name]
        axes[0].text(i, rates[i]+.02, str(e['completed'])+'/'+str(e['summary']['episodes']), ha='center')
    axes[0].set(ylim=(0, 1.15), ylabel='Expert completion rate', title='Assumed response scenarios; all 23 tracks')
    trace_path = validation/'sim6mps_strict_nominal_20261112/trajectories/Shanghai_cl_grid_static_11.npz'
    trace = np.load(trace_path)['samples']
    axes[1].plot(trace[:, 0]*.025, trace[:, 4], label='Expert requested speed')
    axes[1].plot(trace[:, 0]*.025, trace[:, 6], label='Simulated vehicle speed')
    axes[1].set(xlabel='Simulation time (s)', ylabel='Speed (m/s)', title='Shanghai: bounded delayed response (not calibrated to the real car)')
    axes[1].legend()
    for number, run in runs.items():
        points = run['selection_candidates']
        axes[2].plot([p['update'] for p in points], [p['summary']['completion_rate'] for p in points],
                     marker='o', label='Run '+number+' (updates within this run)')
    axes[2].set(xlabel='Training updates', ylabel='Policy completion rate', ylim=(0, 1.05),
                title='Policy selection before collision fix (92 episodes); strict final evaluation is in the report')
    axes[2].legend()
    figure = cli.output.parent/'figures/sim6mps_20261112.png'
    figure.parent.mkdir(exist_ok=True)
    fig.savefig(figure, dpi=140)
    plt.close(fig)
    print(json.dumps(dict(report=str(cli.output), figure=str(figure))))


if __name__ == '__main__':
    main()
