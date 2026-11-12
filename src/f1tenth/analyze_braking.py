"""Re-analyze exported rosbag CSVs without claiming calibrated ground speed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def analyze(directory):
    state_path = directory / 'sensors_core.csv'
    state = pd.read_csv(state_path)
    command = pd.read_csv(directory / 'commands_motor_speed.csv')
    prediction = pd.read_csv(directory / 'agent_prediction.csv')
    result = dict(recording=directory.name.removesuffix('_analysis') if hasattr(str, 'removesuffix')
                  else directory.name[:-9], samples=len(state),
                  source_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
                  state_interval_s=[float(state.t.iloc[0]), float(state.t.iloc[-1])],
                  command_interval_s=[float(command.t.iloc[0]), float(command.t.iloc[-1])],
                  telemetry_end_before_commands_s=float(command.t.iloc[-1]-state.t.iloc[-1]),
                  motor_current_range_A=[float(state.current_motor.min()), float(state.current_motor.max())],
                  battery_current_range_A=[float(state.current_input.min()), float(state.current_input.max())],
                  regen_Ah_increase=float(state.charge_regen.iloc[-1]-state.charge_regen.iloc[0]),
                  regen_Wh_increase=float(state.energy_regen.iloc[-1]-state.energy_regen.iloc[0]),
                  negative_battery_samples=int((state.current_input < -.1).sum()))
    faults = state.fault_code.to_numpy()
    onsets = np.flatnonzero((faults != 0) & np.r_[True, faults[:-1] == 0])
    result['faults'] = []
    for i in onsets:
        t = state.t.iloc[i]
        window = state[(state.t >= t-.25) & (state.t <= t)]
        cmd = command[(command.t >= t-.25) & (command.t <= t)]
        pred = prediction[(prediction.t >= t-.25) & (prediction.t <= t)]
        result['faults'].append(dict(t_s=float(t), code=int(faults[i]),
            preceding_250ms_motor_min_A=float(window.current_motor.min()),
            preceding_250ms_battery_min_A=float(window.current_input.min()),
            preceding_250ms_command_erpm_span=float(cmd.erpm_cmd.max()-cmd.erpm_cmd.min()) if len(cmd) else None,
            preceding_250ms_prediction_speed_span=float(pred.speed.max()-pred.speed.min()) if len(pred) else None))
    good = state[(state.fault_code == 0) & (state.current_input < -.1)]
    if len(good):
        sample = good.loc[good.current_input.idxmin()]
        preceding = command[command.t <= sample.t]
        result['example_regeneration'] = {key: float(sample[key]) for key in
            ['t', 'erpm', 'current_motor', 'current_input', 'voltage_input']}
        result['example_regeneration']['preceding_target_erpm'] = float(preceding.erpm_cmd.iloc[-1]) if len(preceding) else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings-parent', type=Path, default=Path('/home/jiwon/projects'))
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    dirs = sorted(cli.recordings_parent.glob('drive_*_analysis'))
    records = [analyze(path) for path in dirs if (path/'sensors_core.csv').exists()]
    result = dict(ground_speed_calibrated=False, observations=records,
        interpretation=[
            'Negative battery current plus increasing charged energy supports regeneration at the VESC DC input.',
            'Motor/input currents are packet averages and do not reconstruct instantaneous overcurrent peaks.',
            'Command and state packet receipt times are asynchronous, not exact controller execution times.',
            'ERPM response is observable; true ground speed, slip and physical deceleration are not identified.',
            'Fault-free received samples do not establish fault-free operation after telemetry stops.',
            'These non-controlled laps do not uniquely identify delay, PID gains or motor/battery limits.'
        ])
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(recordings=len(records), output=str(cli.output))))


if __name__ == '__main__':
    main()
