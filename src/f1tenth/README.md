# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

You can find the original [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

> Original author: *Hongrui Zheng*

> Author: *Hyeokjin Kwon, Geunje Cheon, Junseok Kim*

<!-- ## Install Docker
If you are not using Ubuntu 18.04, you have to use Docker to install Ubuntu 18.04.
If you are, skip this section.

1. Update
```bash
sudo apt update
sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
```

2. Repository key
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

3. Add apt-repository
```
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```

4. Install recent Docker
```
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io
```

5. Check docker system
```
sudo systemctl status docker

# ● docker.service - Docker Application Container Engine
#      Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset>
#      Active: active (running) since Sat 2024-05-25 18:21:35 KST; 3min 23s ago
# TriggeredBy: ● docker.socket
#        Docs: https://docs.docker.com
# ...
```

6. Make Super USER to not use sudo every time
```
sudo usermod -aG docker $USER
```
- After this, you should relogin.

7. Check example container
```
docker container run hello-world -->
<!-- ``` -->


## Quickstart
### Prepared local simulation environment (Python 3.8)

On this computer, activate the prepared environment and enter the simulation directory:

```bash
source ~/venv/torch210/bin/activate
cd /home/jiwon/projects/Autonomous-Driving-2026-team2/src/f1tenth
```

The environment uses PyTorch 2.1.0+cu118 and the local simulator installed in editable
mode. CUDA computation on the RTX 3050, simulator reset and stepping, and RGB
rendering have been verified. The virtual environment also provides `ffmpeg` using
the executable bundled with `imageio-ffmpeg`.

To reinstall the simulation dependencies into a Python 3.8 environment that already
has PyTorch 2.1 installed, run from `src/f1tenth`:

```bash
python -m pip install -r requirements-sim-py38.txt
python -m pip check
```

This file preserves Gymnasium 0.29.1 and MoviePy 1.0.3, which match the existing
wrapper and video APIs. ROS 2 and Jetson deployment require separate setup.
BC can run independently of the unfinished PPO/SAC exercises. The shared RL network
placeholders raise `NotImplementedError` when used; the RL agents still need their
exercise implementations. `AgentBase` is unchanged. BC collection converts expert
speed into the wrapper's `[-1, 1]` range using both speed limits, and uses the configured
wheelbase. Evaluation distinguishes completed laps from timeouts. The BC buffer
samples the actual dataset length, including when cached data has a different size
from `--total_steps`.

To start BC training from this directory:

```bash
python main_il.py --name imitation --algo_idx 1 --device gpu
```

A short integration run verified expert collection, two GPU training updates,
evaluation, and checkpoint/model saving. This verifies execution, not driving quality.

We recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
cd [your-virtualenv-folder]
virtualenv F1tenth --python=python3.8
source [your-virtualenv-folder]/F1tenth/bin/activate
# pip install poetry
pip install --upgrade pip

cd [your-workspace-folder]
git clone https://github.com/rllab-snu/Autonomous-Driving-Student
cd installation
pip install -e .
cd [your-workspace-folder]
pip install -r requirements.txt
```

Then you can run a quick waypoint follow example by:
```bash
cd installation/examples
python3 waypoint_follow.py --render --map Spielberg
```

## Calibration
After you complete the calibration process on the real RC car, the next step is **to ensure that the configuration settings in your simulation environment match those of the physical RC car**.

This alignment is crucial because it allows the simulation to accurately replicate the behavior of the real car, ensuring that the virtual tests and scenarios you run will closely mirror what would happen in the real world.

```bash
# calculate current max steering radius
cd installation/examples
python3 run_in_circle.py --render
```

You can change ***max_steering*** in *'configs/task/f1tenth.yaml'*.

## Map
There are 2 ways to use maps: 
1. random generate maps 
2. use f1 tracks

### Random Generation
```bash
cd [your-workspace-folder]
cd installation/examples

# you need to use different seed to make different maps.
python3 random_trackgen.py --seed 123 --name your_map_name

# or you can generate with your own points
python3 random_trackgen.py --checkpoints test.csv
```

### F1 Tracks
Use downloaded f1 tracks.
```python
MAPS = ['Sochi', 'Spa', 'Nuerburgring', 'Monza', 'Melbourne', 'Austin', 
        'Silverstone', 'Sakhir', 'IMS', 'Budapest', 'Montreal', 'Sepang', 
        'Oschersleben', 'YasMarina', 'MoscowRaceway', 'Zandvoort', 'Catalunya', 
        'BrandsHatch', 'Shanghai', 'Hockenheim', 'SaoPaulo', 'Spielberg', 'MexicoCity']
```

## Training
### Imitation Learning
The expert is Pure Pursuit. Its shared configuration is `configs/expert.yaml`.
See [전문가 랩타임 최적화 보고서](expert_optimization_report.md) for the current results,
and [전문가 수정 및 검증 보고서](expert_repair_report.md) for the initial diagnosis,
23-track completion results, and video locations.

Validate the expert without training (use the prepared environment above):

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python validate_expert.py \
  --seeds 1 124 7 42 2026 2027 31415 --workers 6 \
  --step-mode native --max-steps 20000
```

`native` uses the same physics, speed conversion, steering bounds and collision
checks as the training wrapper; progress is measured on the dense centerline
polyline instead of repeatedly minimizing distance to its spline. Use
`--step-mode wrapper` to run through the training wrapper directly.

Collect complete, collision-free laps for all 23 tracks, without training:

```bash
python collect_expert.py --output expert_data_verified.pkl --seed 73
```

Collection finishes a full round of all tracks, so `--total-steps` is a minimum,
not an exact sample count. Existing output files are never overwritten.
Each sample is associated with a track and lap through the metadata's episode
index ranges. The loader rejects old data or data collected with a different
controller, vehicle configuration, or physics integrator. The original
`expert_data.pkl` is retained and is no longer loaded automatically.

After reviewing the expert results, train a new run with:

```bash
python main_il.py --name imitation --algo_idx 3 --expert_data expert_data_verified.pkl
```

Training defaults to 5,000 minibatch updates (not 5,000 full dataset passes),
with fixed-start evaluations every 1,000 updates. Selection uses 92 episodes
(23 tracks, grid/random starts, seeds 17/19),
completion rate first and completed-lap time second. The policy receives only
one LiDAR scan; track pose is used by the evaluator to score progress.
Use a fresh `--algo_idx` for each run; existing selected models are protected.

```bash
python main_il.py --name imitation --algo_idx 3 --epochs 5000 \
  --eval_interval 1000 --eval_workers 6 --device gpu

# Separate evaluation: grid/random starts with five separate seeds.
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python evaluate_policy.py \
  --run results/imitation/2 --output results/imitation/2/evaluation_again \
  --seeds 11 53 2718 2028 65537 --workers 6 --video-seed 53
```

See [라이다 신경망 학습 결과](imitation_training_report.md) for experiment 2.

The neural network must be evaluated separately after training. Expert completion
does not itself establish learned-policy completion. These evaluations use tracks
present in the training dataset and do not measure unseen-track generalization.

Original course command:
```bash
python3 main_il.py --name imitation --algo_idx 1 --wandb  # --save_video or --live_video
```


### Reinforcement Learning
You can train your own models using RL algorithms : PPO, SAC.
```bash
python3 main_rl.py --algo ppo --name MYPPO --algo_idx 1 --wandb # --save_video or --live_video
```

### Test
You can test your model with evaluation code.
```bash
python3 main_rl.py --test --algo ppo --name MYPPO --algo_idx 1 --model_num 10000 # --save_video or --live_video
```

### Visualize
You can visualize your model's performance with loggers.
```python
def main():
        ...

        item_list = ['score']   # or other metrics
        algo_list.append({
        'name': 'ppo',
        'logs': [f'results/MYPPO/{i}' for i in range(1,4)]
        })
        algo_list.append({
        'name': 'sac',
        'logs': [f'results/MYSAC/{i}' for i in range(1,4)]
        })

        ...
```
```bash
python3 visualize.py
```

## Leaderboard
After finish your model, you can check success ratio and laptimes using your best model for sim2real.
```bash
python3 leaderboard.py --algo_name MYPPO --algo_idx 1 # --save_video or --live_video
```
