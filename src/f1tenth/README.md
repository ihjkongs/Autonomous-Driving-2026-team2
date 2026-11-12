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

학습 실험은 `results/imitation/번호_설명/`으로 관리한다.
숫자는 실험 식별자이며, 순서대로 실행해야 하는 단계가 아니다.

| 번호 | 폴더 이름 | 역할 |
| --- | --- | --- |
| 1 | `1_bc_mlp_4mps_initial_1080pts_1frame` | 초기 BC 모델 |
| 2 | `2_bc_mlp_4mps_verified_1080pts_1frame_realtest_20261110` | 검증한 전문가 데이터로 학습한 모델 **(2026년 11월 10일 실차 주행 테스트 사용)** |
| 3 | `3_bc_mlp_6mps_temporal_1080pts_4frames` | 라이다 4프레임과 차량 응답 지연을 반영한 첫 6m/s 모델 — 실차 미사용 |
| 4 | `4_bc_mlp_6mps_temporal_refined_1080pts_4frames` | 3번에서 조향 오차 가중치를 조정하고 추가 학습 — 실차 미사용 |
| 5 | `5_bc_cnn_6mps_dagger_avgpool4_270pts` | CNN·복구 데이터 보강, 각 프레임의 이웃한 방향 4개씩 평균하여 270개 사용 — 실차 미사용 |

`realtest_20261110`은 **2026년 11월 10일 실차 주행 테스트에서 사용한 모델**이라는 뜻이다.
학습 날짜나 모든 조건의 무충돌 검증 완료를 뜻하지 않는다.

`1080pts`는 **각 프레임에서 1,080개 거리값을 축소 없이 사용**한다는 뜻이다.
`1frame`은 한 시각의 스캔, `4frames`는 최근 네 시각의 스캔을 각각 유지한다는 뜻이다.
따라서 1·2번은 총 1,080개, 3·4번은 총 4,320개를 입력하며 공간 평균을 사용하지 않는다.

`bc`는 학습 방법, `mlp`·`cnn`은 모델 구조, `4mps`·`6mps`는 목표 속도 상한이다.
`temporal`은 연속 라이다 입력, `dagger`는 모델이 방문한 상태의 전문가 정답을 보강하는 방식을 뜻한다.
`avgpool4_270pts`는 각 시각의 1,080개 거리값을 이웃한 방향 4개씩 평균하여 270개로 줄인다는 뜻이다.
시간상 4프레임을 평균하는 것이 아니다. 최근 4프레임은 따로 유지하며, 이 평균 처리는 모델 파일 내부에 포함된다.
5번도 모방학습이므로 같은 `imitation/` 안에 둔다.

`--algo_idx 2`는 새 이름의 2번 폴더를 찾는다. ROS도 번호로 선택하며,
선택 폴더에 `sim2real/driving_policy.ts`가 있으면 단일 파일 실행기를 사용한다.
예전 숫자 폴더도 지원하되, 같은 번호의 폴더가 두 개 있으면 잘못된 선택을 막기 위해 오류를 낸다.
기본 실행 모델은 여전히 2번이다. 기존 Jetson 폴더는 원격에서 변경하지 않았다.
새 BC 실험에는 사용하지 않은 번호와 `--run-label`을 지정할 수 있다.
예: `--algo_idx 6 --run-label bc_mlp_4mps_new`.
학습 출력은 기본적으로 Git에서 무시한다. 2번의 배포 파일과 5번의 최종 `driving_policy.ts`만 추적 예외로 포함한다.
당시 경로가 들어 있는 원본 백업·로그는 보존하며 [이동 기록](../../reports/experiment_directory_migration_20261112.json)으로 현재 위치를 찾는다.

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
python main_il.py --name imitation --algo_idx 6 --run-label bc_mlp_4mps_new --expert_data expert_data_verified.pkl
```

Training defaults to 5,000 minibatch updates (not 5,000 full dataset passes),
with fixed-start evaluations every 1,000 updates. Selection uses 92 episodes
(23 tracks, grid/random starts, seeds 17/19),
completion rate first and completed-lap time second. The policy receives only
one LiDAR scan; track pose is used by the evaluator to score progress.
Use a fresh `--algo_idx` for each run; existing selected models are protected.

```bash
python main_il.py --name imitation --algo_idx 6 --run-label bc_mlp_4mps_new --epochs 5000 \
  --eval_interval 1000 --eval_workers 6 --device gpu

# Separate evaluation: grid/random starts with five separate seeds.
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python evaluate_policy.py \
  --run results/imitation/2_bc_mlp_4mps_verified_1080pts_1frame_realtest_20261110 --output results/imitation/2_bc_mlp_4mps_verified_1080pts_1frame_realtest_20261110/evaluation_again \
  --seeds 11 53 2718 2028 65537 --workers 6 --video-seed 53
```

See [라이다 신경망 학습 결과](imitation_training_report.md) for experiment 2.

### 2026-11-12: 실차 측정 없이 진행하는 6m/s 실험

`configs/experiment_6mps.yaml`은 별도 선택하는 시뮬레이션 실험이다.
기존 4m/s 기본 설정과 Jetson의 모델 2번을 덮어쓰지 않는다.
최고 목표 속도는 6m/s이며, 코너에서는 경로 곡률과 제동거리를 고려해
미리 감속한다. 실차 가감속 성능을 측정한 결과로 간주하면 안 된다.

실험에는 목표 속도 변화 제한(가속 2, 감속 3m/s²), 명령 전달 지연,
목표 속도에 대한 1차 응답, 서로 다른 가속·제동 한계가 포함된다.
가속·감속 한계는 실제 차량의 속도 변화에 대한 가정이고, 목표 속도 변화
제한은 소프트웨어 명령 처리이므로 서로 구분한다.
기존 `a_max: 9.51`만 낮추는 방식과 달리 가속·제동 성능을 따로 설정한다.
전류·배터리·회생 손실·보호 정지·타이어 종방향 미끄러짐은 이 모델에 없다.
전류 제한 때문에 실제 차량이 정지하는 현상을 해결했다고 주장할 수 없다.

| 실험 파일 | 가속 한계 | 감속 한계 | 응답 시정수 | 지연 |
| --- | --- | --- | --- | --- |
| `experiment_6mps.yaml` | 2m/s² | 3m/s² | 0.25초 | 0.10초 |
| `experiment_6mps_stress.yaml` | 1.5m/s² | 2m/s² | 0.40초 | 0.20초 |
| `experiment_6mps_randomized.yaml` | 1.5–2.5m/s² | 2–3.5m/s² | 0.15–0.40초 | 0.05–0.20초 |

무작위 실험은 에피소드마다 값을 뽑고 실제 사용한 값을 메타데이터에 저장한다.
지연은 시뮬레이션 시간 간격 0.025초 단위로 올림한다.
이 범위는 실측으로 식별한 신뢰구간이 아닌 가정한 시험 범위다.

모델 입력은 라이다 4프레임(4×1080=4320개 값), 과거→현재 순서다.
프레임 간격은 0.025초, 처음과 마지막 프레임 간격은 0.075초다.
초기화 시 첫 프레임으로 채우고, 에피소드 사이의 기록은 섞지 않는다.
위치·트랙 이름·실제 속도는 신경망 입력에 넣지 않는다.
전문가는 경로와 속도를 사용해 정답을 만든다.
시간 정보를 추가해도 모든 환경에서 속도가 유일하게 추정된다는 보장은 없다.

아래 명령은 `src/f1tenth`에서 준비된 가상환경으로 실행한다.
출력 경로가 이미 존재하면 다른 이름을 사용한다.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy

# 전문가: 각 조건에서 23트랙 × 2시작 방식 × 2시드 검증.
python validate_expert.py --experiment configs/experiment_6mps.yaml \
  --seeds 17 19 --workers 6 --step-mode native --max-steps 20000
python validate_expert.py --experiment configs/experiment_6mps_stress.yaml \
  --seeds 53 2718 --workers 6 --step-mode native --max-steps 20000
python validate_expert.py --experiment configs/experiment_6mps_randomized.yaml \
  --seeds 11 65537 --workers 6 --step-mode native --max-steps 20000

# 전체 트랙을 한 묶음씩 끝내므로 최종 샘플 수는 100000보다 클 수 있다.
python collect_expert.py --experiment configs/experiment_6mps_randomized.yaml \
  --step-mode native --output expert_data_6mps_randomized_strict_20261112.pkl \
  --total-steps 100000 --seed 73

# 새 실험 번호를 사용. 기존 데이터는 새 소스 서명과 호환되지 않는다.
python main_il.py --experiment configs/experiment_6mps_randomized.yaml \
  --algo_idx 5 --expert_data expert_data_6mps_randomized_strict_20261112.pkl \
  --epochs 5000 --eval_interval 1000 --eval_workers 6 --device gpu

python evaluate_policy.py --run results/imitation/4_bc_mlp_6mps_temporal_refined_1080pts_4frames \
  --output results/imitation/4_bc_mlp_6mps_temporal_refined_1080pts_4frames/evaluation_strict_independent_again_20261112 \
  --seeds 53 2718 2028 --workers 6

# 기존 rosbag에서 추출한 CSV를 사용한 제동·회생 재분석.
python analyze_braking.py --recordings-parent /home/jiwon/projects \
  --output ../../reports/braking_review_20261112.json
```

`native` 수집은 동일한 물리·명령 변환·라이다 이력을 사용하고,
진행 거리를 중심선 폴리라인으로 계산한다. 보상과 스플라인 좌표 최적화는 생략한다.
통합 테스트에서 wrapper 수집과 관측·정답 배열이 동일한지 확인한다.
정상 완주한 에피소드만 저장하며 기존 데이터 파일은 덮어쓰지 않는다.

새 실험의 모델은 기존 ROS 노드의 1080개 입력과 호환되지 않는다.
현재 Jetson용 모델로 교체하려면 시간 이력 처리·정규화·속도 범위까지
맞춰야 하므로 이 실험만으로 실차 배포가 완료되는 것은 아니다.
전문가 완주율과 신경망 완주율은 별도로 확인한다.
23개 트랙을 모두 학습에 사용한 평가는 처음 보는 트랙에 대한 일반화 검증이 아니다.

진행 결과는 [전체 실험 기록](../../reports/sim6mps_20261112.json),
[기존 rosbag 제동 분석](../../reports/braking_review_20261112.json),
[실험 결과 그림](../../reports/figures/sim6mps_20261112.png)에 저장한다.
보고서는 `python summarize_sim6mps.py --runs 3 4`로 다시 생성할 수 있다.

기본 5,000회 학습(실험 3)은 선택 평가에서 32/92회 완주했다.
추가 학습은 별도 실험 4에서 실험 3의 선택된 가중치와 Adam 상태를 이어받고,
정규화와 데이터 서명이 같은지 확인한다. 조향 오차 가중치를 3으로 높이므로
두 실험의 학습 손실 숫자는 직접 비교하면 안 된다. 주행 완주율로 비교한다.
새 실험 번호를 사용하며 이전 실험의 가중치는 덮어쓰지 않는다.

실험 4의 실행 설정은 `--resume-from results/imitation/3_bc_mlp_6mps_temporal_1080pts_4frames`,
`--steering-loss-weight 3`, `--epochs 10000`, `--eval_interval 2500`이었다.
이 실행은 아래 충돌 판정 수정 이전의 데이터 서명을 사용했다.
현재 코드로 새 학습을 시작할 때는 위의 strict 데이터와 새로운 실험 번호를 사용한다.
이전 서명의 데이터를 강제로 최신 서명으로 바꾸면 안 된다.

추가 학습의 `best_update`는 해당 실험 안에서의 학습 횟수이며,
`0`이면 이전 실험의 선택 모델이 유지됐다는 뜻이다.

추가 분석에서 시뮬레이터 충돌 판정도 수정했다.
기존에는 벽이 차체 안으로 들어와 TTC가 음수가 되면 충돌을 놓칠 수 있었고,
벽까지의 차체 경계 계산에 실제 차체 길이 0.57m 대신 축간거리 0.35m를 썼다.
이제 차체 내부의 장애물은 정지 상태에서도 충돌로 판정하고,
각 차량의 길이·너비·라이다 위치로 경계를 계산한다.
접근 중인 벽의 TTC 판정 시간은 적어도 한 시뮬레이션 스텝을 포함한다.
이 판정은 여전히 라이다와 TTC를 사용하며 완전한 연속 차체 충돌 해석은 아니다.

수정 후 nominal/stress/randomized 전문가 검증을 각각 92회 다시 수행해
276/276회 완주했다. `strict`라는 이름이 붙은 결과가 수정 후의 판정이다.
학습 중 선택 평가 86/92회는 수정 전 판정의 결과이며,
현재 성능은 실험 4의 `evaluation_strict_independent_20261112` 결과로 판단한다.

전문가 데이터도 수정된 판정으로 46랩을 다시 수집했다.
[데이터 대조 기록](../../reports/sim6mps_dataset_equivalence_20261112.json)은
이전 데이터와 수정 후 데이터의 관측·정답 배열 및 SHA256을 비교한다.
데이터 서명은 바뀌어도 실제 학습 배열이 같을 수 있다.
기존 학습 기록은 보존하고 수정 후 별도 평가로 모델을 재검증한다.

[실패 구간 비교](../../reports/policy_failure_diagnostics_strict_20261112.json)는
학습 모델만으로 운전하면서 같은 상태에서 전문가가 요구한 명령을 비교한 결과다.
전문가 명령은 차량 제어에 개입하지 않는다.
기존 단일 차량 reset은 중심선에서 시작하므로, 이번 데이터에는
의도적으로 옆으로 밀리거나 방향이 틀어진 상태의 복귀 사례가 충분히 보장되지 않는다.
이탈 상태의 복구 데이터 보강은 다음 개선 대상으로 남는다.

최종 수정 판정으로 수행한 별도 평가는 **133/138회 완주(96.38%), 5회 충돌**이었다.
전문가가 완주했다고 학습 모델도 완벽하게 주행하는 것은 아니다.
이 실험의 모델 4는 실차용으로 교체하지 않았으며, 현재 Jetson 모델 2와 설정은 유지한다.
시뮬레이션·수집 동등성·학습 재개·충돌 판정 관련 테스트는 29개 통과했다.

The neural network must be evaluated separately after training. Expert completion
does not itself establish learned-policy completion. These evaluations use tracks
present in the training dataset and do not measure unseen-track generalization.


### 모델 단독 주행을 위한 복구 학습과 미학습 트랙 평가 (2026-11-12)

이전 MLP 모델의 충돌을 줄이기 위해 `train_recovery.py`에서 새 CNN을 처음부터 학습한다.
입력은 오래된 순서대로 나열한 라이다 4프레임(1080 × 4)이며,
위치·지도 이름·중심선·전문가 명령을 모델에 넣지 않는다.
차량의 지연과 가감속 한계를 에피소드마다 바꾸는 6 m/s 실험 환경을 사용한다.
6 m/s는 직선에서 허용하는 목표 속도 상한이며 모든 코너에서 유지할 속도가 아니다.

23개 트랙 중 18개에서만 학습한다. Spa, Sakhir, Sepang, MoscowRaceway,
Spielberg는 학습 데이터·복구 수집·체크포인트 선택에서 모두 제외한다.
정규화도 전체 트랙의 평균·분산 대신 모델 내부의 고정된 물리 단위 변환을 사용한다.
처음 학습하는 사례는 126,456개이며 좌우 반전과 작은 라이다 잡음을 추가한다.

완벽한 전문가 주행만 복사하면 모델이 스스로 벗어난 상태의 복구 방법을 배우기 어렵다.
그래서 `utils/recovery_collection.py`는 시작 위치를 좌우 0.25 m,
방향을 0.10 rad 범위에서 흔들고 모델이 방문한 상태에서 전문가 정답을 추가 수집한다.
수집 중에는 초기 안정화·확률적 전문가 개입·이탈 시 개입이 있지만,
평가에서는 이 전문가를 호출하거나 조작을 대신하지 않는다.
일부 수집이 실패해도 실패를 완주로 기록하지 않고, 그 이전의 복구 학습 사례로 구분한다.

기본 학습 8,000회 뒤 4,000회씩 세 차례 보강했다.
개발 평가는 18개 학습 트랙의 두 시작 방식에서 실시한다.
완주율을 먼저 비교하고 동률일 때 랩타임을 비교해 두 번째 보강 모델을 선택했다.
선택된 모델은 기본 사례와 복구 사례 90,268개를 사용했다.
마지막 보강까지 수집한 총 131,066개 전체를 선택 모델이 사용한 것은 아니다.
고정한 모델은 별도 시작 시드로 평가하며, 새로 만든 세 트랙도 평가 전용이다.

| 검증 | 모델 단독 완주 | 충돌 |
| --- | ---: | ---: |
| 기존 23개 트랙 × 두 시작 방식 × 세 시드 | 138/138회 | 0 |
| 위 결과 중 학습에서 제외한 5개 트랙 | 30/30회 | 0 |
| 새 형상의 3개 트랙 × 두 시작 방식 × 두 시드 | 12/12회 | 0 |
| 내보낸 파일: 23개 트랙 × 두 시작 방식, 각각 연속 3바퀴 | 46/46회, 총 138바퀴 | 0 |

개발 평가를 제외한 총 196회 시험을 모두 통과했다.
연속 3바퀴 시험은 차량 상태와 라이다 이력을 매 바퀴 초기화하지 않는다.
평가는 CPU에서 수행했고, 내보낸 파일은 별도로 CPU/CUDA 출력도 대조했다.
최고 관측 속도는 약 5.99 m/s였다. 이전 MLP의 133/138회와는 시드와
학습·평가 절차가 달라 동일 조건의 랩타임 비교로 해석하지 않는다.
자세한 판정·에피소드·파일 해시는 [검증 기록](../../reports/recovery_policy_20261112.json)을 참조한다.

학습과 모델 선택을 다시 실행하려면 `src/f1tenth`에서 실행한다.
새 출력 디렉터리를 지정해야 기존 기록을 덮어쓰지 않는다.

```bash
python train_recovery.py --output results/imitation/7_bc_cnn_6mps_dagger_new \
  --initial-updates 8000 --round-updates 4000 --rounds 3 --workers 6
python export_driving_policy.py --run results/imitation/7_bc_cnn_6mps_dagger_new
python evaluate_policy.py --run results/imitation/7_bc_cnn_6mps_dagger_new \
  --output results/imitation/7_bc_cnn_6mps_dagger_new/check_3laps \
  --standalone --laps 3 --seeds 18419 --workers 6
```

배포용 파일은 `results/imitation/5_bc_cnn_6mps_dagger_avgpool4_270pts/sim2real/driving_policy.ts`다.
CNN 가중치, 입력 전처리, 조향(rad)·목표 속도(m/s) 출력 변환, 입력 규격을 담는다.
실행 시에는 차체를 제어하는 ROS/VESC 소프트웨어와 차량 보정값은 필요하지만,
이 모델을 위해 트랙별 YAML·전문가 경로·별도 정규화 pickle을 복사할 필요는 없다.
이 최종 파일은 `.gitignore` 예외로 Git에 포함한다. 코드와 함께 커밋·푸시한 후
Jetson에서 `git pull`하면 같은 모델 파일을 받으므로 별도 `scp` 전달이 필요 없다.
중간 체크포인트·학습 데이터·5번의 Python pickle 모델은 계속 Git에서 제외한다.

ROS 실행기는 `policy_file:=/절대경로/driving_policy.ts` 인자로 새 파일을 선택한다.
Jetson 프로젝트 위치가 기존과 같다면 다음 경로를 지정한다.

```text
policy_file:=/home/team2/Autonomous-Driving-2026-team2/src/f1tenth/results/imitation/5_bc_cnn_6mps_dagger_avgpool4_270pts/sim2real/driving_policy.ts
```

연속 스캔을 쌓고 스캔이 끊기면 이력을 초기화한다.
기본 모델 2, `dry_run` 기본값, 속도 상한, 가감속 제한, watchdog은 유지한다.
자동으로 Jetson에 배포하거나 실차 주행을 시작하지 않는다.
시뮬레이션의 응답 지연·감속 범위는 실측 보정값이 아니며,
전류 보호·배터리 회생 한계까지 검증한 결과도 아니다.
미학습 트랙 시험은 일반화의 근거이지만 모든 가능한 트랙의 무충돌 보장은 아니다.

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
