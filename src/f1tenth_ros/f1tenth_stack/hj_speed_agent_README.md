# hj_speed_agent (4321 입력: 4 scan + odom 차속) 실차 체크리스트

기존 `agent_node`/`agent.yaml`/`bringup_launch.py`는 수정하지 않았다. 새 노드 `hj_speed_agent_node`는 `/scan`+`/odom`을 받아 `[4프레임×1080, 현재차속]`을 만들고 68(또는 4321 TorchScript) 정책을 돌린다. dry_run 기본 true.

## 실행
- bringup은 `enable_agent:=false`로 띄운다(기존 agent_node와 동시에 /drive 발행 금지).
- `ros2 launch f1tenth_stack hj_speed_agent_launch.py policy_file:=<.../68_.../sim2real/driving_policy.ts>` (기본 dry_run:=true, speed_limit 1.0).
- setup.py는 수정하지 않았다(launch가 `python -m f1tenth_stack.hj_speed_agent_node` 실행). 새 launch/config는 data_files glob으로 설치되므로 `colcon build` 필요.

## 확인 순서 (앞 단계 통과 전 다음 단계 금지)
1. 바퀴 띄우거나 정지 상태, dry_run:=true. `ros2 topic hz /odom`, `/scan` 주기 확인. `/odom` 주기가 odom_timeout(0.25s) 안에 여러 번 들어와야 한다(수십 Hz 기대). VESC가 /sensors/core를 계속 내는지 확인.
2. 정지 상태에서 `ros2 topic echo /odom --field twist.twist.linear.x` ≈ 0, 로그의 `odom_v`와 일치. 손으로 차를 앞으로 밀 때 odom_v가 **양수**여야 한다. 음수면 부호/모터 방향 설정(vesc.yaml speed_to_erpm_gain 부호 등) 문제이며 노드는 -0.1 미만이면 정지(Rejected)한다.
3. 속도 보정: `vesc_to_odom`은 ERPM 변환 gain(vesc.yaml, 현재 2500 offset -325 근처)으로 v를 계산하고 |v|<0.05는 0으로 만든다. 실제 바닥 속도(줄자+시간, 또는 랩 시간)와 odom_v를 낮은 속도에서 비교해 gain을 보정한다. 정책 입력은 이 odom 값을 그대로 쓰므로 오차는 입력 분포 이동이다(시뮬은 hypot(vx,vy) 실제속도, 0..6).
4. 스캔-odom 시각 정렬: 노드는 두 header stamp가 같은 ROS clock일 때만 쓴다(use_header_stamps). 둘 중 stamp 0이면 거부. `odom_max_skew`(0.1s) 초과면 정지. 실제 skew 분포를 로그/`ros2 topic echo --field header.stamp`로 확인해 임계 조정. LiDAR 자체 지연(시뮬 delay100ms 시나리오 참고)은 stamp로 보정되지 않는다.
5. `/agent/prediction`, `/agent/command_preview` 확인(조향 부호, 속도가 speed_limit로 제한되는지, 정지 시 0). odom 케이블 분리/`/odom` 발행 중단 시 0.25s 내 command 0이 나가는지 확인(dry_run에서도 stop은 preview에 발행).
6. 위가 모두 정상일 때만 낮은 speed_limit(예 0.5~1.0)로 `dry_run:=false`. 발진은 slew 제한(가속 2.0) 때문에 첫 사이클 0.
7. 이후 speed_limit를 단계적으로 올린다(정책 native 0..8, 상한 6은 TorchScript 내 cap).

## 동작 요약 (안전 우선)
- 정지(속도0, 이력 reset): odom 미수신/비유한/수신 후 0.25s 경과/스캔-odom skew>0.1s/linear.x<-0.1/>8/스캔 무효/추론 시간초과/예외.
- 부호: 시뮬 차속은 hypot≥0이므로 |linear.x|를 쓰되(−0.1 이내 노이즈만), 그보다 큰 음수는 거부. 데드밴드: vesc_to_odom이 이미 0.05 미만을 0으로 만들며 추가 처리 없음.
- LiDAR 추종 guard: `SpeedAgentCore.guard`(기본 no_guard)가 speed_limit 적용 후·speed_limiter 전에 호출되는 reduce-only hook이다. A9/V2f가 연결되어 있다(아래).

## LiDAR 추종 guard A9 (선택, 기본 OFF)
- 파일: `f1tenth_stack/hj_lidar_following_guard_a9.py`(원본 `src/f1tenth/hj_lidar_following_guard.py`의 **바이트 복사**, SHA 731d2786…), `config/hj_lidar_guard_a9_params.yaml`(동결 `params_A9.json` 바이트 복사, JSON 내용, SHA 3767b3b7…). 시작 시 두 SHA를 확인하고 불일치하면 노드가 시작되지 않는다.
- 원본이 import하는 `algorithm.imitation.highspeed_lidar`, `utils.traffic_expert.safe_following_speed`는 `hj_guard_shims.py`가 로드 시점에만 대신 제공한다(src/f1tenth 불필요; 함수는 원본과 수치 동일, 테스트로 확인). Jetson에 torch는 필요하다(원본이 import).
- 켜기: `ros2 launch f1tenth_stack hj_speed_agent_launch.py policy_file:=... guard_enabled:=true` (또는 yaml `guard_enabled: true`). 먼저 dry_run:=true에서 `/agent/command_preview` 속도가 앞차/장애물 앞에서 줄어드는지 확인. 선택: `guard_variant:=a9|v2f|custom`(launch 인자 또는 yaml). a9/v2f는 파일·SHA·params가 패키지에 고정되어 있고 `guard_file`/`guard_sha256`/`guard_params_file`/`guard_params_sha256`이 빈 문자열이면 variant 값을 쓴다. custom은 4개 모두 지정해야 한다(같은 `guard_speed`/`new_memory` 인터페이스 필요).

## 시스템 71 = BC68 + guard V2f
- 파일: `f1tenth_stack/hj_lidar_following_guard_v2f.py`(원본 `src/f1tenth/hj_lidar_following_guard_v2.py`의 바이트 복사, SHA 234aa1b8...), `config/hj_lidar_guard_v2f_params.yaml`(동결 `frozen_V2f/guarded_V2f_params.json` 바이트 복사, JSON 내용, SHA a44923a9...). A9 파일은 그대로 보존.
- 권장 설정: `config/hj_speed_agent_71.yaml`(guard_enabled true, guard_variant v2f, SHA 명시 고정, dry_run true, speed_limit 1.0). `hj_agent_config:=$(ros2 pkg prefix f1tenth_stack)/share/f1tenth_stack/config/hj_speed_agent_71.yaml`로 지정. 기본 `hj_speed_agent.yaml`은 계속 guard off.
- 공식 A2 채택이 아니다(strict 1건 미달, BC68 고유). 정지 앞차는 보호하지 않는다. 동결 근거: `results/hj/261123/68_.../evaluation/lidar_following_guard/frozen_V2f/freeze_record.json`.
- V2f memory에는 `leader_speed`, `slow` 필드가 추가됐고 reset 규칙(stop/예외/이력 재시작)은 A9와 같다. 아래 '시간 기준 경고'는 V2f에도 동일하다.
- 위치: 정책 속도 → `min(speed_limit)` → **guard(목표속도=min(목표, guard 상한))** → speed_limiter. 속도를 올리지 못한다. 비유한/잘못된 입력이면 guard가 상한 0.
- episode memory 초기화: stop(), 입력 예외(Rejected 포함), 스캔 간격>scan_timeout 등으로 4프레임 이력이 새로 시작될 때.
- 알려진 한계: (1) held 모드가 정지 앞차를 가정(leader_speed 0)해 코너 가림 후 불필요하게 거의 정지할 수 있다. (2) 고장 빔 옆 오인(v2 개발 중). (3) A9의 시뮬 성능이 실차 보장이 아니다.
- **시간 기준 경고**: guard는 호출 횟수 기준이다(`dt_s=0.025`, `frame_dt_s=0.025`, `confirm_steps=10`, `hold_s=1.5`=60스텝). 실제 스캔 주기 T가 0.025가 아니면(`ros2 topic hz /scan`으로 확인; 로직은 변경하지 않음):

| T (s) | 확인 10스텝 | hold 60스텝 | held 이동거리 추정/실제 | motion test가 계산하는 closing / 실제 |
|---|---|---|---|---|
| 0.0125 | 0.125 s | 0.75 s | 2.0배 과대(보수적) | 0.5배: 정적 벽이 closing 0.5v로 보여 static 판정(0.7v 기준) 실패 -> 벽/기둥도 차로 취급, 불필요한 감속 |
| 0.025 | 0.25 s | 1.5 s | 1.0 | 1.0 (학습/동결 조건) |
| 0.05 | 0.5 s | 3.0 s | 0.5배 과소: held 간격이 실제보다 큼(덜 보수적) | 2.0배: 실제 closing 0.35v 이상이면 static 판정 -> 앞차가 ego 속도의 65% 이하여도 무시될 수 있음(위험) |
| 0.1 | 1.0 s | 6.0 s | 0.25배 | 4.0배: 실제 closing 0.175v 이상이면 static(위험) |

  `/scan` 주기가 0.025 s(40 Hz)에서 크게 벗어나면 guard를 켜지 말거나 주기를 맞추는 쪽을 먼저 검토할 것. 또한 스캔 드롭이 나면 history 간격이 늘어 같은 문제가 생긴다.
- 지연: 이 PC(Ryzen 9 5900X, numpy 1스레드급) guard 1회 p50 0.063 ms, p95 0.083 ms, p99 0.114 ms, max 0.23 ms(5628회, 시뮬 2차량 기록). Jetson은 측정 전이며 보통 수 배~10배 느릴 것으로 추정(미검증).

## 차속 입력 선택 speed_source (odom | vesc_core)
- `odom`(기본): `/odom twist.linear.x`(vesc_to_odom). `vesc_core`: `/sensors/core`(VescStateStamped)의 측정 ERPM으로 `v = (erpm - core_speed_offset_erpm)/gain`을 계산한다. 기본 `core_speed_offset_erpm = 0`이라 `v = erpm/gain`이고 ERPM 0 = 정지 = 0 m/s이다.
- 이유(추정, 실차 확인 필요): vesc_to_odom은 `(erpm + speed_to_erpm_offset)/gain`이라 offset -325에서 정지 시 -0.13 m/s가 예상된다. 명령용 offset(`speed_to_erpm_offset`)은 이 노드에서 쓰지 않고 측정 속도 변환과 섞지 않는다.
- 전환: `ros2 launch f1tenth_stack hj_speed_agent_launch.py ... speed_source:=vesc_core`. 데드밴드 `core_deadband`(기본 0.05), 부호·최대·stamp skew·0.25 s timeout 처리는 odom 경로와 같은 assembler다.
- **경고**: `speed_to_erpm_gain`은 이 노드 파라미터이며 `vesc.yaml`에서 자동으로 읽지 않는다(기본 2500). 실행 중인 vesc 노드 값(`ros2 param get /ackermann_to_vesc_node speed_to_erpm_gain`)과 다르면 정책 입력 속도가 틀린다. 보정으로 vesc.yaml gain을 바꾸면 이 값도 바꾼다. 보정 회귀(`ERPM_meas = gain*v_true + b`)의 절편 b가 0이 아닐 때만 `core_speed_offset_erpm := b`로 반영한다.
- vesc_msgs import는 `vesc_core`일 때만 노드에서 한다. 순수 함수 `erpm_to_speed`는 `hj_speed_policy_runtime`에 있고 ROS 없이 테스트된다.
