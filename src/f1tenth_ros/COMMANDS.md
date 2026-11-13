# RC Car ROS 2 상황별 명령어

Jetson의 ROS 2 Foxy에서 사용하는 명령어입니다. 교안 4·6과 현재 팀 코드를 기준으로 정리했습니다.
Jetson 워크스페이스는 교안의 `~/ros2_ws`를 가정합니다. 다른 경로에서 빌드했다면 아래 `source` 경로를 바꾸세요.
이 문서의 명령을 작성하면서 실제 차량을 구동하거나 기록을 수집하지는 않았습니다.

## 빠른 찾기

- [0. 새 터미널 환경 준비](#0-새-터미널-환경-준비)
- [1. 일반 키보드 주행](#1-일반-키보드-주행)
- [2. 노드·토픽 연결 확인](#2-노드토픽-연결-확인)
- [3. LiDAR 수신 확인](#3-lidar-수신-확인)
- [4. 키보드 명령과 VESC 응답 확인](#4-키보드-명령과-vesc-응답-확인)
- [5. 현재 설정값 확인](#5-현재-설정값-확인)
- [6. 문제 재현과 rosbag 수집: 처음부터 끝까지](#6-문제-재현과-rosbag-수집-처음부터-끝까지)
- [7. 기록 내용과 해석 범위](#7-기록-내용과-해석-범위)
- [8. 자주 만나는 상황](#8-자주-만나는-상황)

## 0. 새 터미널 환경 준비

**Jetson에서 새 터미널을 열 때마다** 실행합니다.

```bash
source /opt/ros/foxy/setup.bash
source ~/ros2_ws/install/setup.bash
```

교안대로 `~/.bashrc`에 이미 등록했다면 새 터미널에서 자동으로 적용됩니다.
PC의 학습용 `src/f1tenth/venv/torch210`과 Jetson의 ROS 환경은 별개입니다.

## 1. 일반 키보드 주행

### 터미널 1: 전체 시스템 실행

```bash
ros2 launch f1tenth_stack bringup_launch.py
```

이 터미널은 실행 상태로 둡니다. 이미 bringup이 실행 중이라면 중복 실행하지 않습니다.

### 터미널 2: 키보드 컨트롤 실행

0번의 환경 준비 후 실행합니다.

```bash
ros2 run key_teleop key_teleop
```

키보드 컨트롤 터미널에 포커스를 두고 조작합니다.

| 입력 | 동작 |
|---|---|
| `s` | 주행 명령 발행 시작 |
| `↑` / `↓` | 전진 / 후진 |
| `←` / `→` | 조향 |
| 방향키에서 손 떼기 | 마지막 방향키 입력 후 약 0.4초가 지나면 속도·조향 명령을 0으로 갱신 |
| `q` | 키보드 컨트롤 종료 |

종료할 때는 **방향키에서 손을 떼고 차량이 정지한 것을 확인한 뒤** `q`를 누릅니다.
그다음 터미널 1에서 `Ctrl+C`로 bringup을 종료합니다.
현재 코드에서 `r`은 명령 발행을 중단하므로 정지 버튼으로 사용하지 않습니다.

## 2. 노드·토픽 연결 확인

### 전체 노드 그래프

Jetson 데스크톱 등 GUI를 사용할 수 있는 터미널에서 실행합니다.

```bash
rqt_graph
```

### 실행 중인 노드와 토픽 목록

```bash
ros2 node list
ros2 topic list -t
```

### 특정 노드의 입력·출력

```bash
ros2 node info /key_teleop
ros2 node info /ackermann_mux
ros2 node info /ackermann_to_vesc_node
ros2 node info /vesc_driver_node
```

### 모터 속도 명령을 보내는 노드 확인

```bash
ros2 topic info /commands/motor/speed -v
```

`Publisher count`와 발행 노드 이름을 봅니다. 현재 로컬 bringup 구성에서는
`ackermann_to_vesc_node`가 속도 명령을 발행합니다.
다른 발행자가 있다면 서로 다른 명령을 보내고 있는지 확인합니다.
실제 Jetson의 구성에 따라 노드 이름이나 발행자 수는 달라질 수 있습니다.

## 3. LiDAR 수신 확인

### 스캔 메시지 보기

```bash
ros2 topic echo /scan
```

올바른 명령은 위와 같습니다. 앞에 `run`을 붙이지 않습니다.
기본 출력에서는 긴 배열이 생략될 수 있습니다. 모든 스캔 값을 보려면 다음을 사용합니다.

```bash
ros2 topic echo /scan --full-length
```

### 스캔 수신 주기 확인

```bash
ros2 topic hz /scan
```

`ranges`, `angle_min`, `angle_max`, `angle_increment` 등을 확인합니다.
메시지가 나온다는 사실과 스캔 값이 정상이라는 판단은 별도로 해야 합니다.
관찰이 끝나면 `Ctrl+C`로 해당 확인 명령을 종료합니다.

## 4. 키보드 명령과 VESC 응답 확인

키보드 주행의 명령 전달 경로는 다음과 같습니다.

```text
key_teleop
  → /key_vel
  → keyboard_to_ackermann_drive
  → /teleop
  → ackermann_mux
  → /ackermann_cmd
  → ackermann_to_vesc_node
  → /commands/motor/speed
  → vesc_driver_node → VESC
  → /sensors/core
```

아래 `echo`·`hz` 명령은 **각각 실행한 뒤 Ctrl+C로 종료하거나, 별도 터미널에서 실행**합니다.

| 상황 | 명령어 | 확인할 내용 |
|---|---|---|
| 키보드 입력 확인 | `ros2 topic echo /key_vel` | `linear.x`, `angular.z` |
| 키보드의 주행 명령 확인 | `ros2 topic echo /teleop` | `drive.speed`, `drive.steering_angle` |
| mux가 선택한 명령 확인 | `ros2 topic echo /ackermann_cmd` | 입력 명령이 선택되어 전달되는지 |
| 목표 회전수 확인 | `ros2 topic echo /commands/motor/speed` | `data`: 목표 ERPM |
| VESC 상태 확인 | `ros2 topic echo /sensors/core` | 보고 ERPM·전류·듀티·전압·Fault |
| 조향 명령 확인 | `ros2 topic echo /commands/servo/position` | 제한 전 서보 명령 |
| 제한 후 조향 명령 확인 | `ros2 topic echo /sensors/servo_position_command` | 드라이버가 0~1 등 설정 범위로 제한한 명령 |
| odometry 확인 | `ros2 topic echo /odom` | VESC 회전수를 바탕으로 계산한 속도·위치 |
| 속도 명령 수신 주기 | `ros2 topic hz /commands/motor/speed` | 평균 주기와 메시지 간 최대 간격 |
| VESC 상태 수신 주기 | `ros2 topic hz /sensors/core` | 상태 메시지의 수신 간격 |

속도 변환식은 `ERPM = speed_to_erpm_gain × drive.speed + speed_to_erpm_offset`입니다.
목표 ERPM과 `/sensors/core`의 `state.speed`를 같은 단위로 비교합니다.
`key_teleop`의 `Linear` 표시값은 명령값이며 실제 차량 속도를 측정한 값은 아닙니다.

## 5. 현재 설정값 확인

bringup과 키보드 노드가 실행 중인 상태에서 확인합니다.

```bash
ros2 param get /key_teleop forward_rate
ros2 param get /key_teleop backward_rate
ros2 param get /ackermann_to_vesc_node speed_to_erpm_gain
ros2 param get /ackermann_to_vesc_node speed_to_erpm_offset
ros2 param get /ackermann_to_vesc_node steering_angle_to_servo_gain
ros2 param get /ackermann_to_vesc_node steering_angle_to_servo_offset
ros2 param get /vesc_driver_node speed_min
ros2 param get /vesc_driver_node speed_max
```

실행 중인 설정을 읽는 명령입니다.
VESC 내부의 `Motor Current Max`·`Battery Current Max`·PID·Openloop 값은
이 ROS 파라미터와 별개이며 VESC Tool에서 확인합니다.
특히 ROS의 `current_max`를 VESC 내부 `Motor Current Max`와 혼동하지 않습니다.

## 6. 문제 재현과 rosbag 수집: 처음부터 끝까지

**VESC Tool에서는 바닥에서 RPM 입력으로 잘 가지만 키보드 제어에서는 약한 현상**을 기록하는 순서입니다.
VESC Tool에서 모터 제어를 종료하고 연결을 해제한 뒤 ROS로 시험합니다.
각 새 터미널에서는 먼저 0번의 환경 준비를 합니다.

한 번의 시험을 `~/vesc_logs/run_01`에 모읍니다. 다시 시험할 때는 아래 모든
`run_01`을 `run_02` 등 새로운 이름으로 바꿉니다.
`ros2 bag record`의 출력 폴더는 새로 생성되어야 하므로 `bag` 폴더를 미리 만들지 않습니다.

### 6-1. 터미널 1: bringup과 시작 로그 저장

bringup이 이미 실행 중이면 이 단계는 건너뛰고, 터미널 2에서 폴더를 만들어 기록을 시작합니다.

```bash
mkdir -p ~/vesc_logs/run_01
ros2 launch f1tenth_stack bringup_launch.py 2>&1 | tee ~/vesc_logs/run_01/bringup.log
```

### 6-2. 터미널 2: 전체 토픽 기록 시작

```bash
mkdir -p ~/vesc_logs/run_01
ros2 bag record -a --include-hidden-topics -o ~/vesc_logs/run_01/bag
```

전체 토픽과 숨겨진 토픽을 대상으로 기록합니다. LiDAR 등도 포함되므로
저장 공간을 확인하고 문제 재현에 필요한 구간을 기록합니다.
`/sensors/core` 등 필요한 토픽의 구독이 시작됐는지 출력에서 확인합니다.

### 6-3. 터미널 3: 키보드 컨트롤 실행

키보드 컨트롤이 이미 실행 중이면 그대로 사용합니다.

```bash
ros2 run key_teleop key_teleop
```

아직 움직이지 말고 다음 단계에서 현재 설정을 저장합니다.

### 6-4. 터미널 4: 노드·설정·발행자 정보 저장

```bash
mkdir -p ~/vesc_logs/run_01/context

ros2 node list > ~/vesc_logs/run_01/context/nodes.txt
ros2 topic list -t > ~/vesc_logs/run_01/context/topics.txt

for node in /key_teleop /keyboard_to_ackermann_drive /ackermann_mux /ackermann_to_vesc_node /vesc_driver_node /vesc_to_odom_node; do
  ros2 param dump "$node" --output-dir "$HOME/vesc_logs/run_01/context"
done

for mode in speed current brake duty_cycle position; do
  ros2 topic info "/commands/motor/$mode" -v > "$HOME/vesc_logs/run_01/context/motor_$mode.txt"
done
```

`Node not found`가 나오면 `nodes.txt`에 나온 실제 이름으로 해당 `param dump`를 다시 실행합니다.
ROS 파라미터의 초기값을 저장하는 단계이며, `/parameter_events` 기록만으로 이를 대신할 수 없습니다.

### 6-5. 터미널 3에서 문제 재현

1. `s`를 누르고, 문제를 확인할 수 있는 낮은 속도 명령으로 조작합니다.
2. 공중 시험과 바닥 시험은 차량을 정지시킨 뒤 상태를 바꿔 진행합니다.
3. 각 구간의 시간, 방향키 입력, 모터 소리·떨림, 실제 바퀴 움직임을 남깁니다.
4. VESC Tool과 비교할 때는 바닥 상태와 목표 ERPM을 맞춥니다. m/s 명령과 RPM을 같은 숫자라는 이유로 비교하지 않습니다.

가능하면 바퀴와 모터 소리가 담기는 영상도 촬영합니다.
터미널 4의 다음 명령으로 메모 파일을 작성할 수 있습니다.

```bash
nano ~/vesc_logs/run_01/notes.txt
```

메모 예시:

```text
기록 시작 시각:
공중 시험 시작/종료 시각:
바닥 시험 시작/종료 시각:
VESC Tool에서 잘 갔던 목표 RPM:
ROS에서 보낸 목표 ERPM:
약해진 순간에 매끈하게 도는지 / 덜덜 끊기는지:
모터와 바퀴의 회전이 함께 느려지는지:
```

### 6-6. 종료 순서

1. 터미널 3에서 방향키를 놓고 차량이 정지한 것을 확인한 뒤 `q`로 키보드 컨트롤을 종료합니다.
2. 터미널 2에서 `Ctrl+C`로 bag 기록을 종료하고 파일 저장이 끝날 때까지 기다립니다.
3. 터미널 1에서 `Ctrl+C`로 bringup을 종료합니다.

### 6-7. 저장 결과 확인과 묶기

터미널 4에서 실행합니다.

```bash
ros2 bag info ~/vesc_logs/run_01/bag
```

`/key_vel`, `/teleop`, `/ackermann_cmd`, `/commands/motor/speed`, `/sensors/core`에
메시지가 저장됐는지 확인합니다. 발행되지 않은 선택 토픽은 목록에 없을 수 있습니다.
구독 실패나 저장 오류가 있었다면 먼저 원인을 확인하고 다시 기록합니다.

VESC Tool에서 현재 컨트롤러의 Motor Configuration을 읽은 뒤 XML로 저장하여
`vesc_motor_config.xml`이라는 이름으로 `run_01` 폴더에 함께 보관합니다.
이 과정에서는 교안대로 유지한 전류 제한값을 변경하지 않습니다.

기록을 하나의 파일로 묶으려면 다음을 실행합니다.

```bash
tar -czf ~/vesc_logs/run_01.tar.gz -C ~/vesc_logs run_01
```

## 7. 기록 내용과 해석 범위

| 토픽 | 진단에 사용하는 내용 |
|---|---|
| `/key_vel` | 키보드 입력으로 생성한 속도·회전 명령 |
| `/teleop`, `/drive` | mux에 들어가는 수동·자율주행 명령 |
| `/ackermann_cmd` | mux가 선택한 최종 주행 명령 |
| `/commands/motor/speed` | 목표 ERPM |
| `/commands/motor/unsmoothed_speed` | 속도 완화 노드를 사용한다면 완화 전 명령 |
| `/commands/motor/current`, `/commands/motor/brake` | 전류·제동 명령이 섞여 들어오는지 |
| `/commands/motor/duty_cycle`, `/commands/motor/position` | 다른 제어 방식의 명령이 섞여 들어오는지 |
| `/commands/servo/position`, `/commands/servo/unsmoothed_position` | 조향 명령과, 완화 노드를 사용한다면 완화 전 명령 |
| `/sensors/servo_position_command` | 드라이버가 범위를 제한한 뒤 보낸 서보 명령 |
| `/sensors/core` | `speed`, `current_motor`, `current_input`, `avg_id`, `avg_iq`, `duty_cycle`, `voltage_input`, `fault_code` 등 |
| `/odom` | VESC 회전수로 계산한 차량 속도·위치 |
| `/diagnostics`, `/rosout` | mux 상태·통신 오류·명령 제한 등 |
| `/parameter_events` | 기록 중의 ROS 파라미터 변경 |

`/sensors/core`의 모든 필드는 메시지 정의와 드라이버 구현을 함께 확인해야 합니다.
현재 로컬 드라이버는 `temp_fet`·`temp_motor`를 직접 채우지 않으므로,
두 필드의 `0`을 실제 온도 0도로 해석하지 않습니다.

`/odom`은 독립적인 실측 속도가 아니며, `/sensors/servo_position_command`도 실제 조향각 센서값이 아닙니다.
VESC 내부 설정, 회전자 실제 위치, 바퀴의 실제 움직임은 bag만으로 모두 알 수 없습니다.
VESC 설정 XML과 시험 시각·영상도 함께 보관합니다.
기록 토픽을 지정하는 것만으로 메시지 무손실이 보장되지는 않으므로 recorder 출력과 `bag info`도 확인합니다.

## 8. 자주 만나는 상황

| 상황 | 먼저 확인할 것 |
|---|---|
| `ros2: command not found` | 0번 환경 준비 |
| `Package 'f1tenth_stack' not found` | 실제 빌드한 Jetson 워크스페이스의 `install/setup.bash`를 source했는지 |
| 키보드 화면에서 반응이 없음 | 터미널 포커스, `s` 입력, `/key_vel` 메시지 |
| `/key_vel`은 나오는데 최종 명령이 안 나옴 | `/teleop` → `/ackermann_cmd` → `/commands/motor/speed` 순서로 확인 |
| VESC Tool에서는 바닥 RPM 제어가 정상이고 ROS에서 약함 | 동일 목표 ERPM인지, 속도 명령 주기·다른 제어 명령·드라이버 제한값 확인 |
| `servo command ... clipping` | 요청한 서보 명령과 제한 후 명령, 조향 변환 gain·offset 확인 |
| `Output folder ... already exists` | 새 시험 폴더 `run_02` 등을 사용 |
| 토픽 이름이 다름 | `ros2 topic list -t`, `ros2 node info`로 namespace·remapping 확인 |
| `rqt_graph` GUI가 열리지 않음 | GUI 세션에서 실행하거나 `ros2 node info`·`ros2 topic info` 사용 |

## 참고

- [교안 4: Keyboard Control with ROS2](../../../manuals/4.%20Keyboard_Control_with_ROS2.pdf) — 로컬 작업 공간의 강의 자료
- [교안 2: VESC Setup](../../../manuals/2.%20VESC%20setup_manual_updated_2026.pdf) — 로컬 작업 공간의 강의 자료
- [ROS 2 Foxy: bag record 명령 정의](https://github.com/ros2/rosbag2/blob/foxy/ros2bag/ros2bag/verb/record.py)
- [ROS 2 Foxy: 파라미터 저장 명령 정의](https://github.com/ros2/ros2cli/blob/foxy/ros2param/ros2param/verb/dump.py)
- [ROS 2 Foxy: topic echo 명령 정의](https://github.com/ros2/ros2cli/blob/foxy/ros2topic/ros2topic/verb/echo.py)
