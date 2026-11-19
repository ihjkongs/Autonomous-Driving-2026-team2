환경 설정 (런치 및 키보드 실행용 터미널 각각에서 실행):
```bash
source /opt/ros/foxy/setup.bash
source ~/Autonomous-Driving-2026-team2/install/setup.bash
```

키보드만 확인할 때는 아래 런치 명령에 `enable_agent:=false`를 추가합니다.

1. First, launch the whole RC car ROS 2 system.
```
ros2 launch f1tenth_stack bringup_launch.py 
```


2. Open the other terminal and run:
```
ros2 run key_teleop key_teleop
```
Push button ‘s’, and you can now control your RC car with the arrow buttons on the keyboard.


0-0. 전체 노드 그래프 보고싶을때
```
rqt_graph
```


0-1. Lidar 잘 들어오는지 확인
```
run ros2 topic echo /scan
```
You can see some list of outputs consist of LiDAR scan values.


rosbag
```
ros2 bag record -o vesc_diagnosis /commands/motor/speed /sensors/core /key_vel

```

```
mkdir -p vesc_context

ros2 node list > vesc_context/nodes.txt
ros2 topic list -t > vesc_context/topics.txt

ros2 param dump /key_teleop --output-dir vesc_context
ros2 param dump /ackermann_mux --output-dir vesc_context
ros2 param dump /ackermann_to_vesc_node --output-dir vesc_context
ros2 param dump /vesc_driver_node --output-dir vesc_context
ros2 param dump /vesc_to_odom_node --output-dir vesc_context

```

키보드 관련
```

source /opt/ros/foxy/setup.bash
source ~/Autonomous-Driving-2026-team2/install/setup.bash

mkdir -p ~/vesc_logs
cd ~/vesc_logs

ros2 bag record -a --include-hidden-topics
    
# 자율주행 실행 명령어(젯슨 터미널에서 가상환경으로 실행)_2026.11.10_지원

ros2 launch f1tenth_stack bringup_launch.py \
  project_root:=/home/team2/Autonomous-Driving-2026-team2 \
  agent_python:=/home/team2/venv/torch210/bin/python \
  speed_limit:=4.0 \
  max_acceleration:=2.0 \
  max_deceleration:=3.0 \
  dry_run:=false


# rosbag 실행 명령어(별도의 젯슨 터미널에서 실행)_2026.11.10_지원

source ~/Autonomous-Driving-2026-team2/install/setup.bash

ros2 bag record \
  -o ~/drive_final2500_$(date +%Y%m%d_%H%M%S) \    #여기는 파일명 설정하는거
  /scan /agent/prediction /agent/command_preview \
  /drive /teleop /ackermann_cmd \
  /commands/motor/speed /sensors/core \
  /odom /diagnostics /rosout
