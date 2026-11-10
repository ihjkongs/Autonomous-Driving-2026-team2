# 전문가 랩타임 1차 최적화

이 문서는 전문가 튜닝 단계의 기록이다. 이후 진행한
[라이다 신경망 학습 결과](imitation_training_report.md)는 별도 문서에 정리했다.

실차 발진 문제는 해결될 것으로 가정하고 시뮬레이터 최적화를 진행했다.
실차에서 확인한 최고 속도·제동 성능은 아직 알려져 있지 않다.
아래 속도는 시뮬레이터 실험 범위이며, 실차에서 검증된 속도가 아니다.

## 결과

- 23개 트랙 각각 14회, **322/322회 완주**, 충돌·시간초과·실행 오류 0회.
- 같은 트랙·출발 방식·시드로 비교한 평균 랩타임:
  **164.12초 → 103.81초**.
- 전체 평균 랩타임 **36.75% 감소**.
- 신경망 학습은 실행하지 않았다. 이 수치는 위치·경로를 사용하는 전문가의 결과다.

## 바뀐 설정

| 설정 | 이전 완주 기준선 | 1차 최적화 |
|---|---|---|
| 추종 경로 | 중앙선 | 중앙선 |
| Lookahead 거리 | 1.0m | 1.2m |
| 최고 목표 속도 | 2.5m/s | 4.0m/s |
| 커브 감속 기준 | 1.5m/s² | 3.5m/s² |

커브 감속 기준이 커지면 커브에서 더 높은 속도를 허용한다. 이 값이 실제 횡가속도를
항상 그 이하로 제한한다는 뜻은 아니다. 조향 제한, 물리 모델, 충돌 판정,
제어 주기와 내부 물리 계산 간격은 기존 검증 상태를 유지했다.

현재 활성 설정은 [configs/expert.yaml](configs/expert.yaml)이다.
이전 설정은 [configs/expert_baseline.yaml](configs/expert_baseline.yaml)에 보관했다.

## 탐색 및 검증 범위

1. Lookahead `0.8, 1.0, 1.2m` × 커브 감속 기준 `1.5, 2.5, 3.5m/s²`를 비교했다.
   최고 속도는 현재 환경의 정규화 범위 안인 4m/s로 제한했다.
2. Sochi, Spa, Nuerburgring, Shanghai, Oschersleben, IMS에서 시드 17의
   출발선·임의 출발 조건으로 9개 후보와 이전 기준선을 비교했다. 총 120회다.
3. 후보를 완주 여부로 걸러낸 뒤 평균 랩타임이 짧은 순서로 선택했다.
4. 선택된 후보를 23개 트랙 × 출발 방식 2개, 총 46회 검증했다.
5. 이후 시드 `1, 124, 7, 42, 2026, 2027, 31415`의 322개 출발 조건으로
   다시 검증했다. 비교 대상 기준선과 트랙·출발 방식·시드를 일치시켰다.
6. 영상용으로 46회 추가 주행을 실행하고, 시드 7 출발선 주행의 영상 23개를 저장했다.
7. 별도 시드 73에서 23개 트랙의 완주 데이터도 다시 수집했다.

현재 결과는 **유한한 공통 파라미터 후보 중 선택한 1차 튜닝 결과**다.
전역 최적해, 트랙별 최단 랩타임 또는 미지 트랙의 무충돌을 보장하지 않는다.
현재 속도 범위를 넘어선 탐색, 레이싱 라인 최적화, 제동을 미리 고려한 속도 계획은
이번 단계에 포함하지 않았다.

![트랙별 평균 랩타임 비교](results/expert_optimization/bounded_4mps/lap_time_comparison.png)

## 트랙별 비교

시간은 같은 14개 출발 조건의 평균이다. 감소율의 전체 값은 트랙별 평균 시간의
합을 비교해 계산했다. 트랙별 감소율의 단순 평균과는 다를 수 있다.

| 트랙 | 이전 평균(초) | 새 평균(초) | 감소율 | 반복 완주 | 전문가 영상 |
|---|---|---|---|---|---|
| Sochi | 190.57 | 120.34 | 36.9% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Sochi.mp4) |
| Spa | 224.83 | 141.92 | 36.9% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Spa.mp4) |
| Nuerburgring | 181.75 | 115.21 | 36.6% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Nuerburgring.mp4) |
| Monza | 180.96 | 113.79 | 37.1% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Monza.mp4) |
| Melbourne | 193.48 | 121.93 | 37.0% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Melbourne.mp4) |
| Austin | 174.01 | 110.39 | 36.6% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Austin.mp4) |
| Silverstone | 185.95 | 117.48 | 36.8% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Silverstone.mp4) |
| Sakhir | 181.25 | 114.46 | 36.8% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Sakhir.mp4) |
| IMS | 117.28 | 73.47 | 37.3% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/IMS.mp4) |
| Budapest | 163.88 | 103.86 | 36.6% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Budapest.mp4) |
| Montreal | 119.16 | 75.58 | 36.6% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Montreal.mp4) |
| Sepang | 199.19 | 125.99 | 36.7% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Sepang.mp4) |
| Oschersleben | 108.00 | 69.33 | 35.8% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Oschersleben.mp4) |
| YasMarina | 167.56 | 106.25 | 36.6% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/YasMarina.mp4) |
| MoscowRaceway | 134.81 | 85.96 | 36.2% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/MoscowRaceway.mp4) |
| Zandvoort | 158.28 | 100.27 | 36.7% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Zandvoort.mp4) |
| Catalunya | 169.81 | 107.36 | 36.8% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Catalunya.mp4) |
| BrandsHatch | 143.95 | 90.81 | 36.9% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/BrandsHatch.mp4) |
| Shanghai | 203.87 | 128.81 | 36.8% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Shanghai.mp4) |
| Hockenheim | 147.54 | 93.54 | 36.6% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Hockenheim.mp4) |
| SaoPaulo | 140.82 | 89.21 | 36.7% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/SaoPaulo.mp4) |
| Spielberg | 139.48 | 87.80 | 37.1% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/Spielberg.mp4) |
| MexicoCity | 148.44 | 93.95 | 36.7% | 14/14 | [영상](results/expert_optimization/bounded_4mps/videoset/video/MexicoCity.mp4) |

[수치 원본 CSV](results/expert_optimization/bounded_4mps/comparison.csv),
[집계 JSON](results/expert_optimization/bounded_4mps/winner.json).
탐색별 원본 기록과 당시 코드 사본은 같은 결과 폴더에 있다.

## 데이터와 다음 학습

- 새 활성 데이터: `expert_data_verified.pkl`, **95,463개** 샘플.
- 트랙별 완주 1회씩, 총 23회. 메타데이터에 각 주행의 인덱스 구간을 저장했다.
- 이전 데이터: `expert_data_baseline.pkl`, 이전 메타데이터와 함께 보관했다.
- 현재 `main_il.py`의 기본 데이터 경로는 새 활성 데이터를 가리킨다.
- 기존 학습 모델은 변경하지 않았다.

다음 단계에서는 라이다 입력만 사용하는 신경망을 이 데이터로 학습하고,
신경망 자체의 완주율·랩타임을 다시 측정해야 한다. 전문가가 사용하는 위치·경로는
신경망의 입력에 포함되지 않는다. 실차 전이는 센서 전처리, 관측 이력 사용 여부,
복귀 데이터, 학습하지 않은 트랙 평가와 실차 보정까지 별도로 진행해야 한다.

후속 모방학습에서는 모델 선택 기준을 완주율 우선, 같은 완주율에서는 랩타임
우선으로 수정했다. 현재 학습 결과는 위의 별도 학습 보고서를 참고한다.

영상의 기존 `Steering` 표시는 지도 기준 앞바퀴 방향이다. 실제 입력 조향각은
왼쪽 위 `action[0]`를 참고한다.
