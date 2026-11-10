# 전문가 수정 및 검증 보고서

이 문서는 최고 2.5m/s의 이전 완주 기준선 기록이다. 아래 재실행 명령은 실행 당시의 활성 설정을 사용한다. 현재 활성 설정과 새 데이터는
[후속 랩타임 최적화 보고서](expert_optimization_report.md)를 참고한다.

검증일: 2026-11-07 (한국 시간).

최종 설정으로 **23개 트랙 각각 14회, 총 322/322회 완주**했다.
충돌·시간초과·조기 완주 판정·실행 오류는 모두 0회다.
새 전문가 데이터는 별도로 수집했으며 신경망 학습은 이번 작업에서 실행하지 않았다.

## 무엇이 문제였나

Lookahead 거리 하나만의 문제로 보기 어려웠다. 추종 코드의 결함과
시뮬레이터의 저속 계산 불안정성이 함께 있었다.

1. **계산한 목표점을 사용하지 않았다.** 경로와 Lookahead 원의 실제 교차점을
   계산한 뒤, 그 점이 아니라 교차점 앞의 경로 꼭짓점을 조향 계산에 사용했다.
   실제 교차점을 사용하도록 수정했다.
2. **큰 지도 좌표에서 정밀도가 손실됐다.** 원-선분 교차 계산에서 큰 좌표의
   제곱끼리 빼는 방식과 float32 계산 때문에 잘못된 교차점 또는 목표점 탐색 실패가
   발생했다. 차량과 경로 점 사이의 상대 좌표를 먼저 계산하도록 바꿨다.
   Sochi의 재현 위치 `(-128.30579, -48.25062)`는 경로에서 약 0.00288m 떨어져
   있었는데도 이전 Numba 함수는 교차점을 찾지 못했고, 수정한 함수는 찾았다.
3. **경로 재진입 처리가 불완전했다.** `(x, y, 속도)`가 필요한 곳에서 `(x, y)`만
   반환했다. 경로 앞쪽의 목표점과 속도를 함께 반환하고, 조향 계산에는 실제 목표점
   거리를 사용한다. 경로를 잃으면 4m/s 직진을 반환하던 처리도 오류로 중단하도록 바꿨다.
4. **출발 기준과 추종 기준이 달랐다.** 중앙선에서 출발해 옆의 레이싱 라인으로
   곧바로 합류하는 상황이었다. Sochi 출발점에서는 두 경로가 약 0.82m 떨어져 있었다.
   현재는 완주 안정성을 위해 중앙선을 추종하고, 속도를 제한하며 커브에서 감속한다.
5. **헤어핀에서 경로 순서를 잃을 수 있었다.** 서로 가까운 경로 구간 중 전역에서
   가장 가까운 점만 고르면 이전 구간으로 돌아갈 수 있었다. 직전에 추종하던 구간부터
   앞쪽 구간을 찾도록 바꿨다. 경로의 마지막 점과 첫 점을 연결하는 구간도 명시했다.
6. **저속 물리 계산이 수치적으로 불안정했다.** 최종적으로 남았던 상하이 실패의
   핵심 문제였다. 현재 차량 설정에서 1m/s의 횡방향 선형화 모드 중 하나는 약
   `-123.09/s`인데, RK4로 0.025초를 한 번에 계산하면 이 모드의 크기가 매번
   약 `1.537`배로 증가한다. 실제 실패 주행에서는 회전 속도가 약 `386rad/s`까지
   커졌다. 제어·라이다 주기 0.025초를 유지하면서, 내부 물리 계산을 최대
   0.005초씩 나눴다. 0.001초 기준 계산과 비교한 회귀 테스트가 통과했다.
7. **데이터 수집 기준이 완주를 보장하지 않았다.** 이전 코드는 200스텝보다 오래
   달리면 충돌한 주행도 저장할 수 있었고, 보통 1000스텝에서 끊었다.
   이제 전체 트랙을 순회하며 충돌 없이 한 바퀴를 완주한 데이터만 저장한다.
   코드·물리·차량 설정이 달라진 오래된 데이터는 학습 로더가 거부한다.

각 수정의 개별 기여도를 모두 분리한 실험은 아니다. 위 1–3번은 코드에서 확인한
결함이고, 4번은 안정화를 위한 경로 선택이다. 아래 비교 실험과 저속 수치 재현으로
전체 수정의 효과 및 마지막 물리 계산 문제를 확인했다.

## 비교 실험

| 단계 | 결과 | 기록 |
|---|---|---|
| 수정 전 전문가 | 14/92 완주 | `results/expert_validation/20261106-193408/` |
| 실제 교차점 사용·재진입 데이터 형식 수정 | 16/92 완주 | `results/expert_validation/intersection_fix/` |
| 상대 좌표 계산·중앙선·감속 적용, 물리 수정 전 | 91/92 완주 | `results/expert_validation/centerline_stable_geometry/` |
| 위 설정으로 새 시드 5개 비교, 물리 수정 전 | 227/230 완주 | `results/expert_validation/repeated_23_tracks/` |
| 경로 순서 보존만 추가한 상하이 비교 | 9/14 완주 | `results/expert_validation/shanghai_route_order/` |
| 내부 물리 간격 수정 후 상하이 | 14/14 완주 | `results/expert_validation/shanghai_stable_physics/` |
| **최종 전체 검증** | **322/322 완주** | `results/expert_validation/final_23_tracks/` |

진행 중 중단한 `centerline_conservative/` 실험은 완료된 검증으로 세지 않았다.
초기 실험의 메타데이터에 있는 `collection_rule`은 당시의 이전 수집 규칙 설명이며,
최종 수집 규칙은 아래와 같다.

## 최종 전문가 설정

[configs/expert_baseline.yaml](configs/expert_baseline.yaml)에 공통 설정을 모았다.

| 항목 | 설정 |
|---|---|
| 추종 경로 | 중앙선 (`centerline`) |
| Lookahead 거리 | 1.0m |
| 최고 목표 속도 | 2.5m/s |
| 최저 목표 속도 | 1.0m/s |
| 커브 감속 기준 | 조향 곡률에 따른 횡가속도 1.5m/s² |
| 축간 거리 | 차량 설정의 0.2 + 0.15 = 0.35m |
| 조향 제한 | 기존 ±0.48rad |
| 외부 제어 주기 | 기존 0.025초 |
| RK4 내부 계산 간격 | 최대 0.005초 |

모든 트랙에 같은 설정을 적용했다. 트랙별로 별도의 Lookahead 값을 넣지 않았다.
`lookahead_time: 0.0`이므로 속도에 따른 Lookahead 증가는 현재 사용하지 않는다.
Lookahead를 수동으로 바꾸면 `max_lookahead` 상한도 함께 확인해야 한다.
이 설정은 안정적인 완주를 목표로 하며, 최단 랩타임을 찾은 결과는 아니다.

## 최종 검증 방법과 결과

- 트랙 23개 × 출발 방식 2개 × 시드 7개 = 322회.
- 시드: `1, 124, 7, 42, 2026, 2027, 31415`.
- 출발 방식: 출발선 주변(`cl_grid_static`), 트랙 내 임의 위치(`cl_random_static`).
- 각 회차는 정지 상태에서 새로 출발해 한 바퀴를 주행했다. 재출발 없이 연속
  여러 바퀴를 주행한 실험은 아니다.
- 충돌 없음, 시뮬레이터 완주 체크포인트 통과, 순방향 누적 거리 90% 이상을
  모두 만족해야 완주로 판정한다. 실제 최소 순방향 진행률은
  **99.879%**였다.
- 최종 주행에서 기록된 최대 회전 속도는 **1.495rad/s**였다.
- 한 회차당 제한은 20000스텝(500초)이다. 충돌 판정·지도·마찰 계수·차량 크기·
  조향 제한을 완화하지 않았다.
- 전체 검증은 `native` 모드로 실행했다. 학습 래퍼와 동일한 제어 변환과 실제
  시뮬레이터 물리를 사용하며, 사용하지 않는 보상 계산과 매 스텝의 스플라인
  최적화를 생략한다. 진행 거리는 촘촘한 중앙선 선분에 투영해 측정한다.
  상하이 시드 124의 두 출발 방식은 실제 `F1Wrapper`로도 완주를 재확인했다.
- 코드 회귀 테스트 8개와 기존 차량 동역학 테스트 6개, 총 14개가 통과했다.
- 이 결과는 명시한 출발 조건에서의 검증이다. 모든 가능한 위치·외란에 대한
  수학적 보장을 뜻하지 않으며, 학습된 신경망의 성능은 별도로 검증해야 한다.

| 트랙 | 완주 | 주행 시간 범위(초) | 전문가 주행 |
|---|---|---|---|
| Sochi | 14/14 | 190.45–190.75 | [영상](results/expert_validation/final_23_tracks/video/Sochi.mp4) |
| Spa | 14/14 | 224.70–224.98 | [영상](results/expert_validation/final_23_tracks/video/Spa.mp4) |
| Nuerburgring | 14/14 | 181.50–182.05 | [영상](results/expert_validation/final_23_tracks/video/Nuerburgring.mp4) |
| Monza | 14/14 | 180.93–181.00 | [영상](results/expert_validation/final_23_tracks/video/Monza.mp4) |
| Melbourne | 14/14 | 193.40–193.55 | [영상](results/expert_validation/final_23_tracks/video/Melbourne.mp4) |
| Austin | 14/14 | 173.90–174.18 | [영상](results/expert_validation/final_23_tracks/video/Austin.mp4) |
| Silverstone | 14/14 | 185.78–186.10 | [영상](results/expert_validation/final_23_tracks/video/Silverstone.mp4) |
| Sakhir | 14/14 | 181.15–181.35 | [영상](results/expert_validation/final_23_tracks/video/Sakhir.mp4) |
| IMS | 14/14 | 117.28–117.28 | [영상](results/expert_validation/final_23_tracks/video/IMS.mp4) |
| Budapest | 14/14 | 163.60–164.13 | [영상](results/expert_validation/final_23_tracks/video/Budapest.mp4) |
| Montreal | 14/14 | 119.08–119.25 | [영상](results/expert_validation/final_23_tracks/video/Montreal.mp4) |
| Sepang | 14/14 | 199.08–199.33 | [영상](results/expert_validation/final_23_tracks/video/Sepang.mp4) |
| Oschersleben | 14/14 | 107.75–108.18 | [영상](results/expert_validation/final_23_tracks/video/Oschersleben.mp4) |
| YasMarina | 14/14 | 167.43–167.65 | [영상](results/expert_validation/final_23_tracks/video/YasMarina.mp4) |
| MoscowRaceway | 14/14 | 134.70–134.93 | [영상](results/expert_validation/final_23_tracks/video/MoscowRaceway.mp4) |
| Zandvoort | 14/14 | 158.05–158.53 | [영상](results/expert_validation/final_23_tracks/video/Zandvoort.mp4) |
| Catalunya | 14/14 | 169.53–170.00 | [영상](results/expert_validation/final_23_tracks/video/Catalunya.mp4) |
| BrandsHatch | 14/14 | 143.85–144.05 | [영상](results/expert_validation/final_23_tracks/video/BrandsHatch.mp4) |
| Shanghai | 14/14 | 203.70–204.03 | [영상](results/expert_validation/final_23_tracks/video/Shanghai.mp4) |
| Hockenheim | 14/14 | 147.28–147.70 | [영상](results/expert_validation/final_23_tracks/video/Hockenheim.mp4) |
| SaoPaulo | 14/14 | 140.68–140.93 | [영상](results/expert_validation/final_23_tracks/video/SaoPaulo.mp4) |
| Spielberg | 14/14 | 139.33–139.55 | [영상](results/expert_validation/final_23_tracks/video/Spielberg.mp4) |
| MexicoCity | 14/14 | 148.23–148.53 | [영상](results/expert_validation/final_23_tracks/video/MexicoCity.mp4) |

![23개 트랙의 전문가 완주 경로](results/expert_validation/final_23_tracks/trajectories_overview.png)

영상의 기존 `Steering` 표시는 차체 방향에 바퀴 조향각을 더한 지도 기준 방향이다.
실제 입력 조향각은 영상 왼쪽 위 `action[0]`이며, 기존 ±0.48rad 제한을 적용했다.
23개 영상은 전체 프레임을 디코딩해 파일 정상 여부도 확인했다.
[영상·데이터 검증 결과](results/expert_validation/final_23_tracks/artifact_verification.json).

원본 기록: [회차별 JSONL](results/expert_validation/final_23_tracks/episodes.jsonl),
[트랙별 CSV](results/expert_validation/final_23_tracks/by_track.csv),
[집계 JSON](results/expert_validation/final_23_tracks/summary.json).
검증 당시 코드 사본·해시와 설정은 같은 폴더의 `sources/`, `metadata.json`에 있다.
수정 전 코드 사본과 수치 진단은 `results/expert_validation/diagnosis/`에 있다.

## 새 학습 데이터

- 파일: `expert_data_baseline.pkl`.
- 완주 기록: 23개 트랙에서 각각 1회, 총 23회. 별도 시드 `73`을 사용했다.
- 학습 샘플: **150,995개**.
- 입력: 라이다 1080개 값, `float32`.
- 정답: 정규화된 조향·속도 2개 값, `float32`, 범위 `[-1, 1]`.
- `expert_data_baseline.metadata.json`에 트랙·출발 위치·완주 여부·샘플 인덱스
  구간·전문가 설정·검증용 서명을 저장했다.
- 기존 `expert_data.pkl`과 기존 학습 모델은 보관했다.
- `main_il.py`의 기본 입력을 새 데이터 파일로 변경했고, 기존 파일을 자동으로
  불러오지 않는다. 학습 중 평가 시간 제한도 한 바퀴를 평가할 수 있게 늘렸다.
- 신경망 재학습은 아직 실행하지 않았다.

## 다시 실행하는 방법

```bash
source ~/venv/torch210/bin/activate
cd /home/jiwon/projects/Autonomous-Driving-2026-team2/src/f1tenth

# 학습 없이 전문가만 검증. 실행 결과 폴더는 자동 생성된다.
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python validate_expert.py \
  --seeds 1 124 7 42 2026 2027 31415 --workers 6 \
  --step-mode native --max-steps 20000

# 새 파일 이름을 지정해 데이터를 다시 수집. 기존 파일은 덮어쓰지 않는다.
python collect_expert.py --output expert_data_next.pkl --seed 74
```

## 주요 파일 역할

| 파일 | 역할 |
|---|---|
| `configs/expert.yaml` | 전문가의 경로·거리·속도 설정 |
| `utils/rule_based.py` | Pure Pursuit 조향과 목표점 계산 |
| `installation/f1tenth_gym/envs/integrator.py` | 차량 운동을 작은 시간 간격으로 계산 |
| `validate_expert.py` | 전문가 완주 검증·영상·주행 기록 저장 |
| `collect_expert.py` | 학습 없이 새 전문가 데이터 수집 |
| `utils/expert_data.py` | 완주 데이터 수집·오래된 데이터 거부 |
| `main_il.py` | 검증된 데이터를 읽어 모방학습 진행 |
| `tests/test_expert.py` | 수정한 문제의 재발을 확인하는 테스트 |
