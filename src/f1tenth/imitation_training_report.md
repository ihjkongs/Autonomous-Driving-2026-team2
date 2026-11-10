# 라이다 모방학습 결과 — imitation/2

2026-11-07, 새 전문가 데이터로 신경망 학습과 별도 주행 평가를 실행했다.
현재 저장된 모델은 학습된 신경망이며, 기존 Pure Pursuit 전문가와 구분한다.

## 결과

- 전문가 데이터 **95,463개**, GPU **RTX 3050 / CUDA** 사용.
- 미니배치 1000개씩 **5,000회 업데이트**했다.
  샘플 수 기준 약 52.4회 데이터셋 분량이다.
- 초기 23개 출발 조건 비교에서는 **23/23 완주**했다.
- 이후 모델 선택 범위를 92개 조건으로 넓혔다. 최종 선택 모델은 **92/92 완주**했다.
- 이후 분리된 시드의 230회 평가에서 **230/230 완주
  (100.00%)**했다.
- 충돌 0회, 시간초과 0회,
  조기 체크포인트 0회, 실행 오류 0회.
- 최종 검증 10회를 모두 통과한 트랙: 23/23개.
- 성공한 회차들의 평균 랩타임: 104.00초.
  실패한 회차는 랩타임 평균에 포함하지 않았다.
- 최종 선택 모델은 **4,000회 업데이트 시점**이다.
  마지막 5,000회 시점의 모델을 무조건 선택하지 않았다.

## 신경망이 받은 정보

입력은 **한 프레임의 라이다 거리값 1080개**다. 출력은 조향과 목표 속도 2개다.
기존 1080 → 256 → 256 → 2 신경망과 MSE 모방학습을 사용했다.
평가 중 신경망에 위치, 트랙 이름, 경로 점, 전문가 행동을 제공하지 않았다.
전문가의 개입이나 주행 보정도 사용하지 않았다. 위치와 중앙선은 평가기의
주행 거리 및 완주 판정에만 사용했다.

![학습 곡선과 모델 선택 평가](results/imitation/2/training_curves.png)

## 학습과 모델 선택

기존 평가 점수는 오래 생존하면 커지는 스텝 보상 합이었다. 모델 저장 기준을
다음 순서로 비교하도록 바꿨다.

1. 완주율이 높은 모델.
2. 완주율이 같으면 성공한 주행의 평균 랩타임이 짧은 모델.
3. 완주가 없는 경우에는 누적 진행률이 높은 모델.

초기 후보 비교는 같은 23개 트랙·출발선 모드·시드 17로 평가했다.
이후 초기 선택 모델의 322회 진단에서 311회 완주에 그쳐, 저장된 모델들을
두 출발 방식·시드 17/19의 92개 조건으로 다시 비교했다. 이 진단 기록은
`evaluation/`에 보관했으며, 아래 최종 평가는 또 다른 시드로 실행했다.

| 업데이트 수 | 선택용 평가 완주 | 성공 회차 평균 랩타임(초) |
|---|---|---|
| 1000 | 11/23 | 98.91 |
| 2000 | 21/23 | 100.52 |
| 3000 | 23/23 | 103.64 |
| 4000 | 23/23 | 103.95 |
| 5000 | 23/23 | 103.88 |

보강한 모델 선택 결과:

| 업데이트 수 | 완주 | 성공 평균 랩타임(초) |
|---|---|---|
| 3000 | 89/92 | 103.02 |
| 4000 | 92/92 | 103.97 |
| 5000 | 91/92 | 103.63 |

최초 100회 평균 MSE는 0.049638, 마지막 100회 평균은
0.001258였다. 이는 학습 데이터의 제어값을 얼마나
잘 따라 하는지에 대한 값이며, 무충돌 주행을 보장하는 수치는 아니다.
마지막 MSE는 5,000회 시점의 값이고, 선택된 모델은 4000회 시점이다.

## 별도 반복 검증

23개 트랙 × 출발선/임의 위치 2가지 × 시드 5개 = 230회다.
시드는 `11, 53, 2718, 2028, 65537`이며, 모델 선택용 시드 17/19 및 초기 진단용
시드들과 분리했다.
충돌 없음, 체크포인트 통과, 순방향 누적 거리 90% 이상을 모두 만족해야 완주다.
출발선 주변은 범위가 좁아 다른 시드에서도 일부 위치가 중복될 수 있다.
시간 제한은 20,000스텝(500초), 목표 속도 범위는 1–4m/s다.

**이 23개 트랙은 학습 데이터에도 포함되어 있다.** 출발 조건에 대한 검증이며,
처음 보는 트랙에 대한 일반화 성능이나 실차 성능을 측정한 결과가 아니다.

| 트랙 | 완주 | 성공 평균(초) | 충돌 | 시간초과 | 신경망 주행 영상 |
|---|---|---|---|---|---|
| Sochi | 10/10 | 120.62 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Sochi.mp4) |
| Spa | 10/10 | 142.05 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Spa.mp4) |
| Nuerburgring | 10/10 | 115.03 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Nuerburgring.mp4) |
| Monza | 10/10 | 113.80 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Monza.mp4) |
| Melbourne | 10/10 | 122.45 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Melbourne.mp4) |
| Austin | 10/10 | 110.78 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Austin.mp4) |
| Silverstone | 10/10 | 117.66 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Silverstone.mp4) |
| Sakhir | 10/10 | 114.50 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Sakhir.mp4) |
| IMS | 10/10 | 73.52 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/IMS.mp4) |
| Budapest | 10/10 | 103.98 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Budapest.mp4) |
| Montreal | 10/10 | 75.91 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Montreal.mp4) |
| Sepang | 10/10 | 126.18 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Sepang.mp4) |
| Oschersleben | 10/10 | 69.46 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Oschersleben.mp4) |
| YasMarina | 10/10 | 106.38 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/YasMarina.mp4) |
| MoscowRaceway | 10/10 | 86.31 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/MoscowRaceway.mp4) |
| Zandvoort | 10/10 | 100.21 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Zandvoort.mp4) |
| Catalunya | 10/10 | 107.63 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Catalunya.mp4) |
| BrandsHatch | 10/10 | 90.93 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/BrandsHatch.mp4) |
| Shanghai | 10/10 | 129.60 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Shanghai.mp4) |
| Hockenheim | 10/10 | 93.59 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Hockenheim.mp4) |
| SaoPaulo | 10/10 | 89.27 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/SaoPaulo.mp4) |
| Spielberg | 10/10 | 88.07 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/Spielberg.mp4) |
| MexicoCity | 10/10 | 94.12 | 0 | 0 | [영상: complete](results/imitation/2/evaluation_final/video/MexicoCity.mp4) |

영상은 시드 53의 출발선 주행이다. `complete`는 완주, `collision`은 충돌로
끝난 영상이다. 영상이 존재한다고 모두 완주 영상인 것은 아니다.
23개 파일의 전체 프레임 디코딩을 확인했다. 기존 화면의 `Steering`은 지도 기준
앞바퀴 방향 표시이며, 실제 입력 조향은 왼쪽 위 `action[0]`다.

## 실패한 출발 조건

이번 230개 검증 조건에서는 실패가 없었다.

## 저장 위치

- 선택된 신경망: `results/imitation/2/sim2real/actor.pt`.
- 관측 정규화 통계: 같은 폴더의 `agent_obs_scale.pkl`.
- 선택된 모델과 옵티마이저: `results/imitation/2/checkpoint/model.pt`.
- 각 1,000회 시점 모델: `checkpoint/model_1000.pt`부터 `model_5000.pt`까지.
- 실행 설정·학습 당시 코드·데이터 메타데이터: `backup/`.
- [학습 집계](results/imitation/2/training_summary.json),
  [평가 집계](results/imitation/2/evaluation_final/summary.json),
  [트랙별 CSV](results/imitation/2/evaluation_final/by_track.csv),
  [회차별 원본](results/imitation/2/evaluation_final/episodes.jsonl).
- [신경망 영상 폴더](results/imitation/2/evaluation_final/video).
- [모델·정규화·영상 검증](results/imitation/2/artifact_verification.json).

기존 `results/imitation/1` 모델과 전문가 데이터는 보관했다.
`AgentBase`는 수정하지 않았다. 새 실험은 `--algo_idx 3`처럼 다른 번호를 사용한다.

## 다시 평가하기

```bash
source ~/venv/torch210/bin/activate
cd /home/jiwon/projects/Autonomous-Driving-2026-team2/src/f1tenth
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python evaluate_policy.py \
  --run results/imitation/2 --output results/imitation/2/evaluation_again \
  --seeds 11 53 2718 2028 65537 --workers 6 --video-seed 53
```

실차 배포는 이번에 실행하지 않았다. 실차용 입력 처리의 완성, 센서·차량 보정,
학습하지 않은 트랙 검증이 남아 있다. 실패 조건이 있다면 그 상태에서 복귀하는
전문가 데이터를 보강하는 작업도 다음 개선 대상으로 삼아야 한다.

모델 선택 기준에 대한 회귀 테스트 3개가 통과했다. 모델 파일과 선택된 체크포인트의
가중치 일치, 정규화 통계의 차원·샘플 수, 영상 23개의 전체 프레임 디코딩도 확인했다.
