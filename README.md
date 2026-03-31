# ESS 배터리 수명 예측

초기 100 cycle 안에서 얻을 수 있는 열화, 내부저항, 충전 거동 신호를 이용해 배터리의 `cycle_life`를 조기에 예측하는 것이 본 프로젝트의 목적이다. 본 프로젝트는 MIT-Stanford Battery Dataset을 바탕으로 `Batch 1`에서 학습하고 `Batch 2`에서 일반화 성능을 검증하는 회귀 문제로 구성했다.

## 프로젝트 개요

- 데이터셋: MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터: Batch 1 (`2017-05-12`)
- 평가 데이터: Batch 2 (`2018-02-20`)
- 태스크: Regression (`cycle_life` 예측)

## 파일 구조

```text
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── results/
│   └── model_performance.csv
├── requirements.txt
└── README.md
```

## 환경 설정

```bash
git clone https://github.com/팀명/ess-battery-project
cd ess-battery-project
pip install -r requirements.txt
```

## EDA

### Cycle Life 분포

- `Batch 1`은 `cycle_life`가 전반적으로 높고, `Batch 2`는 단수명 셀 비중이 훨씬 높았다.
- `Batch 1`의 median cycle life는 `870`, `Batch 2`의 median cycle life는 `491`로 배치 간 분포 차이가 뚜렷했다.
- 핵심 발견: 학습 배치와 평가 배치의 수명 분포가 다르기 때문에 random split보다 batch-aware validation이 필요했다.

### 열화 곡선 분석

- 장수명 셀은 더 오랫동안 높은 `Qd retention`을 유지했고, 단수명 셀은 더 이른 시점부터 감소가 빨라졌다.
- stage slope 비교 결과 대부분 `1-100 < 101-300 < 301+` 방향으로 열화 속도가 커지는 경향이 확인되었다.
- 단일 knee point보다 `acceleration onset`이 가속 열화 시작을 더 안정적으로 설명했다.
- 핵심 발견: 초기 100 cycle 안의 감소량, 유지율, 기울기 정보만으로도 이후 수명 차이를 설명할 수 있었다.

### ΔQ(V) 곡선 분석

- 이번 notebook 기반 분석에서는 ΔQ(V) 원곡선을 직접 모델 입력으로 사용하지는 않았다.
- 대신 `Cycle 100 - Cycle 10` 차이를 요약한 `Qd_delta_100_10`, `Qd_retention_100_10`, `Qd_slope_1_100` 같은 early-life summary feature로 치환했다.
- 핵심 발견: 상세 곡선 전체를 쓰기보다 초기 구간의 변화량과 기울기 요약이 더 단순하고 재현 가능했다.

### 충전 속도(C-rate)와 수명의 관계

- `charging_policy`와 `policy_soc_pct`는 배치 내부 분산과 배치 간 shift를 설명하는 중요한 변수였다.
- 같은 feature set이라도 charging policy에 따라 validation 성능이 달라져, 모델 검증에서도 `charging_policy` 기준 그룹 분할이 필요했다.
- 핵심 발견: 프로토콜 이름 자체보다 policy를 숫자형으로 요약하거나, 검증 전략에 policy 그룹을 반영하는 것이 더 안정적이었다.

### 추가 확인 내용

- raw/filtered를 모두 비교해도 핵심 feature 방향성은 크게 변하지 않았고, 특히 slope 해석은 filtered variant가 더 안정적이었다.
- 절대값 feature보다 `variation`, `ratio`, `normalized summary`가 batch 이동에 더 강한 경향이 있었다.

## Modeling

### 피처 엔지니어링 전략

EDA 결과를 바탕으로 초기 100 cycle에서 계산 가능한 feature만 사용했다. 최종 후보군은 `Qd` 감소량과 유지율, `IR` 증가량과 변동성, `QC` 유지율, 평균 충전 시간, 충방전 비율로 구성했다. 이 전략은 한 종류의 신호만 보는 대신,

- 열화 shape: `Qd_delta_100_10`, `Qd_retention_100_10`, `Qd_slope_1_100`, `Qd_100`
- 상태 변화: `IR_delta_100_10`, `IR_cv_1_100`
- 충전/방전 거동: `QC_retention_100_10`, `chargetime_100_mean`, `Qd_QC_ratio_100`

를 함께 사용해 장수명/단수명 차이를 설명하도록 설계했다.

### 모델 선택 및 근거

- 후보 모델: `SVR(RBF)`, `NuSVR`, `KernelRidge`, 간단한 앙상블
- 최종 모델: `SVR(RBF)`
- 선택 이유:
  `Batch 1` 내부에서만 후보 feature와 모델을 비교하고, `charging_policy` 기준 `GroupKFold`와 hold-out validation을 사용해 일반화 성능이 가장 안정적인 조합을 선택했다. 현재 notebook 기준 최종 선택은 `SVR(RBF)`이며, 선택된 feature는 아래 9개다.

- `Qd_delta_100_10`
- `Qd_retention_100_10`
- `IR_delta_100_10`
- `QC_retention_100_10`
- `chargetime_100_mean`
- `IR_cv_1_100`
- `Qd_QC_ratio_100`
- `Qd_slope_1_100`
- `Qd_100`

## 성능 결과

현재 [03_modeling.ipynb](/Users/hyun/workspace/data/notebooks/03_modeling.ipynb) 기준 최종 결과는 아래와 같다.

| Split | Metric |
| --- | ---: |
| Train (`Batch 1 CV`) | `12.55%` |
| Valid (`Batch 1 Hold-out`) | `10.56%` |
| Test (`Batch 2`) | `18.72%` |

해석:

- Train과 Valid는 모두 `10%대 초반`으로 유지되었다.
- Test에서 오차가 다시 커지는 것은 `Batch 1 -> Batch 2` 분포 차이와 일부 충전 프로토콜 편향의 영향으로 보인다.

## 오류 분석

- 가장 크게 틀린 샘플은 `Batch 2`의 `newstructure` 계열 charging policy에 집중되어 있었다.
- 실제 수명이 `800~1200` 수준인 일부 셀과 `500` 이하인 단수명 셀에서 예측이 비슷한 값으로 수렴하는 경향이 있었다.
- 원인 가설:
  `Batch 2`의 특정 프로토콜은 `Batch 1`에 비해 분포가 달라서, 학습 데이터에서 충분히 보지 못한 패턴일 가능성이 높다.
- 개선 방향:
  policy-aware feature 보강, batch-robust scaling, 프로토콜 그룹 기준 validation, 추가 배치 데이터 확장이 필요하다.

## ESS 도메인 해석

초기 100 cycle만으로 수명 경향을 예측할 수 있다는 점은 실제 BESS 운영에서 조기 선별과 운전 전략 조정에 의미가 있다.

- 적용 가능 의사결정:
  조기 불량 셀 선별, 랙 단위 유지보수 우선순위 결정, charging protocol 최적화, 장기 운영 리스크 평가
- 한계:
  배치 간 분포 차이가 크고, 실험실 조건과 실제 ESS 운영 환경의 온도·부하 패턴이 다르다.
- 실배포를 위해 추가로 필요한 것:
  더 다양한 배치 검증, 현장 운영 데이터 연동, 시간에 따라 누적 업데이트되는 online monitoring feature, calibration과 uncertainty estimation

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383-391.

## 팀 구성

- 윤정원: Q1, Q3 EDA
- 배석현: Q2 EDA
- 박나연: Q4 EDA
- 황다빈: Q5 EDA
