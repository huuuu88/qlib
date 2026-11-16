# 라벨 · 신호 · 체결 타이밍

이 문서의 내용은 `research/tests/test_leakage.py`, `research/tests/test_portfolio.py` 가 코드로 강제한다.

## 타임라인 (기본: `LabelSpec(horizon=5, entry="next_open")`)

```
        t 일                t+1 일              ...        t+6 일
  ───────┬───────────────────┬──────────────────────────────┬────────
   종가 확정                시가                            시가
   피처·신호 계산           진입 체결                       청산 체결
   (t 일 종가까지의 정보)
```

| 항목 | 정의 | 코드 |
|---|---|---|
| 신호 시점 | t 일 장 마감 후. t 일 종가까지의 데이터만 사용 | 피처 식의 `Ref` 인자는 항상 양수 |
| 체결 시점 | **t+1 일 시가** (`execution_lag = 1`). 같은 날 종가 체결은 불가 | `labels.execution_date` |
| 보유기간 | N = 5 거래일 | `LabelSpec.horizon` |
| 라벨 | `open[t+1+N] / open[t+1] - 1` = `Ref($open,-6)/Ref($open,-1)-1` | `LabelSpec.expr` |
| 라벨 span | 6 거래일 (t 일 라벨은 t+6 일 시가까지 사용) | `LabelSpec.span` |
| 유동성 필터 | t-1 일까지의 정보로 계산한 `$tradable` | `data/cleaning.py` |

신호일과 체결일은 서로 다른 함수로 다룬다. 백테스터(`portfolio/backtest.py`)는 t 일 신호로 만든 목표 비중을
t+1 일 시가에 체결하고, 그 포지션은 t+1 시가 → t+2 시가 수익률부터 적용받는다. t 일 종가 → t+1 시가 구간(오버나이트 갭)의
수익은 누구도 가져갈 수 없다.

## 왜 시가-시가인가
- 종가 신호를 같은 종가에 체결하는 것은 실행 불가능하다(신호 계산에 그 종가가 필요).
- `next_close` (t+1 종가 진입)도 지원하지만, 하루 종일 정보가 반영된 뒤 진입하므로 단기 신호의 알파가 과소평가된다. 기본은 `next_open`.
- 한계: 시가 단일가(opening auction)에 원하는 수량이 전부 체결된다고 가정한다. 이 가정의 비용은 Phase 4 의
  슬리피지·시장충격·참여율 상한으로 반영한다.

## 분할 (Purged walk-forward)

```
 train (1008일) | 6일 비움 | valid (252일) | 6일 비움 | test (252일)      ← 252일마다 한 칸씩 전진, 재학습
```

- 구간 사이를 라벨 span(6 거래일)만큼 비운다. 비우지 않으면 train 마지막 날의 라벨이 valid 구간의 가격을,
  valid 마지막 날의 라벨(조기 종료에 사용)이 test 구간의 가격을 보게 된다.
- test 구간은 서로 겹치지 않고 이어지며, 모든 예측은 그 시점 이전 데이터로만 학습한 모델에서 나온다.
- `splits.assert_no_label_overlap` 이 매 실행마다 이를 검사한다.

## 정규화
- 허용: 일자 내(횡단면) 변환만 — `CSZScoreNorm`, `CSRankNorm`, `CSWinsorize`, `Fillna`, `DropnaLabel`, `TradableFilter`.
- 금지: 기간 통계를 쓰는 `ZScoreNorm`, `RobustZScoreNorm`, `MinMaxNorm`. 핸들러 생성 시 `assert_cross_sectional_only` 가 거부한다.

## 누수 검사 방법
`pipelines/leakage.py`: cut 시점 이후 데이터가 **아예 없는** 스토어와 전체 스토어에서 같은 피처를 계산해 cut 이전 값을 비교한다.
- Alpha158(157개) · Alpha360(300개) · 커스텀 팩터(10개): 차이 0건.
- 검사기 자체 검증: 미래를 보는 식(`Ref($close,-1)/$close`)과 라벨은 반드시 누수로 잡혀야 하며, 라벨은 정확히 `span × 종목 수` 행만 달라진다.

## 남아 있는 누수 경로 (코드로 막지 못한 것)
- 수정주가의 **수준**에는 미래 배당·분할이 들어 있다. 비율 피처만 쓰는 것으로 회피한다.
- 유니버스 멤버십은 위키피디아 기반 근사치이고, 편출 종목 다수의 시세가 없다(생존편향).
- 섹터 분류는 현재 기준이다.
