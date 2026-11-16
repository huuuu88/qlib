# US 크로스섹션 퀀트 리서치 파이프라인

`microsoft/qlib` fork(v0.9.8) 위에 얹은 리서치 코드. upstream 파일은 수정하지 않고 모든 코드는 `research/` 아래에 둔다.
목표는 수익률이 아니라 **결과를 믿을 수 있는 구조**다.

```
research/
  data/        수집·정제·qlib 변환   (원천/가공 데이터는 data/store/, gitignore)
  factors/     커스텀 팩터, alphalens 검증
  configs/     데이터·실험 설정 (qrun yaml 포함)
  pipelines/   라벨·분할·누수검사·walk-forward·리포트
  portfolio/   포트폴리오 구성, 비용모델, 백테스터
  reports/     자동 생성 리포트 (커밋됨)
  tests/       데이터/누수/포트폴리오 테스트
```

결과부터 보려면: [`reports/final/sp500/REPORT.md`](reports/final/sp500/REPORT.md)

## 처음부터 재현하기 (Windows, PowerShell)

```powershell
# 0) 준비물: Miniforge(conda), Visual Studio 2022 Build Tools(C++ 워크로드), git
#    PyPI 에는 pyqlib 0.9.7 wheel 까지만 있어 fork(v0.9.8)를 소스에서 빌드해야 한다.
#    Git Bash 에서는 setuptools 가 MSVC 를 찾지 못한다. 빌드는 PowerShell/cmd 에서 할 것.

# 1) 환경
conda create -y -n usquant python=3.11 pip
conda activate usquant
pip install -r research/requirements.lock.txt
pip install -e . --no-deps
$env:PYTHONUTF8 = "1"        # 한글 주석이 든 yaml 을 qlib 이 읽을 수 있게 (필수)
$env:PYTHONPATH = "."        # qrun 이 research 패키지를 찾을 수 있게

# 2) 테스트 (데이터 없이도 통과. 스토어가 있으면 스토어 검증 테스트도 함께 돈다)
python -m pytest research/tests -q

# 3) Phase 1 — 데이터 (수집 → 정제 → qlib 변환 → 해시 → 검증 → 품질 리포트)
python -m research.data.build --config research/configs/data_smoke50.yaml     # 50종목, 약 3분
python -m research.data.build --config research/configs/data_sp500.yaml       # S&P 500 근사 PIT, 약 80분
#    보존된 raw 스냅샷으로 재빌드(다운로드 없음, 같은 해시가 나와야 한다)
python -m research.data.build --config research/configs/data_sp500.yaml --skip-download

# 4) Phase 3 — 기준선과 비교 실험 (walk-forward, 252거래일마다 재학습)
python -m research.pipelines.walk_forward --data sp500 --experiment alpha158_lgb   # 기준선
python -m research.pipelines.walk_forward --data sp500 --all                       # 6개 실험 전부
python -m research.factors.evaluate --data sp500                                   # 팩터별 검증 (alphalens)
python -m research.pipelines.summarize --data sp500                                # 모델 비교·채택·시도 횟수
qrun research/configs/baseline_alpha158_lgb.yaml                                   # 기준선 단일 분할 (= fold 5)

# 5) Phase 4 — 포트폴리오·비용 (신호마다 약 10분)
python -m research.portfolio.run --data sp500 --signal alpha158_lgb
python -m research.portfolio.run --data sp500 --signal factors_equal

# 6) Phase 5 — 최종 리포트
python -m research.pipelines.report --data sp500 --signals alpha158_lgb factors_equal

# 실험 추적 UI
mlflow ui --backend-store-uri research/mlruns
```

**같은 숫자를 재현하려면** Yahoo 를 다시 받지 말고 raw 스냅샷(`research/data/store/<name>/raw`, `raw_meta`)을 그대로 두고
`--skip-download` 로 빌드해야 한다. Yahoo 수정주가는 조회 시점마다 달라지므로 새로 받으면 해시와 결과가 달라진다.
스냅샷의 해시는 `reports/phase1*/manifest.json` 에 있다.

## 무엇이 어디에 기록되는가

| 항목 | 위치 |
|---|---|
| 데이터 버전(해시), 편향 플래그 | `reports/phase1*/manifest.json`, `<store>/qlib/research_meta.json` |
| 데이터 품질 | `reports/phase1*/data_quality.md` |
| 라벨·체결 타이밍 | [`pipelines/TIMING.md`](pipelines/TIMING.md) |
| 실험 config·시드·git commit·데이터 해시·지표 | `reports/phase3/<data>/<실험>.json`, MLflow(`research/mlruns`) |
| 시도 횟수 | `reports/trials.jsonl` |
| 팩터별 검증 | `reports/phase3/<data>/factor_validation.md` |
| 모델 비교·채택 | `reports/phase3/<data>/summary.md` |
| 비용 전/후 성과, 민감도, 용량 | `reports/phase4/<data>/<신호>/summary.md` |
| 최종 리포트 | `reports/final/<data>/REPORT.md` |

## 데이터 레이어 설계

| 계층 | 모듈 | 역할 |
|---|---|---|
| 소스 추상화 | `data/sources/base.py` | `DataSource` 인터페이스, 표준 스키마, `SourceCapabilities` |
| Yahoo 어댑터 | `data/sources/yahoo_qlib.py` | fork 의 `scripts/data_collector/yahoo` 를 서브클래싱 |
| S&P 500 근사 PIT | `data/sources/yahoo_wiki_sp500.py` | 위키피디아 편입·편출 이력으로 멤버십 복원 |
| 정제 | `data/cleaning.py` | 결측·무효가격·거래정지·분할 의심·유동성 필터 |
| 유니버스 | `data/universe.py` | (symbol, start, end) PIT 구간 → qlib instruments |
| 펀더멘털 | `data/fundamentals.py` | 공시일(report_date) + lag 정렬 |
| qlib 변환 | `data/qlib_dump.py` | upstream normalize + `dump_bin` |
| 버전 | `data/manifest.py` | 데이터 sha256, git commit, 패키지 버전, 편향 플래그 |
| 검증 | `data/verify.py` | qlib API 로 다시 읽어 원본과 대조 |
| 리포트 | `data/quality.py` | 품질 리포트 |

### 소스 교체 방법
유료 PIT 소스(Sharadar, Norgate, CRSP, Polygon 등)로 바꿀 때:
1. `data/sources/<name>.py` 에 `DataSource` 를 구현하고 `@register_source("<name>")` 등록
2. `capabilities` 에 `includes_delisted / pit_membership / pit_fundamentals` 를 사실대로 기입
3. PIT 멤버십이 있으면 `fetch_membership` 구현
4. config 의 `source:` 만 변경 — 정제·변환·학습·백테스트 코드는 그대로

편향 플래그는 `capabilities` 와 멤버십 종류에서 **자동 도출**되어 `manifest.json` 과
`<qlib_dir>/research_meta.json` 에 기록되고, 이후 모든 리포트 상단에 자동 표기된다.
메타 파일이 없으면 "생존편향 있음"으로 가정한다.

## Yahoo 데이터의 한계 (반드시 읽을 것)

1. **생존편향**: 상장폐지 종목의 시세를 주지 않는다. S&P 500 멤버였던 종목 중 시세가 있는 비율은 품질 리포트의
   "유니버스 커버리지" 표에 연도별로 나온다.
2. **PIT 지수 편입 이력 없음**: `smoke50` 은 현재 목록을 과거에 소급 적용한 것이고(`static_current_list`),
   `sp500` 은 위키피디아 이력으로 복원한 근사치다(`approximate_pit`, 티커 변경 미추적). 둘 다 PIT 로 인정하지 않는다.
3. **수정주가의 비재현성**: adjclose 는 조회 시점 기준으로 재계산된다. raw 스냅샷과 그 해시가 데이터 버전이다.
4. **수정주가와 미래 정보**: 수정주가의 과거 "수준"은 이후의 배당·분할을 반영한다. 수익률·비율 피처는 안전하지만
   가격 수준을 직접 피처로 쓰면 미래 정보가 섞인다. 비수정 종가는 `$close/$factor` 로 얻는다.
5. **분할 오류 가능성**: 미반영/오반영을 탐지해 리포트에 나열하지만 자동으로 고치지 않는다. 3:2 분할은 탐지하지 못한다.
6. **시세가 분할 소급 반영 상태**: Yahoo 의 open/high/low/close/volume 은 이미 분할이 소급 적용된 값이다
   (예: AAPL 2015-01-02 종가가 27.33 으로 내려오지만 실제 체결가는 109.33). 그대로 쓰면 최소가격 필터 같은
   "가격 수준" 규칙에 미래 분할 정보가 샌다. 어댑터가 `splits` 컬럼으로 실제 체결가를 복원한다(`unadjust_splits`).
   Yahoo 는 스핀오프 조정도 `splits` 로 내보내며(GE 등), `splits` 자체가 틀리면 복원도 틀린다.
7. **VWAP 없음**: `$vwap` 피처를 뺐다(Alpha158 → 157개, Alpha360 → 300개).
8. 펀더멘털은 공시일 정보가 없어 사용하지 않는다(가치·퀄리티 팩터 미구현).
9. 섹터는 현재 GICS 분류다(PIT 아님).

## 정제 규칙 요약 (`data/cleaning.py`)
값을 고치지 않는다 — 무효 값은 NaN, 의심 값은 플래그로 남겨 리포트에 노출한다.

| 규칙 | 조건 | 처리 |
|---|---|---|
| 중복 | 같은 (symbol, date) | 첫 행 유지 |
| 무효 가격 | 가격 ≤ 0, OHLC 불일치 | 가격 NaN + `invalid_price` |
| 거래정지 | 거래일인데 행 없음 또는 volume ≤ 0 | `halted`, 당일 `tradable=0` |
| 이상 수익률 | \|수정 일수익률\| > 0.5 | `extreme_return` (목록화) |
| 분할 미반영 의심 | 비수정 종가 비율이 분할 배수 근처이고 factor 불변 | `suspected_split` (목록화) |
| 유동성 | **t-1 까지** 비수정 종가 ≥ $5, 20일 거래대금 중앙값 ≥ $5M | `tradable` → qlib `$tradable` |

## 테스트가 강제하는 것

| 테스트 | 보장 |
|---|---|
| `test_cleaning_has_no_lookahead` | 정제 결과가 미래 데이터와 무관 |
| `test_tradable_does_not_use_same_day_information` | 유동성 필터는 t-1 기준 |
| `test_yahoo_split_unadjust_recovers_traded_prices` | 실제 체결가 복원 |
| `test_pit_membership_is_not_backfilled` | 멤버십 소급 적용 없음 |
| `test_static_list_is_flagged_as_survivorship_biased` | 정적 목록에는 편향 플래그 |
| `test_fundamentals_visible_only_after_report_date` | 공시일 + lag 이후에만 노출 |
| `test_label_matches_documented_timing` | 라벨 = open[t+1+N]/open[t+1]−1 |
| `test_alpha158_alpha360_and_factors_do_not_look_ahead` | 467개 피처 전부 누수 없음 |
| `test_leak_detector_catches_future_features_and_label` | 누수 검사기 자체 검증 |
| `test_walk_forward_folds_are_purged_and_ordered` | 구간 사이 purge |
| `test_non_cross_sectional_processors_are_rejected` | 기간 통계 정규화 금지 |
| `test_handler_output_is_invariant_to_future_data` | 전처리 후 데이터도 미래와 무관 |
| `test_signal_at_t_trades_at_next_open_and_earns_open_to_open` | 백테스터 체결 타이밍 |
| `test_optimizer_does_not_use_future_returns` | 공분산 추정에 미래 수익률 미사용 |
| `test_cvxpy_and_pyportfolioopt_agree` | 최적화 구현 교차검증 |
