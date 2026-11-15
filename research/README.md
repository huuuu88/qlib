# US 크로스섹션 퀀트 리서치 파이프라인

`microsoft/qlib` fork(v0.9.8) 위에 얹은 리서치 코드. upstream 파일은 수정하지 않고 모든 코드는 `research/` 아래에 둔다.
목표는 수익률이 아니라 **결과를 믿을 수 있는 구조**다.

```
research/
  data/        수집·정제·qlib 변환   (원천/가공 데이터는 data/store/, gitignore)
  factors/     커스텀 팩터            (Phase 3)
  configs/     실험·데이터 설정
  pipelines/   walk-forward, 학습, 평가 (Phase 2~3)
  portfolio/   최적화, 제약, 비용모델  (Phase 4)
  reports/     자동 생성 리포트
  tests/       데이터/누수 검증 테스트
```

## 처음부터 재현하기 (Windows 기준)

```bash
# 1) conda env (Miniforge)
conda create -y -n usquant python=3.11 pip
conda activate usquant

# 2) fork(qlib v0.9.8)를 소스에서 빌드 — C 컴파일러 필요
#    (PyPI 에는 0.9.7 wheel 까지만 있고, fork 의 수집 스크립트는 0.9.8 전용 모듈을 쓴다)
#    winget install Microsoft.VisualStudio.2022.BuildTools  (C++ 빌드 도구 워크로드)
#    주의: Git Bash 에서는 setuptools 가 MSVC 를 찾지 못했다. PowerShell/cmd 에서 실행할 것.
pip install -r research/requirements.lock.txt
pip install -e . --no-deps

# 3) 테스트
python -m pytest research/tests -q

# 4) Phase 1 데이터 빌드 (Yahoo 50종목: 수집 → 정제 → qlib 변환 → 해시 → 검증 → 리포트)
python -m research.data.build --config research/configs/data_smoke50.yaml
#    보존된 raw 스냅샷으로 재빌드(다운로드 없음, 같은 해시가 나와야 함)
python -m research.data.build --config research/configs/data_smoke50.yaml --skip-download
```

산출물
- `research/data/store/<name>/` : raw 스냅샷, 정규화 csv, qlib bin, `manifest.json` (gitignore)
- `research/reports/phase1/` : `data_quality.md`, `data_quality.json`, `manifest.json`, `e2e_verify.json`

## 데이터 레이어 설계

| 계층 | 모듈 | 역할 |
|---|---|---|
| 소스 추상화 | `data/sources/base.py` | `DataSource` 인터페이스, 표준 스키마, `SourceCapabilities` |
| Yahoo 어댑터 | `data/sources/yahoo_qlib.py` | fork 의 `scripts/data_collector/yahoo` 를 서브클래싱 |
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
4. config 의 `source:` 만 변경 — 정제·변환·리포트 코드는 그대로

편향 플래그는 `capabilities` 와 멤버십 종류에서 **자동 도출**되어 `manifest.json` 과
`<qlib_dir>/research_meta.json` 에 기록된다. 하위 단계는 `load_data_flags(qlib_dir)` 로 읽어 결과에 표기한다.
메타 파일이 없으면 "생존편향 있음"으로 가정한다.

## Yahoo 데이터의 한계 (반드시 읽을 것)

1. **생존편향**: 상장폐지 종목의 시세를 주지 않는다. 지금 살아 있는 종목만으로 과거를 백테스트하게 되어 성과가 상향 편향된다.
   빌드할 때마다 폐지 종목(`delisted_probes`)을 실제로 요청해 결과를 리포트에 남긴다.
2. **PIT 지수 편입 이력 없음**: 유니버스가 "현재 목록을 과거 전 기간에 소급 적용"한 것이 된다(`static_current_list`).
3. **수정주가의 비재현성**: adjclose 는 조회 시점 기준으로 재계산된다. 같은 기간을 다시 받아도 값이 달라지므로
   **raw 스냅샷과 그 해시가 데이터 버전**이다. 재현은 `--skip-download` 로 한다.
4. **수정주가와 미래 정보**: 수정주가의 과거 "수준"은 이후의 배당·분할을 반영한다. 수익률·비율 피처는 안전하지만
   가격 수준을 직접 피처로 쓰면 미래 정보가 섞인다. 비수정 종가는 `$close/$factor` 로 얻는다.
5. **분할 오류 가능성**: 미반영/오반영을 탐지해 리포트에 나열하지만 자동으로 고치지 않는다.
6. **시세가 분할 소급 반영 상태**: Yahoo 의 open/high/low/close/volume 은 이미 분할이 소급 적용된 값이다
   (예: AAPL 2015-01-02 종가가 27.33 으로 내려오지만 실제 체결가는 109.33). 그대로 쓰면 최소가격 필터 같은
   "가격 수준" 규칙에 미래 분할 정보가 샌다. 어댑터가 `splits` 컬럼으로 실제 체결가를 복원한다(`unadjust_splits`).
   Yahoo 는 스핀오프 조정도 `splits` 로 내보내며(GE 등), `splits` 자체가 틀리면 복원도 틀린다.
7. **VWAP 없음**: Alpha158 의 `$vwap` 계열 피처는 Yahoo 데이터로는 계산되지 않는다(Phase 3 에서 처리 필요).
8. 펀더멘털은 공시일 정보가 없어 사용하지 않는다.

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

`tradable` 은 t-1 까지의 정보만 쓰며, 이는 테스트(`test_cleaning_has_no_lookahead`,
`test_tradable_does_not_use_same_day_information`)로 강제된다.
