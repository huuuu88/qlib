# 클린 환경 재현 기록 (2026-11-16)

README 의 명령만으로 기준선 결과가 다시 나오는지 실제로 확인한 기록이다.

## 절차
1. 레포를 새 폴더에 `git clone` (브랜치 `us-quant-research`, 커밋 `25d4e5f0`). 작업 트리에는 git 에 있는 파일만 있다.
2. 새 conda env(`usquant-repro`, Python 3.11) 생성 → `pip install -r research/requirements.lock.txt` → `pip install -e . --no-deps` (PowerShell).
3. raw 스냅샷(`store/sp500/raw`, `raw_meta`)만 복사. 가공 데이터·모델·예측은 복사하지 않았다.
4. `python -m pytest research/tests -q`
5. `python -m research.data.build --config research/configs/data_sp500.yaml --skip-download`
6. `python -m research.pipelines.walk_forward --data sp500 --experiment alpha158_lgb`

## 결과

| 항목 | 원본 | 클린 환경 | 일치 |
|---|---|---|---|
| 테스트 | 42 passed | 41 passed, 1 skipped (smoke50 스토어가 없어 해당 테스트만 스킵) | – |
| 데이터 해시 raw | `eb0da632cf9d4661…` | `eb0da632cf9d4661…` | 예 |
| 데이터 해시 qlib | `8d3d80297f7615e1…` | `8d3d80297f7615e1…` | 예 |
| 데이터 해시 meta | `cfb5f592168edb95…` | `cfb5f592168edb95…` | 예 |
| IC 평균 | 0.0069755778963849074 | 0.0069755778963849074 | 예 |
| Rank IC 평균 | 0.0042853477639618045 | 0.0042853477639618045 | 예 |
| Rank ICIR | 0.042992707540688437 | 0.042992707540688437 | 예 |
| fold 별 Rank IC (6개) | – | – | 전부 동일 |
| fold 별 LightGBM best iteration | 3, 1, 1, 4, 5, 1 | 3, 1, 1, 4, 5, 1 | 예 |

추가 교차 확인: `qrun research/configs/baseline_alpha158_lgb.yaml` 의 Rank IC `0.007457745951492716` 은
walk-forward fold 5 의 값과 같다.

## 재현 중 발견해 고친 것
- lock 파일을 `pip freeze` 로 만들었더니 conda 가 설치한 패키지의 로컬 경로(`@ file:///…`)가 들어가 새 환경에서 설치가 실패했다.
  `pip list --format=freeze` 형식으로 다시 만들었다.
- qlib 이 한글 주석이 든 yaml 을 시스템 기본 인코딩(cp949)으로 읽다 실패한다. `PYTHONUTF8=1` 이 필요하다(README 반영).
- `qrun` 은 현재 폴더를 모듈 경로에 넣지 않는다. `PYTHONPATH=.` 가 필요하다(README 반영).

## 범위와 한계
- 재현한 것은 **기준선(alpha158_lgb)의 데이터 빌드와 walk-forward 지표**다. 나머지 5개 실험, Phase 4 백테스트, 최종 리포트는
  클린 환경에서 다시 돌리지 않았다(같은 코드·같은 시드이므로 재현될 것으로 예상하지만 확인하지는 않았다).
- 같은 PC·같은 OS·같은 CPU 에서의 재현이다. 다른 하드웨어에서 LightGBM 결과가 비트 단위로 같은지는 확인하지 않았다.
- Yahoo 다운로드부터 다시 하면 수정주가가 달라져 해시와 결과가 바뀐다. raw 스냅샷은 git 에 없으므로 별도로 보관해야 한다.
- 클린 환경의 qlib 버전 문자열은 `0.9.9.dev4` 로 나온다(태그 v0.9.8 이후 커밋이 4개 쌓였기 때문). 소스는 같은 fork 다.
