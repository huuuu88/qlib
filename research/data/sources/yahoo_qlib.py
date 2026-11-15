"""Yahoo 소스: fork 의 qlib 수집 스크립트(scripts/data_collector/yahoo, region US)를 감싼다.

upstream 코드는 수정하지 않고 서브클래싱만 한다.
- 심볼 목록: upstream 은 akshare/나스닥 FTP/NYSE 에서 전 종목을 긁어오지만,
  여기서는 호출자가 준 목록만 받도록 `get_instrument_list` 를 오버라이드한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from .base import DataSource, FetchResult, SourceCapabilities, register_source, validate_prices

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = REPO_ROOT / "scripts"


def _import_collector():
    for p in (str(_SCRIPTS), str(_SCRIPTS / "data_collector" / "yahoo")):
        if p not in sys.path:
            sys.path.append(p)
    import collector  # noqa: WPS433  (scripts/data_collector/yahoo/collector.py)

    return collector


def unadjust_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo 의 open/high/low/close/volume 은 '분할이 소급 반영된' 값이다(배당만 미반영).
    즉 과거 가격 수준에 미래의 분할 정보가 들어 있다. `splits` 컬럼으로 실제 체결 가격/거래량을 복원한다.

    t 일의 실제 가격 = Yahoo 가격 x (t 이후에 일어난 모든 분할 비율의 곱)
    adjclose 는 그대로 둔다 -> factor = adjclose / close 가 분할+배당을 모두 담게 된다.
    `splits` 컬럼이 없으면(기간 중 분할 없음) 아무것도 바꾸지 않는다.
    """
    df = df.sort_values("date").copy()
    if "splits" not in df.columns:
        return df
    ratio = pd.to_numeric(df["splits"], errors="coerce").fillna(0.0)
    ratio = ratio.where(ratio > 0, 1.0)
    # t 이후(당일 제외) 분할 비율의 누적곱
    future = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * future
    df["volume"] = df["volume"] / future
    return df


def make_collector_class(symbols: Sequence[str]):
    collector = _import_collector()

    class _FixedListCollectorUS1d(collector.YahooCollectorUS1d):
        def get_instrument_list(self):
            return list(symbols)

    return _FixedListCollectorUS1d


@register_source("yahoo_qlib")
class YahooQlibSource(DataSource):
    name = "yahoo_qlib"
    capabilities = SourceCapabilities(
        includes_delisted=False,
        pit_membership=False,
        pit_fundamentals=False,
        notes=(
            "상장폐지 종목 시세를 제공하지 않음 -> 생존편향",
            "지수 편입 이력(PIT membership) 없음",
            "수정주가는 조회 시점 기준으로 재계산되어 다운로드 시점마다 값이 달라짐",
            "분할 미반영/오반영 가능성 있음(정제 단계에서 탐지만 하고 자동 수정하지 않음)",
            "시세가 분할 소급 반영 상태로 제공됨 -> splits 컬럼으로 실제 체결가를 복원(스핀오프 조정 포함, splits 오류 시 복원도 틀림)",
            "VWAP 미제공",
        ),
    )

    def __init__(self, delay: float = 0.5, max_collector_count: int = 2):
        self.delay = delay
        self.max_collector_count = max_collector_count

    def fetch_prices(self, symbols: Sequence[str], start: str, end: str, raw_dir) -> FetchResult:
        raw_dir = Path(raw_dir)
        if raw_dir.exists() and any(raw_dir.glob("*.csv")):
            # upstream save_instrument 은 기존 파일에 append 하므로 중복 방지를 위해 빈 폴더를 요구
            raise FileExistsError(f"{raw_dir} already has csv files; use a fresh snapshot dir or --skip-download")
        raw_dir.mkdir(parents=True, exist_ok=True)
        cls = make_collector_class(symbols)
        cls(
            save_dir=raw_dir,
            start=start,
            end=end,
            interval="1d",
            max_workers=1,
            max_collector_count=self.max_collector_count,
            delay=self.delay,
        ).collector_data()
        return self.load_raw(symbols, raw_dir)

    @staticmethod
    def load_raw(symbols: Sequence[str], raw_dir) -> FetchResult:
        """raw_dir 의 스냅샷을 표준 스키마로 읽는다(재다운로드 없이 재현할 때 사용)."""
        from qlib.utils import code_to_fname

        frames, failed = [], []
        for sym in symbols:
            fp = Path(raw_dir) / f"{code_to_fname(sym).upper()}.csv"
            if not fp.exists():
                failed.append(sym)
                continue
            df = pd.read_csv(fp, dtype={"symbol": str}, keep_default_na=False, na_values=[""])
            df["symbol"] = sym
            frames.append(unadjust_splits(df))
        prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["symbol", "date"])
        if frames:
            prices["date"] = pd.to_datetime(prices["date"].astype(str).str.slice(0, 10))
            prices = validate_prices(prices)
        return FetchResult(prices=prices, requested=list(symbols), failed=failed)

    def fetch_calendar(self, start: str, end: str) -> List[pd.Timestamp]:
        _import_collector()
        from data_collector.utils import get_calendar_list

        cal = pd.DatetimeIndex(get_calendar_list("US_ALL")).tz_localize(None).normalize().unique().sort_values()
        return list(cal[(cal >= pd.Timestamp(start)) & (cal < pd.Timestamp(end))])
