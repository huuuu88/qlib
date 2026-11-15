import numpy as np
import pandas as pd
import pytest

from research.data.sources.base import DataSource, FetchResult, SourceCapabilities, validate_prices


def make_prices(symbols=("AAA", "BBB", "CCC"), start="2020-01-01", periods=300, seed=0, price0=50.0):
    """결정적 합성 시세(표준 스키마). adjclose == close (배당/분할 없음)."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start, periods=periods)
    frames = []
    for i, s in enumerate(symbols):
        ret = rng.normal(0.0003, 0.015, size=periods)
        close = price0 * (1 + i) * np.cumprod(1 + ret)
        open_ = close * (1 + rng.normal(0, 0.003, size=periods))
        high = np.maximum(open_, close) * 1.005
        low = np.minimum(open_, close) * 0.995
        vol = rng.integers(1_000_000, 3_000_000, size=periods).astype(float)
        frames.append(
            pd.DataFrame(
                {"symbol": s, "date": cal, "open": open_, "high": high, "low": low, "close": close,
                 "adjclose": close, "volume": vol}
            )
        )
    return validate_prices(pd.concat(frames, ignore_index=True)), list(cal)


class FakePITSource(DataSource):
    """PIT 멤버십과 상폐 종목을 제공하는 유료 소스를 흉내 낸다(소스 교체 테스트용)."""

    name = "fake_pit"
    capabilities = SourceCapabilities(includes_delisted=True, pit_membership=True, pit_fundamentals=True)

    def __init__(self, prices, calendar, membership):
        self._p, self._c, self._m = prices, calendar, membership

    def fetch_prices(self, symbols, start, end, raw_dir):
        return FetchResult(self._p[self._p["symbol"].isin(symbols)], list(symbols), [])

    def load_raw(self, symbols, raw_dir):
        return self.fetch_prices(symbols, None, None, raw_dir)

    def fetch_calendar(self, start, end):
        return self._c

    def fetch_membership(self, universe, symbols, prices):
        return self._m


@pytest.fixture
def synth():
    return make_prices()
