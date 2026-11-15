"""데이터 소스 추상화 계층.

상위 코드(정제, qlib 변환, 품질 리포트)는 이 모듈의 인터페이스와 표준 스키마에만 의존한다.
Yahoo -> Sharadar/Norgate/CRSP/Polygon 으로 교체할 때는 `DataSource` 구현체만 추가하고
config 의 `source:` 키만 바꾸면 된다.

표준 가격 스키마 (long format, 종목-일자당 1행):
    symbol, date, open, high, low, close, adjclose, volume
- open/high/low/close/volume: 비수정(raw) 값
- adjclose: 분할·배당 반영 수정 종가. factor = adjclose / close

표준 멤버십 스키마 (point-in-time 구간, 한 종목이 여러 구간 가질 수 있음):
    symbol, start, end      (양끝 포함)
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Sequence

import pandas as pd

PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adjclose", "volume"]
MEMBERSHIP_COLUMNS = ["symbol", "start", "end"]


@dataclass(frozen=True)
class SourceCapabilities:
    """소스가 보장하는 무결성 속성. 결과물의 편향 플래그는 여기서 자동 도출된다."""

    includes_delisted: bool  # 상장폐지 종목의 과거 시세를 제공하는가
    pit_membership: bool  # 시점별 유니버스(지수 편입 이력)를 제공하는가
    pit_fundamentals: bool  # 공시일 기준 펀더멘털을 제공하는가
    notes: tuple = field(default_factory=tuple)

    @property
    def survivorship_bias(self) -> bool:
        return not (self.includes_delisted and self.pit_membership)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["notes"] = list(self.notes)
        d["survivorship_bias"] = self.survivorship_bias
        return d


@dataclass
class FetchResult:
    prices: pd.DataFrame  # PRICE_COLUMNS
    requested: List[str]
    failed: List[str]  # 요청했지만 데이터를 받지 못한 종목


class DataSource(abc.ABC):
    name: str
    capabilities: SourceCapabilities

    @abc.abstractmethod
    def fetch_prices(self, symbols: Sequence[str], start: str, end: str, raw_dir) -> FetchResult:
        """비수정 OHLCV + adjclose 를 표준 스키마로 반환. 원본은 raw_dir 에 그대로 보존한다."""

    @abc.abstractmethod
    def load_raw(self, symbols: Sequence[str], raw_dir) -> FetchResult:
        """raw_dir 에 보존된 스냅샷을 재다운로드 없이 표준 스키마로 읽는다(재현용)."""

    @abc.abstractmethod
    def fetch_calendar(self, start: str, end: str) -> List[pd.Timestamp]:
        """거래일 캘린더."""

    def fetch_membership(self, universe: str, symbols: Sequence[str], prices: pd.DataFrame) -> pd.DataFrame:
        """PIT 멤버십 구간. PIT 를 지원하지 않는 소스는 NotImplementedError 를 내야 하며,
        호출측(universe.build_membership)이 '정적 목록 + 생존편향 플래그'로 대체한다."""
        raise NotImplementedError(f"{self.name} does not provide point-in-time membership")

    def fetch_fundamentals(self, symbols: Sequence[str], start: str, end: str) -> pd.DataFrame:
        """컬럼에 반드시 `report_date`(공시일) 포함. research.data.fundamentals 참고."""
        raise NotImplementedError(f"{self.name} does not provide fundamentals")


_REGISTRY: Dict[str, Callable[..., DataSource]] = {}


def register_source(name: str):
    def deco(factory):
        _REGISTRY[name] = factory
        return factory

    return deco


def get_source(name: str, **kwargs) -> DataSource:
    if name not in _REGISTRY:
        # 구현 모듈을 지연 임포트해 레지스트리를 채운다
        import importlib

        importlib.import_module(f"research.data.sources.{name}")
    if name not in _REGISTRY:
        raise KeyError(f"unknown data source: {name}. registered={sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """표준 스키마 강제. 모든 소스 구현체는 반환 직전에 호출한다."""
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"price frame missing columns: {missing}")
    out = df[PRICE_COLUMNS].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    for c in PRICE_COLUMNS[2:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)
