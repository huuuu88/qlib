"""라벨 정의와 신호/체결 타이밍. 자세한 설명은 research/pipelines/TIMING.md.

    t 일 장 마감 후  : 신호 계산 (t 일 종가까지의 정보만 사용)
    t+1 일 시가      : 진입 체결            (entry="next_open", 기본)
    t+1+N 일 시가    : 청산 체결
    라벨(t)          = open[t+1+N] / open[t+1] - 1

신호 시점과 체결 시점은 코드에서 서로 다른 함수로 분리한다(signal_date -> execution_date).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

ENTRY_FIELDS = {"next_open": "$open", "next_close": "$close"}


@dataclass(frozen=True)
class LabelSpec:
    horizon: int = 5  # 보유기간 N (거래일)
    entry: str = "next_open"

    def __post_init__(self):
        if self.entry not in ENTRY_FIELDS:
            raise ValueError(f"entry must be one of {list(ENTRY_FIELDS)}")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")

    @property
    def price_field(self) -> str:
        return ENTRY_FIELDS[self.entry]

    @property
    def execution_lag(self) -> int:
        """신호일로부터 체결일까지의 거래일 수. 같은 날 종가 체결(0)은 허용하지 않는다."""
        return 1

    @property
    def span(self) -> int:
        """라벨이 내다보는 미래 거래일 수. t 일 라벨은 t+span 일 가격까지 사용한다 -> purge 길이."""
        return self.execution_lag + self.horizon

    @property
    def expr(self) -> str:
        f = self.price_field
        return f"Ref({f}, -{self.span})/Ref({f}, -{self.execution_lag}) - 1"

    @property
    def name(self) -> str:
        return f"LABEL_{self.entry}_{self.horizon}d"

    def qlib_label(self):
        return [self.expr], [self.name]


def execution_date(signal_date, calendar: Sequence[pd.Timestamp], lag: int = 1) -> pd.Timestamp:
    """신호일 -> 체결일. 캘린더 끝을 넘어가면 예외."""
    cal = pd.DatetimeIndex(calendar)
    i = cal.get_loc(pd.Timestamp(signal_date))
    if i + lag >= len(cal):
        raise IndexError("execution date is beyond the calendar")
    return cal[i + lag]


def label_end_date(signal_date, calendar: Sequence[pd.Timestamp], spec: LabelSpec) -> pd.Timestamp:
    """t 일 라벨이 확정되는 날(= 청산 체결일)."""
    return execution_date(signal_date, calendar, spec.span)
