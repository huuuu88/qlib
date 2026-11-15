"""시점별(point-in-time) 유니버스.

멤버십은 항상 (symbol, start, end) 구간 테이블이다. qlib 의 instruments/<name>.txt 가 같은 구조라
그대로 내려쓴다. '현재 구성종목을 과거에 소급 적용'하는 경로는 `static_current_list` 하나뿐이며,
이 경우 survivorship_bias=True 가 메타데이터에 기록되어 하위 리포트에 자동 표기된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .sources.base import MEMBERSHIP_COLUMNS, DataSource

KIND_PIT = "point_in_time"
KIND_STATIC = "static_current_list"


@dataclass
class Membership:
    name: str
    table: pd.DataFrame  # MEMBERSHIP_COLUMNS
    kind: str

    @property
    def survivorship_bias(self) -> bool:
        return self.kind != KIND_PIT

    def members_on(self, date) -> list:
        d = pd.Timestamp(date)
        t = self.table
        return sorted(t.loc[(t["start"] <= d) & (t["end"] >= d), "symbol"].unique())


def _clip_to_data(table: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """멤버십 구간을 실제 시세가 존재하는 구간으로 자른다(데이터 없는 구간에 종목이 나타나지 않게)."""
    span = prices.dropna(subset=["close"]).groupby("symbol")["date"].agg(["min", "max"])
    t = table.merge(span, left_on="symbol", right_index=True, how="inner")
    t["start"] = t[["start", "min"]].max(axis=1)
    t["end"] = t[["end", "max"]].min(axis=1)
    t = t[t["start"] <= t["end"]]
    return t[MEMBERSHIP_COLUMNS].sort_values(["symbol", "start"]).reset_index(drop=True)


def build_membership(source: DataSource, name: str, symbols: Sequence[str], prices: pd.DataFrame) -> Membership:
    try:
        table = source.fetch_membership(name, symbols, prices)
        kind = KIND_PIT
    except NotImplementedError:
        # PIT 불가: 주어진 목록을 전 기간에 적용(= 소급 적용). 반드시 플래그가 붙는다.
        table = pd.DataFrame(
            {"symbol": list(symbols), "start": pd.Timestamp("1900-01-01"), "end": pd.Timestamp("2200-01-01")}
        )
        kind = KIND_STATIC
    table = table[MEMBERSHIP_COLUMNS].copy()
    table["start"] = pd.to_datetime(table["start"])
    table["end"] = pd.to_datetime(table["end"])
    return Membership(name=name, table=_clip_to_data(table, prices), kind=kind)


def write_instruments(membership: Membership, qlib_dir) -> Path:
    from qlib.utils import code_to_fname

    path = Path(qlib_dir) / "instruments" / f"{membership.name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    t = membership.table
    lines = [
        f"{code_to_fname(s).upper()}\t{a:%Y-%m-%d}\t{b:%Y-%m-%d}" for s, a, b in zip(t["symbol"], t["start"], t["end"])
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
