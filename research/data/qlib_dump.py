"""정제된 표준 가격 -> qlib bin 포맷.

정규화(수정주가, factor, change)와 bin 덤프는 fork 의 upstream 스크립트 로직을 그대로 쓴다
(YahooNormalize1d, dump_bin.DumpDataAll). 여기서는 캘린더 주입과 추가 필드(tradable) 병합만 한다.

qlib 필드 의미 (upstream 규약)
- $open/$high/$low/$close: 수정주가를 종목 첫 종가로 나눈 값(수준은 무의미, 비율만 유효)
- $factor: 수정계수.  비수정 종가 = $close / $factor
- $volume: 수정 거래량.  거래대금(USD) = $close * $volume  (수정/스케일에 불변)
- $change: 일수익률
- $tradable: research.data.cleaning 의 유동성 필터(t-1 기준)

주의: 수정주가는 '미래의 배당/분할'을 반영해 과거 수준을 바꾼다. 수익률·비율 피처는 안전하지만
가격 '수준' 자체를 피처로 쓰면 미래 정보가 섞인다. 수준이 필요하면 $close/$factor 를 쓸 것.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from .sources.yahoo_qlib import _import_collector

BASE_FIELDS = ["open", "close", "high", "low", "volume", "factor", "change"]


def _make_normalizer(calendar: List[pd.Timestamp]):
    collector = _import_collector()

    class _Normalizer(collector.YahooNormalize1d):
        def _get_calendar_list(self):
            return list(calendar)

    return _Normalizer()


def normalize_to_csv(
    cleaned: pd.DataFrame, calendar: List[pd.Timestamp], norm_dir, extra_fields: Sequence[str] = ("tradable",)
) -> int:
    from qlib.utils import code_to_fname

    norm_dir = Path(norm_dir)
    if norm_dir.exists():
        shutil.rmtree(norm_dir)
    norm_dir.mkdir(parents=True)
    normalizer = _make_normalizer(calendar)
    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "adjclose"]
    n = 0
    for sym, g in cleaned.groupby("symbol", sort=True):
        src = g.loc[~g["missing_row"], cols].copy()
        if src["close"].notna().sum() == 0:
            continue
        out = normalizer.normalize(src)
        out["date"] = pd.to_datetime(out["date"])
        out = out.merge(g[["date", *extra_fields]], on="date", how="left")
        out.to_csv(norm_dir / f"{code_to_fname(sym).upper()}.csv", index=False)
        n += 1
    return n


def dump_bin(norm_dir, qlib_dir, extra_fields: Sequence[str] = ("tradable",), max_workers: int = 4) -> None:
    _import_collector()
    from dump_bin import DumpDataAll  # scripts/dump_bin.py

    qlib_dir = Path(qlib_dir)
    if qlib_dir.exists():
        shutil.rmtree(qlib_dir)
    DumpDataAll(
        data_path=str(norm_dir),
        qlib_dir=str(qlib_dir),
        max_workers=max_workers,
        include_fields=",".join([*BASE_FIELDS, *extra_fields]),
    ).dump()
