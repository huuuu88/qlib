"""펀더멘털의 공시일(report_date) 기준 정렬.

회계기간 종료일(period_end)이 아니라 '시장에 공개된 날' 이후에만 값이 보이도록 한다.
값은 report_date 로부터 `lag_days` 거래일이 지난 날부터 사용 가능하다(기본 1: 장 마감 후 공시 가정).
Yahoo 소스는 PIT 펀더멘털이 없으므로 Phase 1 에서는 이 함수와 테스트만 제공한다.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def align_to_report_date(
    fundamentals: pd.DataFrame,
    calendar: Sequence[pd.Timestamp],
    value_cols: Sequence[str],
    lag_days: int = 1,
    report_date_col: str = "report_date",
) -> pd.DataFrame:
    """(symbol, report_date, values...) -> (symbol, date) 일별 패널. 각 date 에는 그 시점에 공개된 최신 값만 온다."""
    if lag_days < 0:
        raise ValueError("lag_days must be >= 0")
    if report_date_col not in fundamentals.columns:
        raise ValueError(f"fundamentals must have `{report_date_col}`; period-end dates are not acceptable")
    cal = pd.DatetimeIndex(sorted(pd.DatetimeIndex(calendar).unique()))
    f = fundamentals.copy()
    f[report_date_col] = pd.to_datetime(f[report_date_col])
    # 공시일 이후(당일 포함) 첫 거래일의 위치 + lag
    pos = cal.searchsorted(f[report_date_col].values, side="left") + lag_days
    f = f[pos < len(cal)].copy()
    f["date"] = cal[pos[pos < len(cal)]]
    out = []
    for sym, g in f.sort_values(report_date_col).groupby("symbol"):
        g = g.drop_duplicates("date", keep="last").set_index("date")[list(value_cols)]
        panel = g.reindex(cal).ffill()
        panel.loc[panel.index < g.index.min()] = np.nan
        panel["symbol"] = sym
        out.append(panel.rename_axis("date").reset_index())
    if not out:
        return pd.DataFrame(columns=["date", "symbol", *value_cols])
    return pd.concat(out, ignore_index=True)[["date", "symbol", *value_cols]]
