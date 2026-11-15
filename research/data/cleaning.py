"""정제 규칙. 입력/출력 모두 표준 가격 스키마(research.data.sources.base).

원칙
- 값을 '고치지' 않는다. 무효 값은 NaN 으로 만들고, 의심 값은 플래그로 남겨 리포트에 노출한다.
- 모든 규칙은 해당 일자까지의 정보만 사용한다(룩어헤드 금지). `tradable` 은 t-1 까지의 정보로 계산한다.

규칙
1. 중복 (symbol, date) 제거(첫 행 유지)
2. 무효 가격: 가격 <= 0 또는 OHLC 불일치(high < low, close/open 이 [low, high] 밖) -> 해당 행 가격 NaN, 플래그
3. 거래정지: 캘린더상 거래일인데 행이 없거나 volume <= 0 -> `halted`
4. 이상 수익률: |수정 일수익률| > extreme_return_threshold -> `extreme_return`
5. 분할 미반영 의심: 비수정 종가 비율이 분할 배수(2,3,4,5,10 등)에 가깝고 factor 가 변하지 않음 -> `suspected_split`
6. 유동성 필터: 직전 adv_window 거래일의 비수정 종가 >= min_price 그리고
   거래대금 중앙값 >= min_median_dollar_volume -> `tradable` (t-1 기준, 1/0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# 3:2 분할(1.5)은 일반적인 급등락과 구분되지 않아 제외한다(미탐 가능성은 리포트 한계로 남김)
SPLIT_RATIOS = (2, 3, 4, 5, 7, 10, 20)
FLAG_COLUMNS = ["invalid_price", "halted", "extreme_return", "suspected_split"]


@dataclass
class CleaningConfig:
    min_price: float = 5.0
    min_median_dollar_volume: float = 5.0e6
    adv_window: int = 20
    extreme_return_threshold: float = 0.5
    split_ratio_tolerance: float = 0.03
    halt_action: str = "flag"

    @classmethod
    def from_dict(cls, d: dict) -> "CleaningConfig":
        unknown = set(d or {}) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown cleaning config keys: {sorted(unknown)}")
        # YAML 이 `5.0e6` 같은 값을 문자열로 읽는 경우가 있어 선언된 타입으로 강제 변환한다
        kw = {k: type(getattr(cls, k))(v) for k, v in (d or {}).items()}
        return cls(**kw)


def _clean_one(df: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: CleaningConfig) -> pd.DataFrame:
    sym = df["symbol"].iloc[0]
    df = df.drop_duplicates("date", keep="first").set_index("date").sort_index()
    n_dup = 0  # 중복 수는 호출측에서 집계

    # 종목 생존 구간 내 캘린더로 재색인 -> 빠진 거래일이 드러난다
    cal = calendar[(calendar >= df.index.min()) & (calendar <= df.index.max())]
    off_calendar = df.index.difference(cal)
    df = df.reindex(cal)
    df["symbol"] = sym
    df["missing_row"] = df["close"].isna() & df["volume"].isna()

    px = ["open", "high", "low", "close"]
    nonpos = (df[px] <= 0).any(axis=1) | (df["adjclose"] <= 0)
    tol = 1e-6
    incons = (
        (df["high"] < df["low"] * (1 - tol))
        | (df["close"] > df["high"] * (1 + tol))
        | (df["close"] < df["low"] * (1 - tol))
        | (df["open"] > df["high"] * (1 + tol))
        | (df["open"] < df["low"] * (1 - tol))
    )
    df["invalid_price"] = (nonpos | incons).fillna(False)
    df.loc[df["invalid_price"], px + ["adjclose"]] = np.nan

    df["halted"] = df["missing_row"] | (df["volume"].fillna(0) <= 0)

    adj = df["adjclose"].ffill()
    df["adj_return"] = adj / adj.shift(1) - 1
    df["extreme_return"] = (df["adj_return"].abs() > cfg.extreme_return_threshold).fillna(False)

    raw = df["close"].ffill()
    ratio = raw / raw.shift(1)
    factor = (df["adjclose"] / df["close"]).ffill()
    factor_change = (factor / factor.shift(1) - 1).abs()
    near_split = pd.Series(False, index=df.index)
    for r in SPLIT_RATIOS:
        for target in (r, 1.0 / r):
            near_split |= (ratio / target - 1).abs() < cfg.split_ratio_tolerance
    df["suspected_split"] = (near_split & (factor_change < 0.01)).fillna(False)

    # 유동성: t-1 까지의 정보만 사용
    dollar_vol = df["close"] * df["volume"]
    med_dv = dollar_vol.rolling(cfg.adv_window, min_periods=cfg.adv_window).median().shift(1)
    prev_close = raw.shift(1)
    df["tradable"] = (
        (prev_close >= cfg.min_price) & (med_dv >= cfg.min_median_dollar_volume) & ~df["halted"]
    ).astype(float)

    df.attrs["off_calendar_rows"] = len(off_calendar)
    df.index.name = "date"
    return df.reset_index()


def clean_prices(
    prices: pd.DataFrame, calendar: List[pd.Timestamp], cfg: CleaningConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """반환: (정제된 가격+플래그, 플래그 발생 이벤트 목록, 집계 통계)"""
    calendar = pd.DatetimeIndex(calendar)
    n_dup = int(prices.duplicated(["symbol", "date"]).sum())
    out, off_cal = [], 0
    for _, g in prices.groupby("symbol", sort=True):
        c = _clean_one(g, calendar, cfg)
        off_cal += c.attrs.get("off_calendar_rows", 0)
        out.append(c)
    cleaned = pd.concat(out, ignore_index=True)

    ev = cleaned.melt(
        id_vars=["symbol", "date", "close", "adjclose", "volume", "adj_return"],
        value_vars=FLAG_COLUMNS,
        var_name="flag",
    )
    events = ev[ev["value"]].drop(columns="value").sort_values(["flag", "symbol", "date"]).reset_index(drop=True)

    stats = {
        "duplicate_rows_removed": n_dup,
        "off_calendar_rows_dropped": int(off_cal),
        "rows": int(len(cleaned)),
        **{f"n_{f}": int(cleaned[f].sum()) for f in FLAG_COLUMNS},
        "tradable_fraction": float(cleaned["tradable"].mean()),
    }
    return cleaned, events, stats
