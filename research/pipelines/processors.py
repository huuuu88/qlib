"""qlib 전처리기 정책: 횡단면(해당 일자 내) 변환만 허용한다.

전체 기간/학습 기간 통계를 쓰는 전처리기(ZScoreNorm, MinMaxNorm, RobustZScoreNorm 등)는
config 검증 단계에서 거부한다.
"""

from __future__ import annotations

import pandas as pd
from qlib.data.dataset.processor import Processor

ALLOWED = {
    "DropnaLabel",
    "DropnaProcessor",
    "CSZScoreNorm",
    "CSRankNorm",
    "CSZFillna",
    "Fillna",
    "FilterCol",
    "DropCol",
    "ProcessInf",
    "TradableFilter",
    "CSWinsorize",
}
FORBIDDEN = {"ZScoreNorm", "MinMaxNorm", "RobustZScoreNorm"}


def _name(p) -> str:
    if isinstance(p, dict):
        return str(p["class"]).split(".")[-1]
    if isinstance(p, str):
        return p.split(".")[-1]
    return type(p).__name__


def assert_cross_sectional_only(*processor_lists) -> None:
    for procs in processor_lists:
        for p in procs or []:
            n = _name(p)
            if n in FORBIDDEN or n not in ALLOWED:
                raise ValueError(
                    f"processor {n!r} is not allowed: only cross-sectional (per-date) transforms may be used"
                )


class TradableFilter(Processor):
    """('meta','tradable') < 0.5 인 행을 제거하고 meta 그룹을 버린다. 행 단위 연산이라 fit 이 없다."""

    def __init__(self, group: str = "meta", col: str = "tradable"):
        self.group, self.col = group, col

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if (self.group, self.col) not in df.columns:
            return df
        keep = df[(self.group, self.col)] > 0.5
        return df.loc[keep].drop(columns=[self.group], level=0)

    def readonly(self):
        return False


class CSWinsorize(Processor):
    """일자별 분위수로 자른다(횡단면)."""

    def __init__(self, fields_group="feature", lower=0.01, upper=0.99):
        self.fields_group, self.lower, self.upper = fields_group, lower, upper

    def __call__(self, df):
        cols = df.columns[df.columns.get_level_values(0) == self.fields_group]
        g = df[cols].groupby(level="datetime", group_keys=False)
        lo = g.transform(lambda x: x.quantile(self.lower))
        hi = g.transform(lambda x: x.quantile(self.upper))
        df[cols] = df[cols].clip(lower=lo, upper=hi)
        return df
