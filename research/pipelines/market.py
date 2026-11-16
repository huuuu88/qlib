"""qlib 스토어에서 (일자 x 종목) 패널을 읽는다. qlib 초기화도 여기서 한다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research.data.manifest import REPO_ROOT, load_data_flags


MLRUNS = REPO_ROOT / "research" / "mlruns"  # MLflow 파일 스토어 (gitignore)


@dataclass
class Panels:
    open: pd.DataFrame  # 수정 시가
    close: pd.DataFrame  # 수정 종가
    ret: pd.DataFrame  # 종가-종가 일수익률
    tradable: pd.DataFrame  # t-1 기준 유동성 필터 (bool)
    member: pd.DataFrame  # 유니버스 멤버 여부 (bool)
    adv: pd.DataFrame  # 20일 거래대금 중앙값 (USD)
    vol: pd.DataFrame  # 60일 일수익률 표준편차
    calendar: pd.DatetimeIndex


def store_dir(name: str) -> Path:
    return REPO_ROOT / "research" / "data" / "store" / name


def init(name: str) -> dict:
    """qlib 초기화. 데이터 메타(해시, 편향 플래그)를 반환한다."""
    import qlib

    qdir = store_dir(name) / "qlib"
    qlib.init(
        provider_uri=str(qdir), region="us", kernels=1, expression_cache=None, dataset_cache=None,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": MLRUNS.as_uri(), "default_exp_name": "default"},
        },
    )
    meta = load_data_flags(qdir)
    mf = store_dir(name) / "manifest.json"
    if mf.exists():
        meta["manifest"] = json.loads(mf.read_text(encoding="utf-8"))
    return meta


def load_panels(universe: str) -> Panels:
    from qlib.data import D

    f = D.features(D.instruments("all"), ["$open", "$close", "$volume", "$tradable"], freq="day")
    f.columns = ["open", "close", "volume", "tradable"]
    wide = {c: f[c].unstack(level="instrument") for c in f.columns}
    cal = pd.DatetimeIndex(D.calendar(freq="day"))
    wide = {k: v.reindex(cal) for k, v in wide.items()}
    cols = wide["close"].columns

    m = D.features(D.instruments(universe), ["$close"], freq="day")
    member = (
        pd.Series(True, index=m.index).unstack(level="instrument").reindex(index=cal, columns=cols).fillna(False).astype(bool)
    )
    c = wide["close"].ffill()
    ret = (c / c.shift(1) - 1).where(wide["close"].notna())
    dv = wide["close"] * wide["volume"]
    return Panels(
        open=wide["open"],
        close=wide["close"],
        ret=ret,
        tradable=wide["tradable"].fillna(0) > 0.5,
        member=member,
        adv=dv.rolling(20, min_periods=10).median(),
        vol=ret.rolling(60, min_periods=20).std(),
        calendar=cal,
    )


def load_sectors(name: str, columns) -> pd.Series:
    """현재 GICS 섹터(PIT 아님). 없으면 Unknown."""
    fp = store_dir(name) / "sectors.csv"
    if not fp.exists():
        return pd.Series("Unknown", index=columns)
    s = pd.read_csv(fp, index_col=0).iloc[:, 0]
    return s.reindex(columns).fillna("Unknown")
