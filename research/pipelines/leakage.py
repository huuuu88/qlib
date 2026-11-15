"""피처 누수(룩어헤드) 검사.

방법: cut 시점 이후 데이터를 '아예 없는' qlib 스토어와 전체 스토어에서 같은 피처를 계산해
cut 이전 값이 동일한지 비교한다. 미래 정보를 쓰는 피처는 두 스토어에서 값이 달라진다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from research.data.qlib_dump import dump_bin, normalize_to_csv


def build_store(cleaned: pd.DataFrame, calendar, out_dir, cut=None) -> Path:
    """정제 데이터 -> qlib 스토어. cut 이 주어지면 cut 이후 데이터를 제거한다."""
    out_dir = Path(out_dir)
    cal = pd.DatetimeIndex(calendar)
    if cut is not None:
        cleaned = cleaned[cleaned["date"] <= pd.Timestamp(cut)]
        cal = cal[cal <= pd.Timestamp(cut)]
    normalize_to_csv(cleaned, list(cal), out_dir / "normalized")
    dump_bin(out_dir / "normalized", out_dir / "qlib", max_workers=1)
    return out_dir / "qlib"


def init_qlib(qlib_dir) -> None:
    import qlib
    from qlib.data.cache import H

    qlib.init(provider_uri=str(qlib_dir), region="us", kernels=1, expression_cache=None, dataset_cache=None)
    for k in ("c", "i", "f"):
        H[k].clear()


def compute_fields(qlib_dir, fields: Sequence[str], names: Sequence[str], end=None) -> pd.DataFrame:
    from qlib.data import D

    init_qlib(qlib_dir)
    df = D.features(D.instruments("all"), list(fields), end_time=end, freq="day")
    df.columns = list(names)
    return df.swaplevel().sort_index()  # (datetime, instrument)


def lookahead_report(
    cleaned: pd.DataFrame, calendar, fields: Sequence[str], names: Sequence[str], cut, work_dir, rtol: float = 1e-5
) -> pd.DataFrame:
    """필드별 누수 여부. 반환 컬럼: n_compared, n_diff, max_abs_diff, leaks"""
    work_dir = Path(work_dir)
    full = compute_fields(build_store(cleaned, calendar, work_dir / "full"), fields, names, end=cut)
    trunc = compute_fields(build_store(cleaned, calendar, work_dir / "trunc", cut=cut), fields, names, end=cut)
    full, trunc = full.align(trunc, join="outer")
    rows = []
    for n in names:
        a, b = full[n].to_numpy(float), trunc[n].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=rtol, atol=1e-8)
        diff = ~(both_nan | close)
        mad = float(np.nanmax(np.abs(a - b)[diff])) if diff.any() and np.isfinite(np.abs(a - b)[diff]).any() else 0.0
        rows.append({"field": n, "n_compared": int(len(a)), "n_diff": int(diff.sum()), "max_abs_diff": mad})
    rep = pd.DataFrame(rows)
    rep["leaks"] = rep["n_diff"] > 0
    return rep
