"""qlib 포맷 변환 결과를 qlib API 로 다시 읽어 원본과 대조한다(end-to-end 검증)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def verify_qlib(qlib_dir, cleaned: pd.DataFrame, universe: str) -> dict:
    import qlib
    from qlib.data import D
    from qlib.utils import code_to_fname

    qlib.init(provider_uri=str(qlib_dir), region="us", kernels=1)
    cal = D.calendar(freq="day")
    inst = D.instruments(market=universe)
    members = D.list_instruments(inst, as_list=True)
    df = D.features(inst, ["$close", "$factor", "$volume", "$tradable", "$change"], freq="day")
    df.columns = ["close", "factor", "volume", "tradable", "change"]
    df = df.reset_index().rename(columns={"instrument": "qsym", "datetime": "date"})

    src = cleaned[["symbol", "date", "close", "volume", "adjclose", "tradable"]].copy()
    src["qsym"] = src["symbol"].map(lambda s: code_to_fname(s).upper())
    m = df.merge(src, on=["qsym", "date"], suffixes=("_q", "_src"))
    ok = m["close_q"].notna() & m["close_src"].notna() & (m["volume_src"] > 0)

    raw_rebuilt = m.loc[ok, "close_q"] / m.loc[ok, "factor"]
    raw_err = float((raw_rebuilt / m.loc[ok, "close_src"] - 1).abs().max())
    dv_q = m.loc[ok, "close_q"] * m.loc[ok, "volume_q"]
    dv_src = m.loc[ok, "close_src"] * m.loc[ok, "volume_src"]
    dv_err = float((dv_q / dv_src - 1).abs().max())
    # qlib 수익률 vs adjclose 수익률
    m = m.sort_values(["qsym", "date"])
    adj_ret = m.groupby("qsym")["adjclose"].transform(lambda s: s.ffill() / s.ffill().shift(1) - 1)
    q_ret = m.groupby("qsym")["close_q"].transform(lambda s: s.ffill() / s.ffill().shift(1) - 1)
    ret_err = float((adj_ret - q_ret)[ok].abs().max())
    trad_mismatch = int((m.loc[ok, "tradable_q"].fillna(-1) != m.loc[ok, "tradable_src"].fillna(-1)).sum())

    res = {
        "calendar_days": len(cal),
        "calendar_first": f"{cal[0]:%Y-%m-%d}",
        "calendar_last": f"{cal[-1]:%Y-%m-%d}",
        "instruments": len(members),
        "feature_rows": int(len(df)),
        "compared_rows": int(ok.sum()),
        "max_rel_err_raw_close": raw_err,
        "max_rel_err_dollar_volume": dv_err,
        "max_abs_err_return": ret_err,
        "tradable_mismatch_rows": trad_mismatch,
    }
    res["passed"] = bool(
        res["instruments"] == src.dropna(subset=["close"])["symbol"].nunique()
        and res["compared_rows"] > 0
        and raw_err < 1e-3
        and dv_err < 1e-3
        and ret_err < 1e-3
        and trad_mismatch == 0
        and np.isfinite(raw_err)
    )
    return res
