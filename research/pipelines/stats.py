"""성과 통계와 과적합 보정(Probabilistic / Deflated Sharpe Ratio).

참고: Bailey & López de Prado (2012, 2014).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

ANN = 252
EULER = 0.5772156649015329


def sharpe(r: pd.Series, ann: int = ANN) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ann)) if len(r) > 2 and sd > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    nav = (1 + pd.Series(r).fillna(0)).cumprod()
    return float((nav / nav.cummax() - 1).min())


def perf_summary(r: pd.Series, ann: int = ANN) -> dict:
    r = pd.Series(r).dropna()
    if len(r) < 3:
        return {}
    nav = (1 + r).cumprod()
    years = len(r) / ann
    down = r[r < 0].std(ddof=1)
    return {
        "days": int(len(r)),
        "total_return": float(nav.iloc[-1] - 1),
        "cagr": float(nav.iloc[-1] ** (1 / years) - 1),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(ann)),
        "sharpe": sharpe(r, ann),
        "sortino": float(r.mean() / down * np.sqrt(ann)) if down and down > 0 else float("nan"),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
        "skew": float(st.skew(r)),
        "kurtosis": float(st.kurtosis(r, fisher=False)),
    }


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR: 관측 Sharpe 가 sr_benchmark(1기간 단위, 비연율)보다 클 확률."""
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = st.skew(r), st.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(st.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """N 번 시도했을 때 진짜 Sharpe 가 0 이어도 우연히 기대되는 최대 Sharpe(1기간 단위)."""
    if n_trials <= 1 or not np.isfinite(sr_std) or sr_std <= 0:
        return 0.0
    return float(
        sr_std * ((1 - EULER) * st.norm.ppf(1 - 1 / n_trials) + EULER * st.norm.ppf(1 - 1 / (n_trials * np.e)))
    )


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sharpes_per_period) -> dict:
    """DSR: 시도 횟수를 반영한 PSR. trial_sharpes_per_period 는 모든 시도의 1기간 Sharpe."""
    sr_std = float(np.nanstd(np.asarray(trial_sharpes_per_period, float), ddof=1)) if n_trials > 1 else 0.0
    sr0 = expected_max_sharpe(n_trials, sr_std)
    return {
        "n_trials": int(n_trials),
        "trial_sharpe_std_ann": sr_std * np.sqrt(ANN),
        "expected_max_sharpe_ann": sr0 * np.sqrt(ANN),
        "psr_vs_zero": probabilistic_sharpe(r, 0.0),
        "deflated_sharpe": probabilistic_sharpe(r, sr0),
    }


def ic_series(pred: pd.Series, label: pd.Series, method: str = "pearson") -> pd.Series:
    """일자별 횡단면 상관. 인덱스 (datetime, instrument)."""
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    return df.groupby(level="datetime").apply(lambda g: g["p"].corr(g["y"], method=method) if len(g) > 5 else np.nan)


def ic_summary(pred: pd.Series, label: pd.Series, horizon: int = 1) -> dict:
    ic, ric = ic_series(pred, label), ic_series(pred, label, "spearman")
    # 라벨이 horizon 일 겹치므로 t-stat 은 유효 표본 수(n/horizon)로 보수적으로 계산한다
    n_eff = max(ric.count() / max(horizon, 1), 1)
    return {
        "ic_mean": float(ic.mean()),
        "ic_ir": float(ic.mean() / ic.std(ddof=1)),
        "rank_ic_mean": float(ric.mean()),
        "rank_ic_ir": float(ric.mean() / ric.std(ddof=1)),
        "rank_ic_tstat_overlap_adj": float(ric.mean() / ric.std(ddof=1) * np.sqrt(n_eff)),
        "rank_ic_positive_rate": float((ric > 0).mean()),
        "n_days": int(ric.count()),
    }
