"""신호 -> 목표 비중.

모든 함수는 신호일 t 의 정보만 쓴다(공분산·베타는 t 일까지의 수익률로 추정).
반환: 목표 비중 DataFrame (index=리밸런싱 신호일, columns=종목).

- topk_equal            : 상위 K 동일가중 (롱온리)
- long_short_quantile   : 상위/하위 분위 동일가중 (달러 중립)
- optimized             : 평균-분산 최적화 (cvxpy). 종목/섹터 한도, 턴오버 한도, 베타·섹터 중립 옵션
- ppo_long_only_weights : 같은 롱온리 문제를 PyPortfolioOpt 로 푼 교차검증용 구현
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, every: int = 5, offset: int = 0) -> pd.DatetimeIndex:
    return index[offset::every]


def score_matrix(pred: pd.Series) -> pd.DataFrame:
    """(datetime, instrument) 예측 -> (일자 x 종목). NaN = 그날 투자 불가."""
    return pred.unstack(level="instrument").sort_index()


def topk_equal(scores: pd.DataFrame, k: int = 50, every: int = 5) -> pd.DataFrame:
    rows = {}
    for t in rebalance_dates(scores.index, every):
        s = scores.loc[t].dropna()
        if len(s) < k:
            continue
        top = s.nlargest(k).index
        rows[t] = pd.Series(1.0 / k, index=top)
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=scores.columns).fillna(0.0)


def long_short_quantile(scores: pd.DataFrame, q: float = 0.2, every: int = 5) -> pd.DataFrame:
    rows = {}
    for t in rebalance_dates(scores.index, every):
        s = scores.loc[t].dropna()
        n = int(len(s) * q)
        if n < 5:
            continue
        w = pd.Series(0.0, index=s.index)
        w[s.nlargest(n).index] = 1.0 / n
        w[s.nsmallest(n).index] = -1.0 / n
        rows[t] = w
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=scores.columns).fillna(0.0)


@dataclass(frozen=True)
class OptConfig:
    mode: str = "long_only"  # long_only | long_short
    risk_aversion: float = 10.0  # 효용 = a'w - (g/2) w'Sw - k|w - w_prev|
    alpha_per_z: float = 0.002  # 신호 1 표준편차당 기대 5일 수익률 가정(20bp)
    horizon: int = 5
    max_weight: float = 0.03  # 종목 한도
    max_sector_weight: float = 0.25  # 섹터 한도 (롱온리)
    max_turnover: Optional[float] = 0.5  # 리밸런싱당 편도 합 한도 (첫 리밸런싱 제외). None = 무제한
    trade_penalty_bps: float = 3.0  # 목적함수의 거래비용 항(선형 비용 가정)
    sector_neutral: bool = True  # long_short 전용
    beta_neutral: bool = True  # long_short 전용
    gross_leverage: float = 2.0  # long_short: |w| 합 한도
    cov_lookback: int = 252
    min_history: int = 126


def estimate_cov(ret_window: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf 축소 공분산(일 단위). 결측은 0 으로 채운다."""
    from sklearn.covariance import LedoitWolf

    x = ret_window.fillna(0.0).to_numpy(float)
    return LedoitWolf().fit(x).covariance_


def estimate_beta(ret_window: pd.DataFrame, bench: pd.Series) -> np.ndarray:
    b = bench.reindex(ret_window.index).fillna(0.0).to_numpy(float)
    x = ret_window.fillna(0.0).to_numpy(float)
    bc = b - b.mean()
    return (x - x.mean(axis=0)).T @ bc / max(float(bc @ bc), 1e-12)


def solve_weights(alpha, cov, w_prev, sectors, cfg: OptConfig, beta=None, first: bool = False) -> np.ndarray:
    import cvxpy as cp

    n = len(alpha)
    w = cp.Variable(n)
    L = np.linalg.cholesky(cov * cfg.horizon + 1e-10 * np.eye(n))
    util = alpha @ w - cfg.risk_aversion / 2 * cp.sum_squares(L.T @ w) - cfg.trade_penalty_bps * 1e-4 * cp.norm1(w - w_prev)
    sec_codes = pd.Categorical(sectors)
    S = np.zeros((len(sec_codes.categories), n))
    S[sec_codes.codes, np.arange(n)] = 1.0
    if cfg.mode == "long_only":
        cons = [cp.sum(w) == 1, w >= 0, w <= cfg.max_weight, S @ w <= cfg.max_sector_weight]
    else:
        cons = [cp.sum(w) == 0, cp.norm1(w) <= cfg.gross_leverage, w <= cfg.max_weight, w >= -cfg.max_weight]
        if cfg.sector_neutral:
            cons.append(S @ w == 0)
        if cfg.beta_neutral and beta is not None:
            cons.append(beta @ w == 0)
    if cfg.max_turnover is not None and not first:
        cons.append(cp.norm1(w - w_prev) <= cfg.max_turnover)
    prob = cp.Problem(cp.Maximize(util), cons)
    prob.solve(solver=cp.CLARABEL)
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"optimizer failed: {prob.status}")
    out = np.where(np.abs(w.value) < 1e-6, 0.0, w.value)
    return out


def _alpha(s: pd.Series, cfg: OptConfig) -> np.ndarray:
    z = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
    return (z.clip(-3, 3) * cfg.alpha_per_z).to_numpy(float)


def optimized(
    scores: pd.DataFrame, ret: pd.DataFrame, sectors: pd.Series, cfg: OptConfig, every: int = 5,
    bench: pd.Series = None, log: list = None,
) -> pd.DataFrame:
    """ret: 종가-종가 일수익률 패널(전체 캘린더). t 일까지의 행만 사용한다."""
    rows = {}
    w_prev_full = pd.Series(0.0, index=scores.columns)
    first = True
    for t in rebalance_dates(scores.index, every):
        s = scores.loc[t].dropna()
        win = ret.loc[:t].iloc[-cfg.cov_lookback :]
        enough = win[s.index].notna().sum() >= cfg.min_history
        s = s[enough[enough].index]
        if len(s) < 50:
            continue
        cov = estimate_cov(win[s.index])
        beta = estimate_beta(win[s.index], bench) if (bench is not None and cfg.mode == "long_short") else None
        # 후보에서 빠진 기존 보유 종목은 전량 청산 대상(w_prev 에서 제외되어 자동으로 0 이 된다)
        w_prev = w_prev_full.reindex(s.index).fillna(0.0).to_numpy(float)
        try:
            w = solve_weights(_alpha(s, cfg), cov, w_prev, sectors.reindex(s.index).fillna("Unknown"), cfg, beta, first)
        except RuntimeError:
            # 턴오버 한도 때문에 실행 불가능하면 한도를 풀고 다시 푼다(기록)
            w = solve_weights(_alpha(s, cfg), cov, w_prev, sectors.reindex(s.index).fillna("Unknown"), cfg, beta, True)
            if log is not None:
                log.append({"date": f"{t:%Y-%m-%d}", "event": "turnover_limit_relaxed"})
        w_prev_full = pd.Series(0.0, index=scores.columns)
        w_prev_full[s.index] = w
        rows[t] = w_prev_full.copy()
        first = False
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=scores.columns).fillna(0.0)


def ppo_long_only_weights(alpha: np.ndarray, cov: np.ndarray, names, sectors: pd.Series, cfg: OptConfig) -> pd.Series:
    """PyPortfolioOpt 로 같은 롱온리 문제(거래비용·턴오버 항 제외)를 푼다 — 구현 교차검증용."""
    from pypfopt import EfficientFrontier

    mu = pd.Series(alpha, index=names)
    S = pd.DataFrame(cov * cfg.horizon, index=names, columns=names)
    ef = EfficientFrontier(mu, S, weight_bounds=(0, cfg.max_weight), solver="CLARABEL")
    sec = sectors.reindex(names).fillna("Unknown")
    ef.add_sector_constraints(sec.to_dict(), {}, {s: cfg.max_sector_weight for s in sec.unique()})
    ef.max_quadratic_utility(risk_aversion=cfg.risk_aversion)
    return pd.Series(ef.weights, index=names)
