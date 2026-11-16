"""일별 백테스터. 타이밍은 research/pipelines/TIMING.md 와 동일하다.

    t 일 종가  : 목표 비중 W[t] 결정 (신호)
    t+1 일 시가: W[t] 로 체결. 이 포지션은 open[t+1] -> open[t+2] 수익률부터 받는다.

- 비중은 NAV 대비 비율. 리밸런싱 사이에는 수익률에 따라 비중이 표류한다(drift).
- 체결일에 시가가 없는 종목(거래정지)은 거래하지 못하고 기존 포지션을 유지한다.
- 주문금액이 ADV x max_participation 을 넘으면 넘는 만큼 체결하지 않는다(용량 제약).
- 비용 전(gross)/후(net) 수익률을 한 번의 시뮬레이션에서 함께 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class BacktestResult:
    daily: pd.DataFrame  # index=체결일(d). gross, net, turnover, cost_*, n_long, n_short, clipped_frac, ...
    weights: pd.DataFrame  # 체결 직후 실제 비중 (리밸런싱일만)
    aum: float

    @property
    def gross(self) -> pd.Series:
        return self.daily["gross"]

    @property
    def net(self) -> pd.Series:
        return self.daily["net"]


def open_to_open_returns(open_px: pd.DataFrame) -> pd.DataFrame:
    """R[d] = open[d+1] / open[d] - 1  (d 일 시가에 들고 있는 포지션이 받는 수익률).
    거래정지로 시가가 비면 직전 시가를 이어 쓴다(그날 수익률 0, 재개 시 한꺼번에 반영)."""
    o = open_px.ffill()
    return o.shift(-1) / o - 1


def run_backtest(
    target_weights: pd.DataFrame,
    open_px: pd.DataFrame,
    adv_dollar: pd.DataFrame,
    daily_vol: pd.DataFrame,
    cost: CostModel,
    aum: float = 1e8,
    start=None,
    end=None,
) -> BacktestResult:
    """target_weights: index=신호일 t, columns=종목. 행이 있는 날만 리밸런싱한다(다음 거래일 시가 체결).
    adv_dollar, daily_vol: index=일자. d 일 체결에는 d-1 일 값을 쓴다."""
    cal = open_px.index
    syms = open_px.columns
    R = open_to_open_returns(open_px).to_numpy(float)
    O = open_px.to_numpy(float)
    ADV = adv_dollar.reindex(index=cal, columns=syms).shift(1).to_numpy(float)
    VOL = daily_vol.reindex(index=cal, columns=syms).shift(1).to_numpy(float)

    # 신호일 -> 체결일(다음 거래일)
    W = target_weights.reindex(columns=syms).fillna(0.0)
    sig_pos = cal.get_indexer(W.index)
    if (sig_pos < 0).any():
        raise ValueError("target_weights has dates outside the calendar")
    exec_pos = {int(p) + 1: W.iloc[i].to_numpy(float) for i, p in enumerate(sig_pos) if p + 1 < len(cal)}
    if not exec_pos:
        raise ValueError("no executable rebalance")

    d0 = min(exec_pos) if start is None else max(min(exec_pos), int(cal.searchsorted(pd.Timestamp(start))))
    d1 = len(cal) - 2 if end is None else min(len(cal) - 2, int(cal.searchsorted(pd.Timestamp(end), side="right")) - 1)

    h = np.zeros(len(syms))  # 현재 비중
    nav = 1.0
    rows, wrows = [], {}
    for d in range(d0, d1 + 1):
        rec = dict.fromkeys(
            ("turnover", "cost_commission", "cost_spread", "cost_slippage", "cost_impact", "clipped",
             "part_wavg", "part_max", "part_wanted_max"), 0.0
        )
        if d in exec_pos:
            target = exec_pos[d]
            can_trade = np.isfinite(O[d])
            trade = np.where(can_trade, target - h, 0.0)
            # 용량: |주문금액| <= max_participation x ADV
            dollars = np.abs(trade) * nav * aum
            cap = cost.max_participation * np.nan_to_num(ADV[d], nan=0.0)
            if np.isfinite(cost.max_participation):
                scale = np.where(dollars > cap, np.divide(cap, dollars, out=np.zeros_like(cap), where=dollars > 0), 1.0)
            else:
                scale = np.ones_like(dollars)
            wanted = np.abs(trade).sum()
            adv_d = np.where(np.nan_to_num(ADV[d], nan=0.0) > 0, ADV[d], np.nan)
            rec["part_wanted_max"] = float(np.nanmax(dollars / adv_d)) if np.isfinite(dollars / adv_d).any() else 0.0
            trade = trade * scale
            part = np.abs(trade) * nav * aum / adv_d  # 참여율 = 주문금액 / ADV
            if np.abs(trade).sum() > 0 and np.isfinite(part).any():
                rec["part_wavg"] = float(np.nansum(part * np.abs(trade)) / np.abs(trade)[np.isfinite(part)].sum())
                rec["part_max"] = float(np.nanmax(part))
            rec["clipped"] = float(1 - np.abs(trade).sum() / wanted) if wanted > 0 else 0.0
            c = cost.trade_cost(trade * nav * aum, ADV[d], VOL[d])
            for k, v in c.items():
                rec[f"cost_{k}"] = v / (nav * aum)
            rec["turnover"] = float(np.abs(trade).sum())  # 편도 합(매수+매도). 0.2 = NAV 의 20% 거래
            h = h + trade
            wrows[cal[d]] = h.copy()
        short_notional = float(-h[h < 0].sum())
        rec["cost_borrow"] = short_notional * cost.borrow_bps_per_year * 1e-4 / 252
        r = np.nan_to_num(R[d], nan=0.0)
        gross = float((h * r).sum())
        total_cost = sum(v for k, v in rec.items() if k.startswith("cost_"))
        net = (1 + gross) * (1 - total_cost) - 1
        rec.update(
            gross=gross, net=net, cost_total=total_cost, n_long=int((h > 1e-9).sum()), n_short=int((h < -1e-9).sum()),
            long_exposure=float(h[h > 0].sum()), short_exposure=short_notional,
        )
        rows.append(rec)
        # 비중 표류. 롱숏은 NAV 가 (1+gross) 로 변한다
        h = h * (1 + r) / (1 + gross)
        nav *= 1 + net
    daily = pd.DataFrame(rows, index=cal[d0 : d1 + 1])
    weights = pd.DataFrame.from_dict(wrows, orient="index", columns=syms)
    return BacktestResult(daily=daily, weights=weights, aum=aum)


def turnover_stats(res: BacktestResult) -> dict:
    d = res.daily
    reb = d[d["turnover"] > 0]
    years = len(d) / 252
    return {
        "rebalances": int(len(reb)),
        "avg_turnover_per_rebalance": float(reb["turnover"].mean()) if len(reb) else 0.0,
        "annual_turnover": float(d["turnover"].sum() / years),  # 편도 합 기준 (2.0 = 연간 NAV 의 2배 거래)
        "avg_clipped_frac": float(reb["clipped"].mean()) if len(reb) else 0.0,
        "max_clipped_frac": float(reb["clipped"].max()) if len(reb) else 0.0,
        "annual_cost": float(d["cost_total"].sum() / years),
        "annual_cost_breakdown": {
            k[5:]: float(d[k].sum() / years)
            for k in ("cost_commission", "cost_spread", "cost_slippage", "cost_impact", "cost_borrow")
        },
        "avg_participation": float(reb["part_wavg"].mean()) if len(reb) else 0.0,  # 거래금액 가중 평균 참여율
        "max_participation": float(reb["part_max"].max()) if len(reb) else 0.0,
        "max_participation_wanted": float(reb["part_wanted_max"].max()) if len(reb) else 0.0,  # 상한이 없었다면
        "avg_n_long": float(d["n_long"].mean()),
        "avg_n_short": float(d["n_short"].mean()),
    }
