"""Phase 4: 신호 -> 포트폴리오 -> 비용 반영 백테스트.

    python -m research.portfolio.run --data sp500 --signal alpha158_lgb

전략(사전에 고정, 모두 trials.jsonl 에 기록)
- topk50_ew          : 상위 50 동일가중, 5거래일마다 리밸런싱 (롱온리)            ← 기준 전략
- ls_quintile        : 상위/하위 20% 동일가중 (달러 중립 롱숏)
- opt_long_only      : 평균-분산 최적화. 종목 3%, 섹터 25%, 리밸런싱당 턴오버 50% 한도
- opt_ls_neutral     : 평균-분산 최적화 롱숏. 달러·섹터·베타 중립, 종목 ±3%, 턴오버 100% 한도
비교 대상
- universe_ew        : 유니버스 전체 동일가중(같은 리밸런싱 주기)
- SPY                : 총수익 ETF, 시가-시가 수익률

산출물: research/reports/phase4/<data>/<signal>/{summary.json, summary.md}, 일별 수익률은 store(gitignore)
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from research.data.manifest import REPO_ROOT, _git
from research.pipelines import market
from research.pipelines.stats import perf_summary, sharpe
from research.pipelines.trials import record_trial

from .backtest import open_to_open_returns, run_backtest, turnover_stats
from .construct import (
    OptConfig, estimate_cov, long_short_quantile, optimized, ppo_long_only_weights, score_matrix, solve_weights,
    topk_equal, _alpha,
)
from .costs import ZERO_COST, CostModel

EVERY = 5
BASE_AUM = 1e8
COST = CostModel()
COST_MULTS = [0.0, 0.5, 1.0, 2.0, 4.0]
AUMS = [1e7, 1e8, 1e9, 1e10]
PRIMARY = "topk50_ew"
OPT_LO = OptConfig(mode="long_only", max_weight=0.03, max_sector_weight=0.25, max_turnover=0.5)
OPT_LS = OptConfig(mode="long_short", max_weight=0.03, max_turnover=1.0, sector_neutral=True, beta_neutral=True)
BENCH = "SPY"


def build_weights(scores, panels, sectors, bench_ret, log) -> dict:
    eligible = scores.notna()
    ew = eligible.div(eligible.sum(axis=1), axis=0).iloc[::EVERY].astype(float)
    return {
        "topk50_ew": topk_equal(scores, 50, EVERY),
        "ls_quintile": long_short_quantile(scores, 0.2, EVERY),
        "opt_long_only": optimized(scores, panels.ret, sectors, OPT_LO, EVERY, log=log),
        "opt_ls_neutral": optimized(scores, panels.ret, sectors, OPT_LS, EVERY, bench=bench_ret, log=log),
        "universe_ew": ew,
    }


def ppo_crosscheck(scores, panels, sectors, n_dates: int = 8) -> dict:
    """cvxpy 구현과 PyPortfolioOpt 를 같은 문제(거래비용·턴오버 항 제외)로 비교."""
    cfg = OptConfig(mode="long_only", max_weight=OPT_LO.max_weight, max_sector_weight=OPT_LO.max_sector_weight,
                    trade_penalty_bps=0.0, max_turnover=None)
    dates = scores.index[:: max(len(scores) // n_dates, 1)][:n_dates]
    diffs = []
    for t in dates:
        s = scores.loc[t].dropna()
        win = panels.ret.loc[:t].iloc[-cfg.cov_lookback :]
        s = s[win[s.index].notna().sum() >= cfg.min_history]
        cov = estimate_cov(win[s.index])
        sec = sectors.reindex(s.index).fillna("Unknown")
        a = _alpha(s, cfg)
        mine = solve_weights(a, cov, np.zeros(len(s)), sec, cfg, first=True)
        ppo = ppo_long_only_weights(a, cov, list(s.index), sec, cfg).to_numpy()
        diffs.append(float(np.abs(mine - ppo).max()))
    return {"n_dates": len(diffs), "max_abs_weight_diff": max(diffs), "mean_abs_weight_diff": float(np.mean(diffs))}


def ls_sharpe_zero_cost(scores: pd.DataFrame, panels) -> float:
    """시도 간 비교용 표준 지표: 상/하위 20% 롱숏, 비용 0, 1기간(일) Sharpe."""
    w = long_short_quantile(scores, 0.2, EVERY)
    r = run_backtest(w, panels.open, panels.adv, panels.vol, ZERO_COST, BASE_AUM).gross
    return float(r.mean() / r.std(ddof=1))


def run(data: str, signal: str) -> dict:
    t0 = time.time()
    meta = market.init(data)
    panels = market.load_panels(data)
    exp_dir = market.store_dir(data) / "experiments" / signal
    pred = pd.read_pickle(exp_dir / "pred.pkl")
    scores = score_matrix(pred).reindex(columns=panels.open.columns)
    sectors = market.load_sectors(data, panels.open.columns)
    bench_ret = panels.ret[BENCH] if BENCH in panels.ret else None
    log = []
    weights = build_weights(scores, panels, sectors, bench_ret, log)

    start, end = scores.index[0], scores.index[-1]
    out_dir = exp_dir / "portfolio"
    out_dir.mkdir(exist_ok=True)
    strategies, returns = {}, {}
    for name, w in weights.items():
        res = run_backtest(w, panels.open, panels.adv, panels.vol, COST, BASE_AUM)
        returns[name] = res.daily[["gross", "net"]]
        res.daily.to_pickle(out_dir / f"{name}.pkl")
        res.weights.to_pickle(out_dir / f"{name}_weights.pkl")
        strategies[name] = {
            "gross": perf_summary(res.gross),
            "net": perf_summary(res.net),
            "turnover": turnover_stats(res),
        }
        # 비용 민감도: 비용 배수 x AUM
        grid = {}
        for aum in AUMS:
            for mult in COST_MULTS:
                r = run_backtest(w, panels.open, panels.adv, panels.vol, COST.scaled(mult), aum)
                ts = turnover_stats(r)
                grid[f"aum={aum:.0e}|cost_x{mult}"] = {
                    "net_sharpe": sharpe(r.net), "net_cagr": perf_summary(r.net)["cagr"],
                    "annual_cost": ts["annual_cost"], "avg_clipped": ts["avg_clipped_frac"],
                    "avg_participation": ts["avg_participation"], "max_participation_wanted": ts["max_participation_wanted"],
                }
        strategies[name]["sensitivity"] = grid
        if name != "universe_ew":
            record_trial("portfolio", f"{data}/{signal}/{name}", {"strategy": name, "signal": signal, "every": EVERY,
                         "cost": COST.__dict__, "aum": BASE_AUM}, {"net_sharpe": strategies[name]["net"]["sharpe"]})

    # 벤치마크: SPY 시가-시가
    idx = returns[PRIMARY].index
    if BENCH in panels.open:
        spy = open_to_open_returns(panels.open[[BENCH]])[BENCH].reindex(idx)
        spy.to_frame("ret").to_pickle(out_dir / "benchmark_spy.pkl")
        strategies[BENCH] = {"gross": perf_summary(spy), "net": perf_summary(spy), "turnover": None}

    result = {
        "data": data, "signal": signal, "flags": meta.get("flags"), "data_hash": meta.get("data_hash"),
        "git_commit": _git("rev-parse", "HEAD"), "period": [f"{idx[0]:%Y-%m-%d}", f"{idx[-1]:%Y-%m-%d}"],
        "rebalance_every_days": EVERY, "base_aum": BASE_AUM, "cost_model": COST.__dict__,
        "opt_long_only": OPT_LO.__dict__, "opt_ls_neutral": OPT_LS.__dict__,
        "sector_data": "current GICS (not point-in-time)",
        "optimizer_events": log, "ppo_crosscheck": ppo_crosscheck(scores, panels, sectors),
        "ls_quintile_zero_cost_sharpe_per_day": ls_sharpe_zero_cost(scores, panels),
        "strategies": strategies, "primary": PRIMARY, "elapsed_sec": round(time.time() - t0, 1),
    }
    rep = REPO_ROOT / "research" / "reports" / "phase4" / data / signal
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (rep / "summary.md").write_text(to_markdown(result), encoding="utf-8")
    print((rep / "summary.md").read_text(encoding="utf-8"))
    return result


def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.2f}%"


def to_markdown(r: dict) -> str:
    S = r["strategies"]
    L = [f"# Phase 4 — 포트폴리오·비용 ({r['data']} / 신호 {r['signal']})", ""]
    if (r.get("flags") or {}).get("survivorship_bias", True):
        L += ["> **⚠ 생존편향 존재** — 아래 성과는 상향 편향되어 있다.", ""]
    c = r["cost_model"]
    L += [
        f"- 기간: {r['period'][0]} ~ {r['period'][1]} (walk-forward test 구간만), 리밸런싱 {r['rebalance_every_days']}거래일마다",
        f"- 체결: 신호 다음 거래일 시가. 기준 운용자산 ${r['base_aum']:,.0f}",
        f"- 비용 가정(편도): 수수료 {c['commission_bps']}bp + 반스프레드 {c['half_spread_bps']}bp + 슬리피지 {c['slippage_bps']}bp"
        f" + 충격 {c['impact_coef']}×σ×√(주문/ADV), 대차 {c['borrow_bps_per_year']}bp/년, 참여율 상한 {c['max_participation']:.0%}",
        "",
        "## 비용 전/후 성과",
        "",
        "| 전략 | CAGR 전 | CAGR 후 | 변동성 | Sharpe 전 | Sharpe 후 | Sortino 후 | MDD 후 | 연 턴오버 | 연 비용 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, s in S.items():
        g, nn, t = s["gross"], s["net"], s["turnover"]
        L.append(
            f"| {n} | {_pct(g['cagr'])} | {_pct(nn['cagr'])} | {_pct(nn['ann_vol'])} | {g['sharpe']:.2f} | {nn['sharpe']:.2f} | "
            f"{nn['sortino']:.2f} | {_pct(nn['max_drawdown'])} | "
            + (f"{t['annual_turnover']:.1f}x | {_pct(t['annual_cost'])} |" if t else "– | – |")
        )
    L += ["", "턴오버는 매수+매도 합(편도 합) 기준이다. 1.0x = 1년에 NAV 만큼 거래.", "", "## 턴오버·용량 지표 (기준 운용자산)", "",
          "| 전략 | 리밸런싱 수 | 회당 턴오버 | 평균 보유 종목(롱/숏) | 평균 참여율 | 최대 참여율 | 미체결 비율 |", "|---|---|---|---|---|---|---|"]
    for n, s in S.items():
        t = s["turnover"]
        if t:
            L.append(f"| {n} | {t['rebalances']} | {_pct(t['avg_turnover_per_rebalance'])} | {t['avg_n_long']:.0f} / {t['avg_n_short']:.0f} | "
                     f"{_pct(t['avg_participation'])} | {_pct(t['max_participation'])} | {_pct(t['avg_clipped_frac'])} |")
    L += ["", "## 연 비용 구성 (기준 운용자산)", "", "| 전략 | 수수료 | 스프레드 | 슬리피지 | 시장충격 | 대차 |", "|---|---|---|---|---|---|"]
    for n, s in S.items():
        if s["turnover"]:
            b = s["turnover"]["annual_cost_breakdown"]
            L.append(f"| {n} | " + " | ".join(_pct(b[k]) for k in ("commission", "spread", "slippage", "impact", "borrow")) + " |")
    for n in [k for k in S if S[k].get("sensitivity") and k != "universe_ew"]:
        g = S[n]["sensitivity"]
        L += ["", f"## 비용 민감도 — {n} (비용 후 Sharpe / CAGR)", "", "| 운용자산 | " + " | ".join(f"비용 ×{m}" for m in COST_MULTS) + " | 평균 참여율 | 상한 없을 때 최대 참여율 | 미체결 비율 |",
              "|---|" + "---|" * (len(COST_MULTS) + 3)]
        for aum in AUMS:
            cells = [f"{g[f'aum={aum:.0e}|cost_x{m}']['net_sharpe']:.2f} / {_pct(g[f'aum={aum:.0e}|cost_x{m}']['net_cagr'])}" for m in COST_MULTS]
            b = g[f"aum={aum:.0e}|cost_x1.0"]
            L.append(f"| ${aum:,.0f} | " + " | ".join(cells) + f" | {_pct(b['avg_participation'])} | {_pct(b['max_participation_wanted'])} | {_pct(b['avg_clipped'])} |")
    x = r["ppo_crosscheck"]
    L += ["", "## 최적화 구현 교차검증", "",
          f"같은 롱온리 문제를 cvxpy 직접 구현과 PyPortfolioOpt 로 각각 풀어 {x['n_dates']}개 리밸런싱일에서 비교: "
          f"종목 비중 최대 차이 {x['max_abs_weight_diff']:.2e}, 평균 {x['mean_abs_weight_diff']:.2e}.",
          f"턴오버 한도를 지킬 수 없어 한도를 푼 리밸런싱: {len(r['optimizer_events'])}회.",
          "", "섹터 분류는 현재 GICS 기준이며 시점별(PIT) 값이 아니다."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sp500")
    ap.add_argument("--signal", default="alpha158_lgb")
    a = ap.parse_args()
    run(a.data, a.signal)
