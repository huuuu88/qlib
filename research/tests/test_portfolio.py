"""Phase 4 테스트: 체결 타이밍, 비용 계산, 용량 제약, 포트폴리오 제약, 통계."""

import numpy as np
import pandas as pd
import pytest

from research.pipelines.stats import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe, sharpe
from research.portfolio.backtest import run_backtest, turnover_stats
from research.portfolio.construct import (
    OptConfig, estimate_beta, estimate_cov, long_short_quantile, optimized, ppo_long_only_weights, solve_weights,
    topk_equal,
)
from research.portfolio.costs import ZERO_COST, CostModel


def _panel(n_days=60, n_sym=4, seed=1):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2022-01-03", periods=n_days)
    syms = [f"S{i}" for i in range(n_sym)]
    opens = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_sym)), axis=0), index=cal, columns=syms)
    adv = pd.DataFrame(1e9, index=cal, columns=syms)
    vol = pd.DataFrame(0.02, index=cal, columns=syms)
    return cal, syms, opens, adv, vol


# ---------- 타이밍 ----------
def test_signal_at_t_trades_at_next_open_and_earns_open_to_open():
    cal, syms, opens, adv, vol = _panel()
    t = cal[10]
    W = pd.DataFrame([[1.0, 0, 0, 0]], index=[t], columns=syms)
    res = run_backtest(W, opens, adv, vol, ZERO_COST)
    assert res.daily.index[0] == cal[11]  # 체결은 다음 거래일
    exp = opens["S0"].shift(-1) / opens["S0"] - 1
    np.testing.assert_allclose(res.gross.to_numpy(), exp.loc[cal[11] : cal[-2]].to_numpy(), rtol=1e-12)
    # t 일 시가 -> t+1 일 시가 수익(신호일의 움직임)은 받지 못한다
    assert cal[10] not in res.daily.index


def test_future_prices_do_not_change_past_returns():
    cal, syms, opens, adv, vol = _panel()
    W = pd.DataFrame(0.25, index=cal[5:40:5], columns=syms)
    a = run_backtest(W, opens, adv, vol, CostModel(), end=cal[30])
    o2 = opens.copy()
    o2.loc[cal[35] :] *= 3
    b = run_backtest(W, o2, adv, vol, CostModel(), end=cal[30])
    pd.testing.assert_frame_equal(a.daily, b.daily)


# ---------- 비용 ----------
def test_linear_cost_arithmetic_and_gross_net_relation():
    cal, syms, opens, adv, vol = _panel()
    W = pd.DataFrame([[0.25] * 4], index=[cal[10]], columns=syms)
    cm = CostModel(commission_bps=1, half_spread_bps=2, slippage_bps=3, impact_coef=0.0, borrow_bps_per_year=0)
    res = run_backtest(W, opens, adv, vol, cm)
    d = res.daily.iloc[0]
    assert d["turnover"] == pytest.approx(1.0)
    assert d["cost_total"] == pytest.approx(6e-4)
    assert d["net"] == pytest.approx((1 + d["gross"]) * (1 - 6e-4) - 1)
    assert (res.daily["cost_total"].iloc[1:] == 0).all()  # 리밸런싱이 없으면 비용도 없다
    zero = run_backtest(W, opens, adv, vol, ZERO_COST)
    pd.testing.assert_series_equal(zero.gross, zero.net, check_names=False)
    pd.testing.assert_series_equal(zero.gross, res.gross)  # 비용은 gross 를 바꾸지 않는다


def test_impact_grows_with_aum_and_cost_scales_with_multiplier():
    cal, syms, opens, adv, vol = _panel()
    W = pd.DataFrame(0.25, index=cal[5:50:5], columns=syms)
    W.iloc[::2] = [0.4, 0.3, 0.2, 0.1]
    cm = CostModel(max_participation=np.inf)
    small = run_backtest(W, opens, adv, vol, cm, aum=1e7).daily
    big = run_backtest(W, opens, adv, vol, cm, aum=1e9).daily
    assert big["cost_impact"].sum() == pytest.approx(small["cost_impact"].sum() * 10, rel=1e-2)  # sqrt(100). NAV 경로 차이만큼 오차
    assert big["cost_spread"].sum() == pytest.approx(small["cost_spread"].sum(), rel=1e-2)
    x2 = run_backtest(W, opens, adv, vol, cm.scaled(2), aum=1e7).daily
    assert x2["cost_total"].sum() == pytest.approx(small["cost_total"].sum() * 2, rel=1e-2)


def test_short_positions_pay_borrow():
    cal, syms, opens, adv, vol = _panel()
    W = pd.DataFrame([[0.5, 0.5, -0.5, -0.5]], index=[cal[10]], columns=syms)
    cm = CostModel(0, 0, 0, 0, borrow_bps_per_year=252.0)
    d = run_backtest(W, opens, adv, vol, cm).daily
    assert d["cost_borrow"].iloc[0] == pytest.approx(1.0 * 252e-4 / 252)
    assert d["n_long"].iloc[0] == 2 and d["n_short"].iloc[0] == 2


# ---------- 용량 / 거래정지 ----------
def test_participation_cap_clips_orders():
    cal, syms, opens, adv, vol = _panel()
    adv = adv.copy()
    adv["S0"] = 1e6  # 유동성이 매우 낮은 종목
    W = pd.DataFrame([[1.0, 0, 0, 0]], index=[cal[10]], columns=syms)
    res = run_backtest(W, opens, adv, vol, CostModel(max_participation=0.05), aum=1e8)
    assert res.weights.iloc[0]["S0"] == pytest.approx(0.05 * 1e6 / 1e8)  # ADV 의 5% 만 체결
    assert res.daily["clipped"].iloc[0] == pytest.approx(1 - 0.0005)
    assert turnover_stats(res)["max_clipped_frac"] > 0.99


def test_halted_name_is_not_traded():
    cal, syms, opens, adv, vol = _panel()
    opens = opens.copy()
    opens.loc[cal[11], "S1"] = np.nan  # 체결일에 시가 없음
    W = pd.DataFrame([[0.5, 0.5, 0, 0]], index=[cal[10]], columns=syms)
    res = run_backtest(W, opens, adv, vol, ZERO_COST)
    assert res.weights.iloc[0]["S1"] == 0 and res.weights.iloc[0]["S0"] == pytest.approx(0.5)


# ---------- 포트폴리오 구성 ----------
def _scores(n_days=320, n_sym=80, seed=5):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2021-01-04", periods=n_days)
    syms = [f"A{i:02d}" for i in range(n_sym)]
    mkt = rng.normal(0, 0.01, n_days)
    beta = rng.uniform(0.5, 1.5, n_sym)
    ret = pd.DataFrame(np.outer(mkt, beta) + rng.normal(0, 0.015, (n_days, n_sym)), index=cal, columns=syms)
    scores = pd.DataFrame(rng.normal(size=(n_days, n_sym)), index=cal, columns=syms).iloc[260:]
    sectors = pd.Series([f"sec{i % 5}" for i in range(n_sym)], index=syms)
    return cal, syms, ret, scores, sectors, pd.Series(mkt, index=cal)


def test_topk_and_long_short_weights():
    _, _, _, scores, _, _ = _scores()
    w = topk_equal(scores, k=10, every=5)
    assert np.allclose(w.sum(axis=1), 1) and ((w > 0).sum(axis=1) == 10).all() and len(w) == 12
    assert (scores.loc[w.index[0]][w.iloc[0] > 0].min() >= scores.loc[w.index[0]][w.iloc[0] == 0].max())
    ls = long_short_quantile(scores, q=0.2, every=5)
    assert np.allclose(ls.sum(axis=1), 0) and np.allclose(ls.abs().sum(axis=1), 2)


def test_optimizer_respects_constraints_long_only():
    _, _, ret, scores, sectors, _ = _scores()
    cfg = OptConfig(max_weight=0.05, max_sector_weight=0.3, max_turnover=0.4)
    w = optimized(scores, ret, sectors, cfg, every=5)
    assert np.allclose(w.sum(axis=1), 1, atol=1e-5) and w.min().min() >= -1e-7 and w.max().max() <= 0.05 + 1e-5
    sec_w = w.T.groupby(sectors).sum().T
    assert sec_w.max().max() <= 0.3 + 1e-5
    turn = w.diff().abs().sum(axis=1).iloc[1:]
    assert turn.max() <= 0.4 + 1e-4
    # 신호가 좋은 종목에 더 많이 싣는다
    t = w.index[0]
    assert np.corrcoef(w.loc[t], scores.loc[t])[0, 1] > 0.3


def test_optimizer_long_short_is_dollar_sector_and_beta_neutral():
    _, _, ret, scores, sectors, bench = _scores()
    cfg = OptConfig(mode="long_short", max_weight=0.05, max_turnover=None)
    w = optimized(scores, ret, sectors, cfg, every=20, bench=bench)
    assert np.allclose(w.sum(axis=1), 0, atol=1e-5) and (w.abs().sum(axis=1) <= 2 + 1e-4).all()
    assert w.T.groupby(sectors).sum().abs().max().max() < 1e-5
    t = w.index[-1]
    beta = estimate_beta(ret.loc[:t].iloc[-252:], bench)
    assert abs(float(beta @ w.loc[t].to_numpy())) < 1e-5


def test_optimizer_does_not_use_future_returns():
    cal, _, ret, scores, sectors, _ = _scores()
    cfg = OptConfig(max_turnover=None)
    a = optimized(scores.iloc[:21], ret, sectors, cfg, every=5)
    r2 = ret.copy()
    r2.loc[scores.index[21] :] *= 5
    b = optimized(scores.iloc[:21], r2, sectors, cfg, every=5)
    pd.testing.assert_frame_equal(a, b)


def test_cvxpy_and_pyportfolioopt_agree():
    _, syms, ret, scores, sectors, _ = _scores()
    cfg = OptConfig(max_weight=0.05, max_sector_weight=0.3, trade_penalty_bps=0.0, max_turnover=None)
    t = scores.index[0]
    s = scores.loc[t]
    alpha = (((s - s.mean()) / s.std(ddof=0)).clip(-3, 3) * cfg.alpha_per_z).to_numpy()
    cov = estimate_cov(ret.loc[:t].iloc[-252:])
    mine = solve_weights(alpha, cov, np.zeros(len(s)), sectors, cfg, first=True)
    ppo = ppo_long_only_weights(alpha, cov, syms, sectors, cfg)
    assert np.abs(mine - ppo.to_numpy()).max() < 2e-3


# ---------- 통계 ----------
def test_psr_and_deflated_sharpe():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500))
    sr_pp = r.mean() / r.std(ddof=1)
    assert probabilistic_sharpe(r, sr_pp) == pytest.approx(0.5, abs=1e-9)
    assert probabilistic_sharpe(r, 0.0) > 0.5
    assert expected_max_sharpe(1, 0.05) == 0.0
    assert expected_max_sharpe(100, 0.05) > expected_max_sharpe(10, 0.05) > 0
    trials = rng.normal(0, 0.03, 20)
    d1, d20 = deflated_sharpe(r, 1, trials[:1]), deflated_sharpe(r, 20, trials)
    assert d1["deflated_sharpe"] == pytest.approx(d1["psr_vs_zero"])
    assert d20["deflated_sharpe"] < d20["psr_vs_zero"]  # 시도가 많을수록 깎인다
    assert sharpe(r) == pytest.approx(sr_pp * np.sqrt(252))
