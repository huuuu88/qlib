"""팩터별 검증 (alphalens-reloaded).

    python -m research.factors.evaluate --data sp500

- 선행수익률은 라벨과 같은 타이밍으로 만든다: alphalens 에 넘기는 가격을 P[t] = open[t+1] 로 두면
  p 기간 선행수익률이 open[t+1+p]/open[t+1]-1 이 된다(t 일 종가 신호 -> t+1 시가 진입).
- 대상: 유니버스 멤버이면서 $tradable 인 종목-일.
- 각 팩터는 trials.jsonl 에 '시도'로 기록된다.
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from research.data.manifest import REPO_ROOT
from research.factors.library import FACTORS, UNAVAILABLE, factor_fields
from research.pipelines import market
from research.pipelines.trials import record_trial

PERIODS = (1, 5, 10, 21)
QUANTILES = 5
MAIN = 5  # 라벨 보유기간


def load_factor_frame(universe: str) -> pd.DataFrame:
    from qlib.data import D

    fields, names = factor_fields()
    df = D.features(D.instruments(universe), fields + ["$tradable"], freq="day")
    df.columns = names + ["tradable"]
    df = df[df["tradable"] > 0.5].drop(columns="tradable")
    return df.swaplevel().sort_index()  # (datetime, instrument)


def evaluate_one(factor: pd.Series, exec_px: pd.DataFrame) -> dict:
    import alphalens as al

    f = factor.dropna()
    f.index = f.index.set_names(["date", "asset"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fd = al.utils.get_clean_factor_and_forward_returns(
            f, exec_px, quantiles=QUANTILES, periods=PERIODS, max_loss=1.0
        )
        ret_cols = [c for c in fd.columns if c not in ("factor", "factor_quantile", "group")]
        col = dict(zip(PERIODS, ret_cols))
        ric = al.performance.factor_information_coefficient(fd)  # 일자별 Spearman
        mq, _ = al.performance.mean_return_by_quantile(fd, by_date=False, demeaned=True)
        turn_top = al.performance.quantile_turnover(fd["factor_quantile"], QUANTILES, period=MAIN)
        turn_bot = al.performance.quantile_turnover(fd["factor_quantile"], 1, period=MAIN)
        autocorr = al.performance.factor_rank_autocorrelation(fd, period=MAIN)
    ic = fd.groupby(level="date").apply(lambda g: g["factor"].corr(g[col[MAIN]]))  # Pearson
    r = ric[col[MAIN]]
    n_eff = r.count() / MAIN  # 5일 라벨이 겹치므로 유효 표본을 1/5 로 본다
    return {
        "n_obs": int(len(fd)),
        "n_days": int(r.count()),
        "ic_mean": float(ic.mean()),
        "rank_ic_mean": float(r.mean()),
        "rank_ic_ir": float(r.mean() / r.std(ddof=1)),
        "rank_ic_tstat_overlap_adj": float(r.mean() / r.std(ddof=1) * np.sqrt(n_eff)),
        "rank_ic_decay": {f"{p}d": float(ric[col[p]].mean()) for p in PERIODS},
        "quantile_mean_return_5d": {f"Q{int(q)}": float(v) for q, v in mq[col[MAIN]].items()},
        "spread_q5_q1_5d": float(mq[col[MAIN]].iloc[-1] - mq[col[MAIN]].iloc[0]),
        "turnover_top_5d": float(turn_top.mean()),
        "turnover_bottom_5d": float(turn_bot.mean()),
        "rank_autocorr_5d": float(autocorr.mean()),
    }


def run(data: str, universe: str = None, start: str = "2016-01-04") -> dict:
    universe = universe or data
    meta = market.init(data)
    panels = market.load_panels(universe)
    exec_px = panels.open.ffill().shift(-1)  # P[t] = open[t+1]
    frame = load_factor_frame(universe).loc[start:]
    out = {"data": data, "universe": universe, "period_start": start, "flags": meta.get("flags"),
           "data_hash": meta.get("data_hash"), "periods": PERIODS, "quantiles": QUANTILES, "factors": {}}
    for name in FACTORS:
        res = evaluate_one(frame[name], exec_px)
        out["factors"][name] = {"family": FACTORS[name]["family"], "expr": FACTORS[name]["expr"], **res}
        record_trial("factor", f"{data}/{name}", {"expr": FACTORS[name]["expr"], "start": start},
                     {k: res[k] for k in ("rank_ic_mean", "rank_ic_tstat_overlap_adj", "spread_q5_q1_5d")})
        print(name, round(res["rank_ic_mean"], 4), round(res["rank_ic_tstat_overlap_adj"], 2))
    out["period_end"] = f"{frame.index[-1][0]:%Y-%m-%d}"
    rep = REPO_ROOT / "research" / "reports" / "phase3" / data
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "factor_validation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (rep / "factor_validation.md").write_text(to_markdown(out), encoding="utf-8")
    return out


def to_markdown(out: dict) -> str:
    L = [f"# 팩터별 검증 — {out['universe']} ({out['period_start']} ~ {out['period_end']})", ""]
    if (out.get("flags") or {}).get("survivorship_bias", True):
        L += ["> **⚠ 생존편향 존재** — 아래 수치는 상향 편향되어 있을 수 있다.", ""]
    L += [
        "alphalens-reloaded 로 계산. 선행수익률은 t+1 시가 진입 기준, 분위수는 일자별 5분위, 수익률은 횡단면 평균 차감.",
        "t-stat 은 5일 라벨의 겹침을 감안해 유효 표본을 1/5 로 줄인 값이다.",
        "",
        "| 팩터 | 계열 | IC | Rank IC | ICIR | t-stat | Q5−Q1 (5일) | 상위 분위 턴오버 | 순위 자기상관 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n, r in out["factors"].items():
        L.append(
            f"| {n} | {r['family']} | {r['ic_mean']:.4f} | {r['rank_ic_mean']:.4f} | {r['rank_ic_ir']:.3f} | "
            f"{r['rank_ic_tstat_overlap_adj']:.2f} | {r['spread_q5_q1_5d'] * 1e4:.1f} bp | {r['turnover_top_5d']:.3f} | "
            f"{r['rank_autocorr_5d']:.3f} |"
        )
    L += ["", "## Rank IC decay (보유기간별 평균 Rank IC)", "", "| 팩터 | " + " | ".join(f"{p}일" for p in out["periods"]) + " |",
          "|---|" + "---|" * len(out["periods"])]
    for n, r in out["factors"].items():
        L.append(f"| {n} | " + " | ".join(f"{v:.4f}" for v in r["rank_ic_decay"].values()) + " |")
    L += ["", "## 분위수별 평균 5일 수익률 (횡단면 평균 차감, bp)", "", "| 팩터 | Q1 | Q2 | Q3 | Q4 | Q5 |", "|---|---|---|---|---|---|"]
    for n, r in out["factors"].items():
        L.append(f"| {n} | " + " | ".join(f"{v * 1e4:.1f}" for v in r["quantile_mean_return_5d"].values()) + " |")
    L += ["", "## 구현하지 못한 팩터", ""] + [f"- **{k}**: {v}" for k, v in UNAVAILABLE.items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sp500")
    run(ap.parse_args().data)
