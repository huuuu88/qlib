"""Phase 5: 최종 리포트 자동 생성.

    python -m research.pipelines.report --data sp500 --signals alpha158_lgb [chosen_signal]

입력: Phase 1 품질 리포트, Phase 3 실험 결과, Phase 4 백테스트 결과(모두 실제 실행 산출물).
출력: research/reports/final/<data>/{REPORT.md, report.json, tearsheet_*.html}
모든 수치는 산출물에서 읽거나 여기서 계산한다. 손으로 적는 숫자는 없다.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile

import numpy as np
import pandas as pd

from research.data.manifest import REPO_ROOT, _git, hash_dir
from research.factors.library import FACTORS
from research.portfolio.construct import score_matrix
from research.portfolio.run import BENCH, PRIMARY, ls_sharpe_zero_cost

from . import market
from .stats import ANN, deflated_sharpe, perf_summary, sharpe
from .trials import load_trials, trial_counts
from .walk_forward import BASELINE, EXPERIMENTS

FF_URLS = {
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}
LONG_ONLY = {"topk50_ew", "opt_long_only", "universe_ew"}


# ---------- Fama-French ----------
def _parse_ff(raw: bytes) -> pd.DataFrame:
    txt = zipfile.ZipFile(io.BytesIO(raw)).read(zipfile.ZipFile(io.BytesIO(raw)).namelist()[0]).decode("latin-1")
    lines = txt.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",") )
    rows = []
    for l in lines[start + 1 :]:
        p = [x.strip() for x in l.split(",")]
        if len(p) < 2 or not (p[0].isdigit() and len(p[0]) == 8):
            if rows:
                break
            continue
        rows.append(p)
    cols = [c.strip() for c in lines[start].split(",")[1:]]
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype(float) / 100.0


def load_ff(data: str, refresh: bool = False) -> pd.DataFrame:
    """Ken French 데이터 라이브러리의 일별 5팩터 + 모멘텀. 스냅샷을 store 에 저장한다."""
    d = market.store_dir(data) / "ff"
    fp = d / "ff_daily.csv"
    if fp.exists() and not refresh:
        return pd.read_csv(fp, index_col=0, parse_dates=True)
    import requests

    d.mkdir(parents=True, exist_ok=True)
    parts = [_parse_ff(requests.get(u, timeout=120, headers={"User-Agent": "Mozilla/5.0"}).content) for u in FF_URLS.values()]
    ff = parts[0].join(parts[1], how="inner")
    ff.columns = [c.replace("-", "_").strip() for c in ff.columns]
    ff = ff.rename(columns={"Mom": "MOM", "Mkt_RF": "MKT_RF"})
    ff.to_csv(fp)
    return ff


def ff_regression(ret: pd.Series, ff: pd.DataFrame, excess: bool) -> dict:
    """수익률은 시가-시가(d 시가 -> d+1 시가)라서 FF 의 d 일과 d+1 일 팩터에 걸친다. 두 날을 모두 넣고 베타를 합산한다."""
    import statsmodels.api as sm

    fac = [c for c in ff.columns if c != "RF"]
    X = pd.concat([ff[fac], ff[fac].shift(-1).add_suffix("_next")], axis=1)
    y = ret - (ff["RF"].reindex(ret.index) if excess else 0.0)
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < 250:
        return {"error": f"only {len(df)} overlapping days"}
    m = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    betas = {f: float(m.params[f] + m.params[f + "_next"]) for f in fac}
    return {
        "period": [f"{df.index[0]:%Y-%m-%d}", f"{df.index[-1]:%Y-%m-%d}"], "n_days": int(len(df)), "excess_of_rf": excess,
        "alpha_ann": float(m.params["const"] * ANN), "alpha_tstat": float(m.tvalues["const"]),
        "betas": betas, "r2": float(m.rsquared),
    }


# ---------- 구간별 ----------
def regime_table(ret: pd.Series, spy: pd.Series, rf: pd.Series) -> list:
    df = pd.DataFrame({"r": ret, "spy": spy.reindex(ret.index)}).dropna()
    month = df.index.to_period("M")
    spy_m = (1 + df["spy"]).groupby(month).prod() - 1
    up = pd.Series(month.map(spy_m > 0).to_numpy(), index=df.index)
    rv = df["spy"].rolling(21).std().shift(1)  # 전일까지의 21일 실현변동성
    high_vol = rv > rv.expanding(min_periods=63).median()  # 그 시점까지의 중앙값과 비교(사후 분류 아님)
    rf_ann = (rf.reindex(df.index).ffill() * ANN)
    regimes = {
        "상승 월 (SPY 월수익률 > 0)": up, "하락 월 (SPY 월수익률 ≤ 0)": ~up,
        "고변동 (SPY 21일 변동성 > 누적 중앙값)": high_vol.fillna(False), "저변동": (~high_vol).where(rv.notna(), False),
        "고금리 (무위험 연 3% 이상)": rf_ann >= 0.03, "저금리 (무위험 연 3% 미만)": rf_ann < 0.03,
    }
    rows = []
    for name, mask in regimes.items():
        mask = mask.reindex(df.index).fillna(False).astype(bool)
        s, b = df.loc[mask, "r"], df.loc[mask, "spy"]
        if len(s) < 20:
            rows.append({"regime": name, "days": int(len(s))})
            continue
        rows.append({"regime": name, "days": int(len(s)), "ann_return": float(s.mean() * ANN), "sharpe": sharpe(s),
                     "spy_ann_return": float(b.mean() * ANN), "active_ann": float((s - b).mean() * ANN)})
    return rows


def vs_benchmark(ret: pd.Series, spy: pd.Series) -> dict:
    df = pd.DataFrame({"r": ret, "b": spy}).dropna()
    act = df["r"] - df["b"]
    beta = float(np.cov(df["r"], df["b"])[0, 1] / df["b"].var())
    return {"active_ann": float(act.mean() * ANN), "tracking_error": float(act.std(ddof=1) * np.sqrt(ANN)),
            "information_ratio": sharpe(act), "beta": beta, "correlation": float(df["r"].corr(df["b"]))}


def monthly_table(ret: pd.Series) -> pd.DataFrame:
    m = (1 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1
    t = m.unstack()
    t["연간"] = (1 + ret).groupby(ret.index.year).prod() - 1
    return t


# ---------- 시도별 Sharpe (DSR 용) ----------
def trial_sharpes(data: str, panels, test_index) -> dict:
    """모든 신호 시도(모델 + 개별 팩터)를 같은 방식(상/하위 20% 롱숏, 비용 0)으로 평가한 1일 Sharpe."""
    from research.factors.evaluate import load_factor_frame

    out = {}
    for name in EXPERIMENTS:
        fp = market.store_dir(data) / "experiments" / name / "pred.pkl"
        if fp.exists():
            sc = score_matrix(pd.read_pickle(fp)).reindex(columns=panels.open.columns)
            out[f"model:{name}"] = ls_sharpe_zero_cost(sc, panels)
    frame = load_factor_frame(data)
    frame = frame[frame.index.get_level_values(0).isin(test_index)]
    for f in FACTORS:
        sc = frame[f].unstack(level="instrument").reindex(columns=panels.open.columns)
        out[f"factor:{f}"] = ls_sharpe_zero_cost(sc, panels)
    return out


def build(data: str, signals: list) -> dict:
    from qlib.workflow import R

    meta = market.init(data)
    panels = market.load_panels(data)
    rep = REPO_ROOT / "research" / "reports"
    out_dir = rep / "final" / data
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = json.loads((rep / ("phase1_" + data if (rep / ("phase1_" + data)).exists() else "phase1") / "data_quality.json").read_text(encoding="utf-8"))
    selection = json.loads((rep / "phase3" / data / "selection.json").read_text(encoding="utf-8"))
    ff = load_ff(data)

    R_ = {"data": data, "signals": {}, "flags": meta.get("flags"), "data_hash": meta.get("data_hash"),
          "git_commit": _git("rev-parse", "HEAD"), "selection": selection, "trials": trial_counts(),
          "ff_snapshot_hash": hash_dir(market.store_dir(data) / "ff"), "ff_last_date": f"{ff.index[-1]:%Y-%m-%d}",
          "coverage": quality["tables"].get("universe_coverage"), "quality_summary": {k: quality[k] for k in ("period", "symbols", "cleaning_stats")}}

    first = None
    for sig in signals:
        p4 = json.loads((rep / "phase4" / data / sig / "summary.json").read_text(encoding="utf-8"))
        pdir = market.store_dir(data) / "experiments" / sig / "portfolio"
        spy = pd.read_pickle(pdir / "benchmark_spy.pkl")["ret"]
        daily = {n: pd.read_pickle(pdir / f"{n}.pkl") for n in p4["strategies"] if n != BENCH}
        if first is None:
            first = (sig, daily, spy)
            ts = trial_sharpes(data, panels, daily[PRIMARY].index)
            R_["trial_sharpes_ls_zero_cost_ann"] = {k: v * np.sqrt(ANN) for k, v in ts.items()}
        port_sr = [d["net"].mean() / d["net"].std(ddof=1) for n, d in daily.items() if n != "universe_ew"]
        all_sr = list(ts.values()) + port_sr
        n_trials = R_["trials"]["total"]
        S = {}
        for n, d in daily.items():
            S[n] = {
                "gross": p4["strategies"][n]["gross"], "net": p4["strategies"][n]["net"], "turnover": p4["strategies"][n]["turnover"],
                "dsr": deflated_sharpe(d["net"], n_trials, all_sr),
                "ff": ff_regression(d["net"], ff, excess=n in LONG_ONLY),
                "vs_spy": vs_benchmark(d["net"], spy) if n in LONG_ONLY else None,
                "regimes": regime_table(d["net"], spy, ff["RF"]),
            }
        S[BENCH] = {"gross": p4["strategies"][BENCH]["gross"], "net": p4["strategies"][BENCH]["net"], "turnover": None,
                    "ff": ff_regression(spy, ff, excess=True)}
        R_["signals"][sig] = {"period": p4["period"], "cost_model": p4["cost_model"], "base_aum": p4["base_aum"],
                              "strategies": S, "sensitivity": {n: p4["strategies"][n]["sensitivity"] for n in daily if n != "universe_ew"},
                              "monthly_primary_net": json.loads(monthly_table(daily[PRIMARY]["net"]).to_json()),
                              "ppo_crosscheck": p4["ppo_crosscheck"]}
        R_["signals"][sig]["tearsheet"] = _tearsheet(daily[PRIMARY]["net"], spy, out_dir / f"tearsheet_{sig}_{PRIMARY}.html", f"{sig} / {PRIMARY} (net)")

    (out_dir / "report.json").write_text(json.dumps(R_, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(to_markdown(R_), encoding="utf-8")
    with R.start(experiment_name=f"final_report_{data}", recorder_name="+".join(signals)):
        R.log_params(data_hash_qlib=(R_["data_hash"] or {}).get("qlib"), git_commit=R_["git_commit"], n_trials=R_["trials"]["total"],
                     survivorship_bias=(R_["flags"] or {}).get("survivorship_bias"), chosen=selection["chosen"])
        for sig, v in R_["signals"].items():
            for n, s in v["strategies"].items():
                if n != BENCH:
                    R.log_metrics(**{f"{sig}.{n}.net_sharpe": s["net"]["sharpe"], f"{sig}.{n}.gross_sharpe": s["gross"]["sharpe"],
                                     f"{sig}.{n}.dsr": s["dsr"]["deflated_sharpe"]})
        for f in out_dir.glob("*"):
            R.log_artifact(str(f))
    print(f"report written: {out_dir / 'REPORT.md'}")
    return R_


def _tearsheet(ret: pd.Series, spy: pd.Series, path, title: str) -> dict:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import quantstats as qs

        qs.reports.html(ret.rename("strategy"), benchmark=spy.reindex(ret.index).rename("SPY"), output=str(path), title=title)
        return {"ok": True, "file": path.name}
    except Exception as e:  # 실패하면 숨기지 않고 리포트에 적는다
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _p(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.{d}f}%"


def to_markdown(R_: dict) -> str:
    fl = R_["flags"] or {}
    sel, tr = R_["selection"], R_["trials"]
    L = [f"# 최종 리포트 — {R_['data']}", "",
         "이 문서는 `python -m research.pipelines.report` 가 실행 산출물에서 자동 생성한다. 모든 숫자는 실제 실행 결과다.", ""]
    if fl.get("survivorship_bias", True):
        L += ["> **⚠ 생존편향 존재 (survivorship_bias = true)** — 아래 성과 수치는 전부 상향 편향되어 있다. 3절 참고.", ""]
    L += [f"- 데이터 해시(qlib): `{(R_['data_hash'] or {}).get('qlib')}`", f"- git commit: `{R_['git_commit']}`",
          f"- 총 시도 횟수: **{tr['total']}** (" + ", ".join(f"{k} {v}" for k, v in tr["by_kind"].items()) + ")",
          f"- Phase 3 채택 모델: `{sel['chosen']}` (기준선 `{sel['baseline']}`)", ""]
    for sig, v in R_["signals"].items():
        S, c = v["strategies"], v["cost_model"]
        L += [f"## 1. 성과 — 신호 `{sig}`" + (" (기준선)" if sig == sel["baseline"] else "") + (" (채택)" if sig == sel["chosen"] else ""), "",
              f"기간 {v['period'][0]} ~ {v['period'][1]} (walk-forward 표본 외 구간). 신호 다음 날 시가 체결, 5거래일마다 리밸런싱, 운용자산 ${v['base_aum']:,.0f}.",
              "", "### 비용 전/후", "",
              "| 전략 | 누적 전 | 누적 후 | CAGR 전 | CAGR 후 | 변동성 | Sharpe 전 | Sharpe 후 | Sortino 후 | MDD 후 | 연 턴오버 | 연 비용 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, s in S.items():
            g, nn, t = s["gross"], s["net"], s["turnover"]
            L.append(f"| {n} | {_p(g['total_return'])} | {_p(nn['total_return'])} | {_p(g['cagr'])} | {_p(nn['cagr'])} | {_p(nn['ann_vol'])} | "
                     f"{g['sharpe']:.2f} | {nn['sharpe']:.2f} | {nn['sortino']:.2f} | {_p(nn['max_drawdown'])} | "
                     + (f"{t['annual_turnover']:.1f}x | {_p(t['annual_cost'])} |" if t else "– | – |"))
        L += ["", "### 과적합 보정", "",
              f"시도 {tr['total']}회를 반영한 Deflated Sharpe Ratio(DSR). DSR 은 '시도 횟수만큼 골랐을 때 우연히 나올 최대 Sharpe'를 넘을 확률이다. 0.95 미만이면 유의하다고 볼 수 없다.",
              "", "| 전략 | Sharpe 후 | PSR (Sharpe>0 확률) | 우연히 기대되는 최대 Sharpe | DSR |", "|---|---|---|---|---|"]
        for n, s in S.items():
            if "dsr" in s:
                d = s["dsr"]
                L.append(f"| {n} | {s['net']['sharpe']:.2f} | {d['psr_vs_zero']:.3f} | {d['expected_max_sharpe_ann']:.2f} | {d['deflated_sharpe']:.3f} |")
        L += ["", "### 벤치마크(SPY) 대비 — 롱온리 전략, 비용 후", "", "| 전략 | 초과수익(연) | 추적오차 | 정보비율 | 베타 | 상관 |", "|---|---|---|---|---|---|"]
        for n, s in S.items():
            if s.get("vs_spy"):
                b = s["vs_spy"]
                L.append(f"| {n} | {_p(b['active_ann'])} | {_p(b['tracking_error'])} | {b['information_ratio']:.2f} | {b['beta']:.2f} | {b['correlation']:.2f} |")
        L += ["", "### 팩터 노출 (Fama-French 5팩터 + 모멘텀, 일별 회귀, Newey-West 5 lag)", "",
              f"롱온리와 SPY 는 무위험수익률 초과분, 롱숏은 수익률 그대로 회귀. 시가-시가 수익률이라 당일·익일 팩터를 함께 넣고 베타를 합산했다. FF 데이터 마지막 날짜 {R_['ff_last_date']}.",
              "", "| 전략 | 알파(연) | t(알파) | MKT | SMB | HML | RMW | CMA | MOM | R² | 회귀 기간 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, s in S.items():
            f = s.get("ff") or {}
            if "betas" in f:
                b = f["betas"]
                L.append(f"| {n} | {_p(f['alpha_ann'])} | {f['alpha_tstat']:.2f} | " + " | ".join(f"{b[k]:.2f}" for k in ("MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"))
                         + f" | {f['r2']:.2f} | {f['period'][0]} ~ {f['period'][1]} |")
            elif f:
                L.append(f"| {n} | 회귀 불가: {f.get('error')} |||||||||| ")
        L += ["", f"### 구간별 성과 — `{PRIMARY}` (비용 후)", "", "| 구간 | 일수 | 연환산 수익 | Sharpe | SPY 연환산 | 초과(연) |", "|---|---|---|---|---|---|"]
        for r in S[PRIMARY]["regimes"]:
            L.append(f"| {r['regime']} | {r['days']} | " + (f"{_p(r['ann_return'])} | {r['sharpe']:.2f} | {_p(r['spy_ann_return'])} | {_p(r['active_ann'])} |" if "sharpe" in r else "표본 부족 ||||"))
        mt = pd.DataFrame(v["monthly_primary_net"])
        mt.index = mt.index.astype(int)
        L += ["", f"### 월별 수익률 — `{PRIMARY}` (비용 후, %)", "", "| 연도 | " + " | ".join(str(c) for c in mt.columns) + " |", "|---|" + "---|" * len(mt.columns)]
        for y, row in mt.sort_index().iterrows():
            L.append(f"| {y} | " + " | ".join("" if pd.isna(x) else f"{x * 100:.1f}" for x in row) + " |")
        ts = v["tearsheet"]
        L += ["", "### quantstats tear sheet", "", (f"`{ts['file']}` (누적수익, 월별 히트맵, 낙폭 등)" if ts["ok"] else f"생성 실패: `{ts['error']}`"), "",
              "### 비용 민감도 (비용 후 Sharpe)", ""]
        for n, g in v["sensitivity"].items():
            aums = sorted({k.split("|")[0] for k in g}, key=lambda s: float(s.split("=")[1]))
            mults = sorted({float(k.split("x")[-1]) for k in g})
            L += [f"**{n}**", "", "| 운용자산 | " + " | ".join(f"×{m}" for m in mults) + " | 평균 참여율 | 미체결 |", "|---|" + "---|" * (len(mults) + 2)]
            for a in aums:
                b = g[f"{a}|cost_x1.0"]
                L.append(f"| ${float(a.split('=')[1]):,.0f} | " + " | ".join(f"{g[f'{a}|cost_x{m}']['net_sharpe']:.2f}" for m in mults)
                         + f" | {_p(b['avg_participation'])} | {_p(b['avg_clipped'])} |")
            L.append("")
    # ---- 해석 ----
    L += ["## 2. 모델 선택 근거", "", "| 실험 | Rank IC | t-stat | 기준선 대비 차이 (t) |", "|---|---|---|---|"]
    for n, e in sel["experiments"].items():
        vb = sel["vs_baseline"].get(n)
        L.append(f"| {n} | {e['rank_ic_mean']:.4f} | {e['rank_ic_tstat_overlap_adj']:.2f} | " + (f"{vb['mean_diff']:+.4f} ({vb['t_overlap_adj']:+.2f})" if vb else "기준선") + " |")
    ts = R_.get("trial_sharpes_ls_zero_cost_ann", {})
    if ts:
        L += ["", "시도별 표준 지표(상/하위 20% 롱숏, 비용 0, 연환산 Sharpe): " + ", ".join(f"{k.split(':')[1]} {v:.2f}" for k, v in ts.items()) + "."]
    L += ["", "## 3. 결과 해석 — 반드시 읽을 것", "", "### 생존편향", ""]
    cov = R_.get("coverage") or []
    if cov:
        L += ["유니버스 멤버십은 위키피디아 편입·편출 이력으로 복원한 근사치이고, 편출된 종목 다수는 Yahoo 에 시세가 없다.",
              "아래 커버리지가 1 에 못 미치는 만큼 '당시 지수에 있었지만 백테스트에는 없는 종목'이 있다. 이들은 인수·상장폐지된 종목이어서 "
              "누락은 무작위가 아니다(파산·부진 종목이 빠지면 성과가 좋아 보이고, 프리미엄 받고 인수된 종목이 빠지면 나빠 보인다).", "",
              "| 연도 | 멤버 수 | 시세 있는 멤버 | 커버리지 |", "|---|---|---|---|"]
        L += [f"| {int(r['year'])} | {r['members']:.0f} | {r['members_with_data']:.0f} | {r['coverage']:.1%} |" for r in cov]
    L += ["", "### 데이터 한계", "",
          "- Yahoo 시세는 조회 시점 기준으로 수정되며 재현은 raw 스냅샷 해시로만 보장된다.",
          "- 수정주가의 수준에는 미래 배당·분할이 들어 있다. 비율 피처만 사용해 회피했다.",
          "- VWAP 이 없어 Alpha158 은 157개, Alpha360 은 300개 피처로 줄였다. 원 논문/벤치마크 수치와 직접 비교할 수 없다.",
          "- 펀더멘털(가치·퀄리티) 팩터는 공시일 기준 데이터가 없어 넣지 않았다.",
          "- 섹터는 현재 GICS 분류다. 섹터 제약·중립화에 미래 정보가 약간 섞여 있다.",
          "- 벤치마크 SPY 는 배당 재투자 가정의 수정주가다.",
          f"- 정제 통계: {json.dumps(R_['quality_summary']['cleaning_stats'], ensure_ascii=False)}",
          "", "### 과적합 위험", "",
          f"- 지금까지 평가한 설정은 {tr['total']}개다(팩터·모델·포트폴리오 구성 포함). 하이퍼파라미터 탐색은 하지 않았지만 모델 채택과 전략 비교는 test 구간을 보고 했다.",
          "- DSR 이 0.95 에 못 미치는 전략은 '여러 번 시도해서 얻은 우연'과 구분되지 않는다.",
          "- test 구간은 약 6년(6개 fold)뿐이다. 한 국면(2021~2026)의 결과이며 다른 국면에서의 성과를 보장하지 않는다.",
          "", "### 비용 가정", "",
          "- 비용 파라미터는 실제 체결 데이터로 보정한 값이 아니라 대형주에 대한 가정이다. 민감도 표에서 ×2, ×4 를 함께 볼 것.",
          "- 시가 단일가에 전량 체결된다고 가정한다. 참여율 상한을 넘는 주문은 체결하지 않고 다음 리밸런싱으로 넘긴다.",
          "- 공매도는 대차비용만 반영했다. 대차 불가·리콜·증거금은 반영하지 않았다.",
          "- 세금, 현금 이자, 배당 원천징수는 반영하지 않았다."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sp500")
    ap.add_argument("--signals", nargs="+", default=[BASELINE])
    a = ap.parse_args()
    build(a.data, a.signals)
