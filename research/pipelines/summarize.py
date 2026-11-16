"""Phase 3 요약: 모델 비교, 채택 규칙, 시도 횟수.

    python -m research.pipelines.summarize --data sp500

채택 규칙(사전 고정): 단순한 것부터 시작해, 더 복잡한 모델은 '현재 채택된 모델 대비 일별 Rank IC 차이'의
t-stat(라벨 겹침 보정)이 2 를 넘을 때만 채택한다. 넘지 못하면 단순한 모델을 유지한다.
주의: 이 비교는 test 구간 결과를 보고 하는 선택이므로 그 자체가 데이터 스누핑이다. 그래서 모든 후보를 시도 횟수에 넣는다.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from research.data.manifest import REPO_ROOT

from . import market
from .trials import trial_counts
from .walk_forward import BASELINE, EXPERIMENTS, LABEL

# 단순 -> 복잡
COMPLEXITY = ["factors_equal", "factors_ridge", "alpha158_ridge", "factors_lgb", "alpha158_lgb", "alpha360_lgb"]
T_REQUIRED = 2.0


def paired_t(a: pd.Series, b: pd.Series, span: int) -> dict:
    d = (a - b).dropna()
    t = d.mean() / d.std(ddof=1) * np.sqrt(len(d) / span)
    return {"mean_diff": float(d.mean()), "t_overlap_adj": float(t), "n_days": int(len(d))}


def run(data: str) -> dict:
    rep = REPO_ROOT / "research" / "reports" / "phase3" / data
    span = LABEL["horizon"] + 1
    res, ric = {}, {}
    for name in COMPLEXITY:
        fp = rep / f"{name}.json"
        if fp.exists():
            res[name] = json.loads(fp.read_text(encoding="utf-8"))
            ric[name] = pd.read_pickle(market.store_dir(data) / "experiments" / name / "rank_ic.pkl")
    order = [n for n in COMPLEXITY if n in res]
    chosen, steps = order[0], []
    for cand in order[1:]:
        pt = paired_t(ric[cand], ric[chosen], span)
        adopt = pt["t_overlap_adj"] > T_REQUIRED
        steps.append({"candidate": cand, "incumbent": chosen, **pt, "adopted": bool(adopt)})
        if adopt:
            chosen = cand
    vs_base = {n: paired_t(ric[n], ric[BASELINE], span) for n in order if n != BASELINE and BASELINE in ric}
    out = {"data": data, "baseline": BASELINE, "chosen": chosen, "rule_t_required": T_REQUIRED, "steps": steps,
           "vs_baseline": vs_base, "trials": trial_counts(),
           "experiments": {n: {"note": res[n]["note"], "n_features": res[n]["n_features"], **res[n]["metrics"],
                               "fold_rank_ic": [f["rank_ic"] for f in res[n]["folds"]]} for n in order}}
    (rep / "selection.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (rep / "summary.md").write_text(to_markdown(out, res[order[-1] if BASELINE not in res else BASELINE]), encoding="utf-8")
    print((rep / "summary.md").read_text(encoding="utf-8"))
    return out


def to_markdown(o: dict, ref: dict) -> str:
    wf = ref["walk_forward"]
    L = [f"# Phase 3 — 모델 비교 ({o['data']})", ""]
    if (ref.get("flags") or {}).get("survivorship_bias", True):
        L += ["> **⚠ 생존편향 존재** — 아래 수치는 상향 편향되어 있을 수 있다.", ""]
    L += [
        f"- 라벨: `{ref['label']['expr']}` (t 일 종가 신호 → t+1 시가 진입, {ref['label']['horizon']}거래일 보유)",
        f"- Walk-forward: train {wf['train_days']}일 / valid {wf['valid_days']}일 / test {wf['test_days']}일, "
        f"**{wf['retrain_every_days']}거래일마다 재학습**, 구간 사이 {wf['purge_days']}일 purge, fold {wf['n_folds']}개",
        f"- test 구간(표본 외): {ref['test_period'][0]} ~ {ref['test_period'][1]}",
        f"- 하이퍼파라미터: 사전 고정(탐색 없음). 시드 {ref['seed']}. 데이터 해시 `{(ref['data_hash'] or {}).get('qlib', '')[:16]}…`",
        "",
        "## Fold 구성",
        "",
        "| fold | train | valid | test |",
        "|---|---|---|---|",
    ]
    for f in ref["folds"]:
        L.append(f"| {f['fold']} | {f['train'][0]} ~ {f['train'][1]} | {f['valid'][0]} ~ {f['valid'][1]} | {f['test'][0]} ~ {f['test'][1]} |")
    L += ["", "## 모델 비교 (전체 test 구간, 5일 선행수익률 대비)", "",
          "| 실험 | 설명 | 피처 수 | IC | Rank IC | Rank ICIR | t-stat | Rank IC>0 비율 | 기준선 대비 Rank IC 차이 (t) |", "|---|---|---|---|---|---|---|---|---|"]
    for n, e in o["experiments"].items():
        vb = o["vs_baseline"].get(n)
        L.append(
            f"| {n}{' **(기준선)**' if n == o['baseline'] else ''} | {e['note']} | {e['n_features']} | {e['ic_mean']:.4f} | "
            f"{e['rank_ic_mean']:.4f} | {e['rank_ic_ir']:.3f} | {e['rank_ic_tstat_overlap_adj']:.2f} | {e['rank_ic_positive_rate']:.1%} | "
            + (f"{vb['mean_diff']:+.4f} ({vb['t_overlap_adj']:+.2f}) |" if vb else "– |")
        )
    L += ["", "t-stat 은 5일 라벨의 겹침을 감안해 유효 표본을 1/6 로 줄여 계산했다. |t| < 2 는 0 과 구분되지 않는다.", "",
          "## Fold 별 Rank IC", "", "| 실험 | " + " | ".join(f"fold {i}" for i in range(wf["n_folds"])) + " |", "|---|" + "---|" * wf["n_folds"]]
    for n, e in o["experiments"].items():
        L.append(f"| {n} | " + " | ".join(f"{v:.4f}" for v in e["fold_rank_ic"]) + " |")
    L += ["", "## 모델 채택", "",
          f"규칙: 단순한 모델에서 시작해, 더 복잡한 모델은 현재 채택 모델 대비 Rank IC 차이의 t-stat 이 {o['rule_t_required']} 를 넘을 때만 채택.", "",
          "| 후보 | 현재 채택 | Rank IC 차이 | t-stat | 채택 |", "|---|---|---|---|---|"]
    for s in o["steps"]:
        L.append(f"| {s['candidate']} | {s['incumbent']} | {s['mean_diff']:+.4f} | {s['t_overlap_adj']:+.2f} | {'예' if s['adopted'] else '아니오'} |")
    t = o["trials"]
    L += ["", f"**채택: `{o['chosen']}`**", "",
          "이 선택은 test 구간 결과를 보고 한 것이므로 데이터 스누핑에 해당한다. 모든 후보를 시도 횟수에 포함한다.", "",
          "## 시도 횟수", "", f"- 총 시도: **{t['total']}** (" + ", ".join(f"{k} {v}" for k, v in t["by_kind"].items()) + ")",
          "- 기록 파일: `research/reports/trials.jsonl` (같은 설정의 재실행은 새 시도로 세지 않는다)",
          "- 과적합 보정 지표(Deflated Sharpe)는 포트폴리오 수익률이 필요하므로 최종 리포트에서 계산한다."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sp500")
    run(ap.parse_args().data)
