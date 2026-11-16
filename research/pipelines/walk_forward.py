"""Walk-forward 학습/예측.

    python -m research.pipelines.walk_forward --data sp500 --experiment alpha158_lgb
    python -m research.pipelines.walk_forward --data sp500 --all

- 재학습 주기: 252 거래일(약 1년). train 1008일 / valid 252일 / test 252일, 구간 사이 라벨 span 만큼 purge.
- 하이퍼파라미터는 사전에 고정한다(탐색 없음). 모든 실험은 trials.jsonl 에 기록된다.
- 산출물: <store>/experiments/<name>/pred.pkl (gitignore), research/reports/phase3/<data>/<name>.json
- MLflow: research/mlruns (gitignore) 에 params/metrics/artifacts 기록.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from research.data.manifest import REPO_ROOT, _git

from . import market
from .labels import LabelSpec
from .splits import assert_no_label_overlap, walk_forward_folds
from .stats import ic_series, ic_summary
from .trials import record_trial

SEED = 42
WF = dict(train_days=1008, valid_days=252, test_days=252, step_days=252, embargo_days=0, start_index=252)
LABEL = dict(horizon=5, entry="next_open")

# qlib 벤치마크(Alpha158 + LightGBM)의 공개 하이퍼파라미터를 그대로 쓴다. 이 데이터로 튜닝하지 않았다.
LGB_PARAMS = dict(
    loss="mse", colsample_bytree=0.8879, learning_rate=0.2, subsample=0.8789, lambda_l1=205.6999,
    lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=4, seed=SEED, deterministic=True,
    force_row_wise=True, verbosity=-1,
)
RIDGE_PARAMS = dict(estimator="ridge", alpha=100.0)

EXPERIMENTS = {
    "alpha158_lgb": dict(feature_set="alpha158", model="lgb", params=LGB_PARAMS, note="기준선"),
    "alpha360_lgb": dict(feature_set="alpha360", model="lgb", params=LGB_PARAMS, note="Alpha360 비교"),
    "alpha158_ridge": dict(feature_set="alpha158", model="ridge", params=RIDGE_PARAMS, note="단순 선형"),
    "factors_lgb": dict(feature_set="factors", model="lgb", params=LGB_PARAMS, note="커스텀 팩터 10개"),
    "factors_ridge": dict(feature_set="factors", model="ridge", params=RIDGE_PARAMS, note="커스텀 팩터 선형"),
    "factors_equal": dict(feature_set="factors", model="equal", params={}, note="팩터 순위 단순 평균(학습 없음)"),
}
BASELINE = "alpha158_lgb"


def _model(kind: str, params: dict):
    if kind == "lgb":
        from qlib.contrib.model.gbdt import LGBModel

        return LGBModel(**params)
    if kind == "ridge":
        from qlib.contrib.model.linear import LinearModel

        return LinearModel(**params)
    raise ValueError(kind)


def out_dirs(data: str, name: str):
    exp = market.store_dir(data) / "experiments" / name
    rep = REPO_ROOT / "research" / "reports" / "phase3" / data
    exp.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    return exp, rep


def run_experiment(data: str, name: str, universe: str = None) -> dict:
    from qlib.workflow import R

    meta = market.init(data)
    with R.start(experiment_name=f"walk_forward_{data}", recorder_name=name):
        return _run(data, name, universe, meta)


def _run(data: str, name: str, universe: str, meta: dict) -> dict:
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    from .handlers import LINEAR_INFER, TREE_INFER, USHandler

    random.seed(SEED)
    np.random.seed(SEED)
    cfg = EXPERIMENTS[name]
    universe = universe or data
    cal = pd.DatetimeIndex(D.calendar(freq="day"))
    spec = LabelSpec(**LABEL)
    folds = walk_forward_folds(cal, spec.span, **WF)
    assert_no_label_overlap(folds, cal, spec.span)

    infer = TREE_INFER if cfg["model"] == "lgb" else LINEAR_INFER
    handler = USHandler(universe, cal[0], cal[-1], feature_set=cfg["feature_set"], infer_processors=infer, **LABEL)

    preds, fold_rows = [], []
    for f in folds:
        if cfg["model"] == "equal":
            x = handler.fetch(slice(*f.test), col_set="feature", data_key=DataHandlerLP.DK_I)
            p = x.mean(axis=1)
            best_iter = None
        else:
            ds = DatasetH(handler, segments=f.segments())
            m = _model(cfg["model"], cfg["params"])
            m.fit(ds)
            p = m.predict(ds, segment="test")
            best_iter = getattr(getattr(m, "model", None), "best_iteration", None)
        preds.append(p)
        fold_rows.append({"fold": f.k, **{k: [f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"] for k, (a, b) in f.segments().items()},
                          "n_pred": int(len(p)), "best_iteration": best_iter})
    pred = pd.concat(preds).sort_index().rename("score")
    if pred.index.names[0] != "datetime":
        pred = pred.swaplevel().sort_index()

    # 평가는 정규화하지 않은 실제 라벨로 한다
    y = D.features(D.instruments(universe), [spec.expr], freq="day").iloc[:, 0].swaplevel().sort_index()
    y = y.reindex(pred.index)
    summ = ic_summary(pred, y, horizon=spec.span)
    ric = ic_series(pred, y, "spearman")
    for row in fold_rows:
        seg = ric.loc[row["test"][0] : row["test"][1]]
        row["rank_ic"] = float(seg.mean())

    exp_dir, rep_dir = out_dirs(data, name)
    pred.to_pickle(exp_dir / "pred.pkl")
    ric.to_pickle(exp_dir / "rank_ic.pkl")
    result = {
        "experiment": name,
        "note": cfg["note"],
        "data": data,
        "universe": universe,
        "data_hash": meta.get("data_hash"),
        "flags": meta.get("flags"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "seed": SEED,
        "label": {**LABEL, "expr": spec.expr, "span": spec.span},
        "walk_forward": {**WF, "retrain_every_days": WF["step_days"], "purge_days": spec.span, "n_folds": len(folds)},
        "feature_set": cfg["feature_set"],
        "n_features": len(handler.get_cols(col_set="feature")),
        "model": cfg["model"],
        "params": cfg["params"],
        "test_period": [f"{pred.index[0][0]:%Y-%m-%d}", f"{pred.index[-1][0]:%Y-%m-%d}"],
        "metrics": summ,
        "folds": fold_rows,
    }
    (rep_dir / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    record_trial(
        "model", f"{data}/{name}",
        {"feature_set": cfg["feature_set"], "model": cfg["model"], "params": cfg["params"], "label": LABEL, "wf": WF},
        {k: summ[k] for k in ("rank_ic_mean", "rank_ic_ir", "ic_mean")},
    )
    _mlflow_log(result, rep_dir / f"{name}.json")
    print(json.dumps({"experiment": name, **summ}, indent=1))
    return result


def _mlflow_log(result: dict, artifact: Path) -> None:
    """qlib 의 실험 관리자(R, MLflow 백엔드)에 기록한다. 학습 중 LightGBM 손실 곡선도 같은 run 에 남는다."""
    from qlib.workflow import R

    R.log_params(**{
        "feature_set": result["feature_set"], "model": result["model"], "seed": result["seed"],
        "n_features": result["n_features"], "git_commit": result["git_commit"], "git_dirty": result["git_dirty"],
        "data_hash_qlib": (result["data_hash"] or {}).get("qlib"), "label_expr": result["label"]["expr"],
        "survivorship_bias": (result["flags"] or {}).get("survivorship_bias"),
        **{f"wf_{k}": v for k, v in result["walk_forward"].items()},
        **{f"p_{k}": v for k, v in result["params"].items()},
    })
    R.log_metrics(**{k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))})
    R.log_artifact(str(artifact))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sp500")
    ap.add_argument("--experiment", choices=list(EXPERIMENTS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for n in list(EXPERIMENTS) if a.all else [a.experiment or BASELINE]:
        run_experiment(a.data, n)
