"""Phase 2 누수 방지 테스트: 라벨/체결 타이밍, 피처 룩어헤드, Purged 분할, 횡단면 정규화."""

import numpy as np
import pandas as pd
import pytest

from research.data.cleaning import CleaningConfig, clean_prices
from research.factors.library import FACTORS, factor_fields
from research.pipelines.labels import LabelSpec, execution_date, label_end_date
from research.pipelines.leakage import build_store, compute_fields, init_qlib, lookahead_report
from research.pipelines.splits import Fold, assert_no_label_overlap, walk_forward_folds

from .conftest import make_prices

N_DAYS, CUT = 420, 330


@pytest.fixture(scope="module")
def data():
    prices, cal = make_prices(symbols=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"), periods=N_DAYS, seed=3)
    cleaned, _, _ = clean_prices(prices, cal, CleaningConfig())
    return cleaned, cal


@pytest.fixture(scope="module")
def store(data, tmp_path_factory):
    cleaned, cal = data
    return build_store(cleaned, cal, tmp_path_factory.mktemp("store"))


# ---------- 라벨 / 체결 타이밍 ----------
@pytest.mark.parametrize("entry,col", [("next_open", "open"), ("next_close", "close")])
def test_label_matches_documented_timing(data, store, entry, col):
    """라벨(t) == price[t+1+N] / price[t+1] - 1  (t 일 가격은 라벨에 들어가지 않는다)"""
    cleaned, cal = data
    spec = LabelSpec(horizon=5, entry=entry)
    lab = compute_fields(store, [spec.expr], ["y"])["y"].unstack()
    px = cleaned.pivot(index="date", columns="symbol", values=col)  # adjclose == close 인 합성 데이터
    expected = px.shift(-spec.span) / px.shift(-spec.execution_lag) - 1
    # qlib 은 float32 로 저장하므로 절대오차 기준으로 비교한다
    np.testing.assert_allclose(lab.to_numpy(float), expected.reindex_like(lab).to_numpy(float), atol=2e-5)
    assert lab.iloc[-spec.span :].isna().all().all()  # 미래 가격이 없는 마지막 span 일은 라벨이 없어야 한다


def test_signal_and_execution_dates_are_separate():
    cal = list(pd.bdate_range("2021-01-04", periods=30))
    spec = LabelSpec(horizon=5)
    assert spec.execution_lag >= 1  # 같은 날 종가 체결 금지
    assert execution_date(cal[3], cal) == cal[4]
    assert label_end_date(cal[3], cal, spec) == cal[3 + 6]
    with pytest.raises(IndexError):
        execution_date(cal[-1], cal)
    with pytest.raises(ValueError):
        LabelSpec(entry="same_close")


# ---------- 피처 룩어헤드 ----------
def test_alpha158_alpha360_and_factors_do_not_look_ahead(data, tmp_path):
    from research.pipelines.handlers import alpha158_fields, alpha360_fields

    cleaned, cal = data
    init_qlib_needed = [alpha158_fields(), alpha360_fields(), factor_fields()]
    fields = [f for fs, _ in init_qlib_needed for f in fs]
    names = [f"{i}_{n}" for i, (_, ns) in enumerate(init_qlib_needed) for n in ns]
    assert len(alpha158_fields()[0]) == 157 and len(alpha360_fields()[0]) == 300 and len(FACTORS) == 10
    assert not any("$vwap" in f for f in fields)
    rep = lookahead_report(cleaned, cal, fields, names, cal[CUT], tmp_path)
    assert rep["n_compared"].min() > 0
    assert not rep["leaks"].any(), rep[rep["leaks"]]


def test_leak_detector_catches_future_features_and_label(data, tmp_path):
    """검사기 자체의 검증: 미래를 보는 식과 라벨은 반드시 '누수'로 잡혀야 한다."""
    cleaned, cal = data
    spec = LabelSpec(horizon=5)
    fields = ["Ref($close,-1)/$close", "$close/Ref($close,1)", spec.expr, "Mean($close,5)/$close"]
    rep = lookahead_report(cleaned, cal, fields, ["leaky", "clean", "label", "clean2"], cal[CUT], tmp_path)
    rep = rep.set_index("field")
    assert rep.loc["leaky", "leaks"] and rep.loc["label", "leaks"]
    assert not rep.loc["clean", "leaks"] and not rep.loc["clean2", "leaks"]
    # 라벨은 cut 직전 span 일 x 종목 수 만큼만 달라져야 한다
    assert rep.loc["label", "n_diff"] == spec.span * cleaned["symbol"].nunique()


# ---------- Purged walk-forward ----------
def test_walk_forward_folds_are_purged_and_ordered():
    cal = list(pd.bdate_range("2015-01-01", periods=3000))
    spec = LabelSpec(horizon=5)
    folds = walk_forward_folds(cal, spec.span, train_days=1008, valid_days=252, test_days=252, step_days=252)
    assert len(folds) >= 6
    assert_no_label_overlap(folds, cal, spec.span)
    c = pd.DatetimeIndex(cal)
    for f in folds:
        assert c.get_loc(f.valid[0]) - c.get_loc(f.train[1]) == spec.span + 1
        assert c.get_loc(f.test[0]) - c.get_loc(f.valid[1]) == spec.span + 1
    # 마지막 test 일의 라벨도 캘린더 안에서 확정되어야 한다
    assert c.get_loc(folds[-1].test[1]) + spec.span <= len(c) - 1
    # embargo 를 주면 간격이 그만큼 더 벌어진다
    f2 = walk_forward_folds(cal, spec.span, embargo_days=10)[0]
    assert c.get_loc(f2.valid[0]) - c.get_loc(f2.train[1]) == spec.span + 11


def test_unpurged_split_is_rejected():
    cal = list(pd.bdate_range("2015-01-01", periods=600))
    bad = Fold(0, (cal[0], cal[299]), (cal[300], cal[399]), (cal[400], cal[499]))  # 구간이 맞닿아 있음
    with pytest.raises(AssertionError):
        assert_no_label_overlap([bad], cal, LabelSpec(horizon=5).span)
    assert_no_label_overlap([bad], cal, 0)  # 라벨이 미래를 보지 않는다면 문제 없음


# ---------- 횡단면 정규화 ----------
def test_non_cross_sectional_processors_are_rejected():
    from research.pipelines.processors import assert_cross_sectional_only

    assert_cross_sectional_only([{"class": "CSZScoreNorm"}, {"class": "CSRankNorm"}, {"class": "DropnaLabel"}])
    for bad in ("ZScoreNorm", "RobustZScoreNorm", "MinMaxNorm", "SomethingUnknown"):
        with pytest.raises(ValueError):
            assert_cross_sectional_only([{"class": bad}])


def test_handler_output_is_invariant_to_future_data(data, store, tmp_path):
    """핸들러(피처 + 횡단면 전처리)를 통과한 추론용 데이터가 cut 이후 데이터 유무와 무관해야 한다."""
    from qlib.data.dataset.handler import DataHandlerLP

    from research.pipelines.handlers import LINEAR_INFER, USHandler

    cleaned, cal = data
    cut = cal[CUT]

    def infer_frame(qdir):
        init_qlib(qdir)
        h = USHandler("all", cal[0], cut, feature_set="factors", infer_processors=LINEAR_INFER)
        return h.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)

    full = infer_frame(store)
    trunc = infer_frame(build_store(cleaned, cal, tmp_path / "t", cut=cut))
    assert len(full) > 0
    pd.testing.assert_frame_equal(full, trunc, rtol=1e-5)
    # 순위 정규화는 일자별로 이루어진다: 값의 범위가 일자와 무관하게 일정하다
    assert full.abs().max().max() < 1.8


def test_untradable_rows_are_excluded_from_learning(data, store):
    from qlib.data.dataset.handler import DataHandlerLP

    from research.pipelines.handlers import USHandler

    cleaned, cal = data
    init_qlib(store)
    h = USHandler("all", cal[0], cal[-1], feature_set="factors")
    learn = h.fetch(col_set="label", data_key=DataHandlerLP.DK_L)
    idx = learn.index.to_frame(index=False)
    trad = cleaned.set_index(["date", "symbol"])["tradable"]
    got = trad.reindex(pd.MultiIndex.from_frame(idx[["datetime", "instrument"]])).to_numpy()
    assert len(learn) > 0 and (got > 0.5).all()
    # 라벨은 일자별 z-score 로만 정규화된다
    assert learn.groupby(level="datetime").mean().abs().max().max() < 1e-6
