"""Phase 1 데이터 레이어 테스트: 정제 규칙, 룩어헤드, PIT 유니버스, 공시일 지연, 해시, qlib 왕복."""

import json

import numpy as np
import pandas as pd
import pytest

from research.data.cleaning import CleaningConfig, clean_prices
from research.data.fundamentals import align_to_report_date
from research.data.manifest import hash_dir, load_data_flags
from research.data.sources.base import PRICE_COLUMNS, get_source, validate_prices
from research.data.universe import KIND_PIT, KIND_STATIC, build_membership, write_instruments

from .conftest import FakePITSource, make_prices

CFG = CleaningConfig(min_price=5.0, min_median_dollar_volume=5e6, adv_window=20)


# ---------- 정제 ----------
def test_duplicates_removed_and_schema(synth):
    prices, cal = synth
    dup = pd.concat([prices, prices.iloc[:5]], ignore_index=True)
    cleaned, _, stats = clean_prices(dup, cal, CFG)
    assert stats["duplicate_rows_removed"] == 5
    assert not cleaned.duplicated(["symbol", "date"]).any()
    assert set(PRICE_COLUMNS) <= set(cleaned.columns)


def test_invalid_price_is_nan_and_flagged(synth):
    prices, cal = synth
    prices.loc[10, "close"] = -1.0
    prices.loc[20, "high"] = prices.loc[20, "low"] * 0.5
    cleaned, events, _ = clean_prices(prices, cal, CFG)
    bad = cleaned[cleaned["invalid_price"]]
    assert len(bad) == 2 and bad["close"].isna().all()
    assert (events["flag"] == "invalid_price").sum() == 2


def test_missing_trading_day_and_zero_volume_are_halted(synth):
    prices, cal = synth
    d_missing, d_zero = cal[50], cal[60]
    prices = prices[~((prices["symbol"] == "AAA") & (prices["date"] == d_missing))].copy()
    prices.loc[(prices["symbol"] == "AAA") & (prices["date"] == d_zero), "volume"] = 0
    cleaned, _, _ = clean_prices(prices, cal, CFG)
    a = cleaned[cleaned["symbol"] == "AAA"].set_index("date")
    assert a.loc[d_missing, "halted"] and a.loc[d_zero, "halted"]
    assert a.loc[d_missing, "tradable"] == 0 and a.loc[d_zero, "tradable"] == 0
    assert a["halted"].sum() == 2


def test_unadjusted_split_is_flagged_but_adjusted_split_is_not(synth):
    prices, cal = synth
    d = cal[100]
    # AAA: 2:1 분할이 수정주가에 반영되지 않음(raw, adj 모두 절반)
    m = (prices["symbol"] == "AAA") & (prices["date"] >= d)
    prices.loc[m, ["open", "high", "low", "close", "adjclose"]] /= 2
    # BBB: 정상 반영(raw 만 절반, adjclose 는 연속)
    m = (prices["symbol"] == "BBB") & (prices["date"] >= d)
    prices.loc[m, ["open", "high", "low", "close"]] /= 2
    cleaned, events, _ = clean_prices(prices, cal, CFG)
    sp = events[events["flag"] == "suspected_split"]
    assert list(sp["symbol"]) == ["AAA"] and list(sp["date"]) == [d]
    ex = events[events["flag"] == "extreme_return"]
    assert "BBB" not in set(ex["symbol"])


def test_liquidity_filter_min_price_and_warmup():
    prices, cal = make_prices(symbols=("PENNY", "BIG"), price0=1.0)  # PENNY ~ $1, BIG ~ $2
    prices.loc[prices["symbol"] == "BIG", ["open", "high", "low", "close", "adjclose"]] *= 50
    cleaned, _, _ = clean_prices(prices, cal, CFG)
    assert cleaned.loc[cleaned["symbol"] == "PENNY", "tradable"].sum() == 0
    big = cleaned[cleaned["symbol"] == "BIG"].reset_index(drop=True)
    assert big.loc[: CFG.adv_window - 1, "tradable"].sum() == 0  # 윈도우가 차기 전에는 거래 불가
    assert big.loc[CFG.adv_window + 1 :, "tradable"].mean() > 0.95


# ---------- 룩어헤드 ----------
@pytest.mark.parametrize("cut", [80, 150, 250])
def test_cleaning_has_no_lookahead(synth, cut):
    """cut 시점 이후 데이터를 바꾸거나 지워도 cut 이전의 정제 결과는 동일해야 한다."""
    prices, cal = synth
    full, _, _ = clean_prices(prices, cal, CFG)
    t = cal[cut]

    truncated, _, _ = clean_prices(prices[prices["date"] <= t], cal, CFG)
    shocked = prices.copy()
    fut = shocked["date"] > t
    shocked.loc[fut, ["open", "high", "low", "close", "adjclose"]] *= 7.3
    shocked.loc[fut, "volume"] *= 0.001
    shocked_c, _, _ = clean_prices(shocked, cal, CFG)

    cols = ["symbol", "date", "close", "adjclose", "volume", "tradable", "halted", "extreme_return", "suspected_split"]
    ref = full.loc[full["date"] <= t, cols].reset_index(drop=True)
    for other in (truncated, shocked_c):
        pd.testing.assert_frame_equal(ref, other.loc[other["date"] <= t, cols].reset_index(drop=True))


def test_tradable_does_not_use_same_day_information(synth):
    """t 일의 tradable 은 t 일 가격/거래량과 무관해야 한다(t-1 기준). 단, t 일 거래정지는 예외."""
    prices, cal = synth
    base, _, _ = clean_prices(prices, cal, CFG)
    t = cal[120]
    mod = prices.copy()
    m = mod["date"] == t
    mod.loc[m, ["open", "high", "low", "close", "adjclose"]] *= 0.01  # 당일 가격 폭락
    mod.loc[m, "volume"] = 1.0  # 당일 거래대금 급감 (정지는 아님)
    after, _, _ = clean_prices(mod, cal, CFG)
    assert (base.loc[base["date"] == t, "tradable"].values == after.loc[after["date"] == t, "tradable"].values).all()


# ---------- PIT 유니버스 ----------
def test_static_list_is_flagged_as_survivorship_biased(synth):
    prices, cal = synth
    src = get_source("yahoo_qlib")
    assert src.capabilities.survivorship_bias
    mem = build_membership(src, "u", ["AAA", "BBB", "CCC"], prices)
    assert mem.kind == KIND_STATIC and mem.survivorship_bias


def test_pit_membership_is_not_backfilled(synth, tmp_path):
    prices, cal = synth
    # BBB 는 중간에 편입, CCC 는 중간에 편출. AAA 는 두 번 편입(구간 2개)
    table = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "CCC"],
            "start": [cal[0], cal[200], cal[100], cal[0]],
            "end": [cal[50], cal[299], cal[299], cal[150]],
        }
    )
    src = FakePITSource(prices, cal, table)
    mem = build_membership(src, "pit", ["AAA", "BBB", "CCC"], prices)
    assert mem.kind == KIND_PIT and not mem.survivorship_bias and not src.capabilities.survivorship_bias
    assert mem.members_on(cal[10]) == ["AAA", "CCC"]  # BBB 는 아직 편입 전
    assert mem.members_on(cal[120]) == ["BBB", "CCC"]
    assert mem.members_on(cal[250]) == ["AAA", "BBB"]  # CCC 는 편출 후 사라짐
    lines = write_instruments(mem, tmp_path).read_text().strip().splitlines()
    assert len(lines) == 4 and lines[0].split("\t")[0] == "AAA"


def test_membership_clipped_to_available_data(synth):
    prices, cal = synth
    prices = prices[~((prices["symbol"] == "CCC") & (prices["date"] < cal[100]))]
    mem = build_membership(get_source("yahoo_qlib"), "u", ["AAA", "CCC", "NODATA"], prices)
    t = mem.table.set_index("symbol")
    assert t.loc["CCC", "start"] == cal[100] and "NODATA" not in t.index
    assert mem.members_on(cal[10]) == ["AAA"]


# ---------- 펀더멘털 공시일 지연 ----------
def test_fundamentals_visible_only_after_report_date():
    cal = list(pd.bdate_range("2021-01-01", periods=120))
    f = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "period_end": pd.to_datetime(["2020-12-31", "2021-03-31"]),
            "report_date": [cal[20], cal[80]],
            "eps": [1.0, 2.0],
        }
    )
    p = align_to_report_date(f, cal, ["eps"], lag_days=1).set_index("date")["eps"]
    assert p.loc[: cal[20]].isna().all()  # 공시 당일까지는 보이지 않음
    assert p.loc[cal[21]] == 1.0 and p.loc[cal[80]] == 1.0 and p.loc[cal[81]] == 2.0
    p0 = align_to_report_date(f, cal, ["eps"], lag_days=0).set_index("date")["eps"]
    assert np.isnan(p0.loc[cal[19]]) and p0.loc[cal[20]] == 1.0
    with pytest.raises(ValueError):
        align_to_report_date(f.drop(columns="report_date"), cal, ["eps"])


# ---------- 해시 / 메타 ----------
def test_hash_is_deterministic_and_content_sensitive(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"123")
    (tmp_path / "y.txt").write_text("abc")
    h1 = hash_dir(tmp_path)
    assert h1 == hash_dir(tmp_path)
    (tmp_path / "research_meta.json").write_text("{}")  # 메타 파일은 해시에서 제외
    assert h1 == hash_dir(tmp_path)
    (tmp_path / "y.txt").write_text("abd")
    assert h1 != hash_dir(tmp_path)


def test_missing_meta_defaults_to_biased(tmp_path):
    assert load_data_flags(tmp_path)["flags"]["survivorship_bias"] is True
    (tmp_path / "research_meta.json").write_text(json.dumps({"flags": {"survivorship_bias": False}}))
    assert load_data_flags(tmp_path)["flags"]["survivorship_bias"] is False


def test_validate_prices_rejects_missing_columns(synth):
    prices, _ = synth
    with pytest.raises(ValueError):
        validate_prices(prices.drop(columns="adjclose"))


# ---------- Yahoo 분할 소급 반영 복원 ----------
def test_yahoo_split_unadjust_recovers_traded_prices():
    from research.data.sources.yahoo_qlib import unadjust_splits

    cal = pd.bdate_range("2020-01-01", periods=10)
    true_close = np.array([400.0, 404, 408, 412, 416, 105, 106, 107, 10.8, 10.9])  # 5일째 4:1, 8일째 10:1
    splits = np.array([0, 0, 0, 0, 0, 4.0, 0, 0, 10.0, 0])
    future = np.array([40.0] * 5 + [10.0] * 3 + [1.0] * 2)
    yahoo = pd.DataFrame(
        {"date": cal, "open": true_close / future, "high": true_close / future, "low": true_close / future,
         "close": true_close / future, "volume": 1000.0 * future, "adjclose": true_close / future, "splits": splits}
    )
    out = unadjust_splits(yahoo)
    np.testing.assert_allclose(out["close"].values, true_close)
    np.testing.assert_allclose(out["volume"].values, 1000.0)
    # 실제 체결가는 미래 분할과 무관해야 한다: 분할 이전까지만 받은 스냅샷(=당시 Yahoo 값)과 같아야 함
    early = yahoo.iloc[:5].assign(open=true_close[:5], high=true_close[:5], low=true_close[:5], close=true_close[:5],
                                  volume=1000.0, splits=0.0)
    np.testing.assert_allclose(unadjust_splits(early)["close"].values, out["close"].values[:5])
    # splits 컬럼이 없으면 그대로
    pd.testing.assert_frame_equal(unadjust_splits(yahoo.drop(columns="splits")), yahoo.drop(columns="splits"))
