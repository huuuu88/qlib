"""qlib 변환 왕복 테스트.

1) 합성 데이터: 정제 -> upstream normalize -> dump_bin -> qlib API 로 읽어 원본과 대조, PIT instruments 확인
2) 실제 스토어(research/data/store/smoke50)가 있으면 manifest 해시 재계산 + end-to-end 검증
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from research.data.cleaning import CleaningConfig, clean_prices
from research.data.manifest import REPO_ROOT, hash_dir, load_data_flags
from research.data.qlib_dump import dump_bin, normalize_to_csv
from research.data.universe import build_membership, write_instruments
from research.data.verify import verify_qlib

from .conftest import FakePITSource, make_prices

STORE = REPO_ROOT / "research" / "data" / "store" / "smoke50"


def test_synthetic_roundtrip_with_pit_universe(tmp_path):
    prices, cal = make_prices(symbols=("AAA", "BBB", "CCC"), periods=200)
    # BBB 에 2:1 분할(정상 반영)을 넣어 factor 경로를 검증
    m = (prices["symbol"] == "BBB") & (prices["date"] >= cal[100])
    prices.loc[m, ["open", "high", "low", "close"]] /= 2
    prices.loc[m, "volume"] *= 2
    cleaned, _, _ = clean_prices(prices, cal, CleaningConfig())
    table = pd.DataFrame(
        {"symbol": ["AAA", "BBB", "CCC"], "start": [cal[0], cal[60], cal[0]], "end": [cal[199], cal[199], cal[120]]}
    )
    mem = build_membership(FakePITSource(prices, cal, table), "pit", ["AAA", "BBB", "CCC"], cleaned)

    assert normalize_to_csv(cleaned, cal, tmp_path / "norm") == 3
    dump_bin(tmp_path / "norm", tmp_path / "qlib", max_workers=1)
    write_instruments(mem, tmp_path / "qlib")

    res = verify_qlib(tmp_path / "qlib", cleaned, "pit")
    # PIT 유니버스라 검증 대상 종목 수는 3
    assert res["passed"], res

    from qlib.data import D

    def members(d):
        return sorted(D.list_instruments(D.instruments("pit"), start_time=d, end_time=d, as_list=True))

    assert members(cal[10]) == ["AAA", "CCC"]
    assert members(cal[150]) == ["AAA", "BBB"]


@pytest.mark.skipif(not (STORE / "manifest.json").exists(), reason="smoke50 store not built")
def test_smoke50_store_end_to_end():
    manifest = json.loads((STORE / "manifest.json").read_text(encoding="utf-8"))
    assert hash_dir(STORE / "qlib") == manifest["data_hash"]["qlib"]
    assert hash_dir(STORE / "raw") == manifest["data_hash"]["raw"]
    flags = load_data_flags(STORE / "qlib")["flags"]
    assert flags["survivorship_bias"] is True  # Yahoo 소스는 반드시 플래그가 붙어야 한다
    cleaned = pd.read_pickle(STORE / "cleaned.pkl")
    res = verify_qlib(STORE / "qlib", cleaned, manifest["universe"]["name"])
    assert res["passed"], res
    assert res["instruments"] == manifest["counts"]["normalized"]
