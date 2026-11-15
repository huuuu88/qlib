"""Phase 1 데이터 빌드 CLI.

    python -m research.data.build --config research/configs/data_smoke50.yaml
    python -m research.data.build --config ... --skip-download   # 보존된 raw 스냅샷으로 재빌드

단계: 수집 -> 정제 -> PIT 유니버스 -> qlib 정규화/bin 덤프 -> 해시 manifest -> 검증 -> 품질 리포트
산출물: <store_dir>/<name>/{raw, raw_probes, normalized, qlib, cleaned.pkl, events.csv, manifest.json}
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from .cleaning import CleaningConfig, clean_prices
from .manifest import REPO_ROOT, build_manifest, write_manifest
from .qlib_dump import dump_bin, normalize_to_csv
from .quality import compute_quality, write_report
from .sources.base import get_source
from .universe import build_membership, write_instruments
from .verify import verify_qlib


def _read_symbols(path: Path) -> list:
    syms = [s.strip() for s in path.read_text(encoding="utf-8").splitlines()]
    syms = [s for s in syms if s and not s.startswith("#")]
    if len(set(syms)) != len(syms):
        raise ValueError("duplicate symbols in universe file")
    return syms


def run(config_path: str, skip_download: bool = False) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    store = (REPO_ROOT / cfg["store_dir"] / cfg["name"]).resolve()
    report_dir = (REPO_ROOT / cfg["report_dir"]).resolve()
    raw_dir, probe_dir, norm_dir, qlib_dir = (store / d for d in ("raw", "raw_probes", "normalized", "qlib"))
    meta_dir = store / "raw_meta"
    ucfg = cfg["universe"]
    probe_syms = list(ucfg.get("delisted_probes", []))
    benchmarks = list(cfg.get("benchmarks", []))

    source = get_source(cfg["source"], **cfg.get("download", {}))
    if ucfg.get("symbols_from_source"):
        # 유니버스 종목을 소스의 멤버십 이력에서 얻는다(편출 종목 포함). 메타 스냅샷은 raw_meta 에 보존.
        if not skip_download:
            source.snapshot_meta(meta_dir)
        source.bind(meta_dir, cfg["start"])
        universe_symbols = source.list_symbols(cfg["start"], cfg["end"], meta_dir)
        source.sector_map(meta_dir).to_csv(store / "sectors.csv")
    else:
        universe_symbols = _read_symbols(REPO_ROOT / ucfg["symbols_file"])
    symbols = sorted(set(universe_symbols) | set(benchmarks))

    # 1) 수집 (원본 스냅샷 보존)
    if skip_download:
        fetch = source.load_raw(symbols, raw_dir)
        probe_fetch = source.load_raw(probe_syms, probe_dir)
    else:
        for d in (raw_dir, probe_dir):
            if d.exists():
                shutil.rmtree(d)
        fetch = source.fetch_prices(symbols, cfg["start"], cfg["end"], raw_dir)
        probe_fetch = source.fetch_prices(probe_syms, cfg["start"], cfg["end"], probe_dir) if probe_syms else None
    if fetch.prices.empty:
        raise RuntimeError("no price data fetched")
    probes = []
    for s in probe_syms:
        got = probe_fetch is not None and s not in probe_fetch.failed
        rows = int((probe_fetch.prices["symbol"] == s).sum()) if got else 0
        probes.append({"symbol": s, "data_returned": bool(got), "rows": rows})

    calendar = source.fetch_calendar(cfg["start"], cfg["end"])
    (store / "calendar.txt").write_text("\n".join(f"{d:%Y-%m-%d}" for d in calendar), encoding="utf-8")

    # 2) 정제
    cleaned, events, stats = clean_prices(fetch.prices, calendar, CleaningConfig.from_dict(cfg.get("cleaning")))
    cleaned.to_pickle(store / "cleaned.pkl")
    events.to_csv(store / "events.csv", index=False)

    # 3) qlib 변환 + PIT 유니버스
    n_norm = normalize_to_csv(cleaned, calendar, norm_dir)
    dump_bin(norm_dir, qlib_dir)
    membership = build_membership(
        source, ucfg["name"], universe_symbols, cleaned[cleaned["symbol"].isin(universe_symbols)]
    )
    write_instruments(membership, qlib_dir)

    # 4) 해시/manifest
    counts = {
        "requested": len(symbols),
        "normalized": n_norm,
        "universe_symbols": len(universe_symbols),
        "universe_symbols_with_data": int(membership.table["symbol"].nunique()),
        "benchmarks": benchmarks,
        "failed": fetch.failed,
        "trading_days": len(calendar),
    }
    manifest = build_manifest(cfg, source, membership, raw_dir, qlib_dir, counts, meta_dir)
    write_manifest(manifest, store, report_dir)

    # 5) 검증 + 리포트
    verify = verify_qlib(qlib_dir, cleaned, ucfg["name"], membership.table["symbol"].nunique())
    (report_dir / "e2e_verify.json").write_text(json.dumps(verify, indent=2), encoding="utf-8")
    q = compute_quality(cleaned, events, stats, fetch, probes, membership, calendar, manifest)
    report = write_report(q, report_dir)

    summary = {"report": str(report), "verify": verify, "flags": manifest["flags"], "data_hash": manifest["data_hash"]}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not verify["passed"]:
        sys.exit("END-TO-END VERIFY FAILED")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    run(a.config, a.skip_download)
