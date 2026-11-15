"""데이터 버전(해시) + 재현성 메타데이터.

실험은 manifest 의 `data_hash.qlib` 를 데이터 버전으로 기록한다.
`research_meta.json` 은 qlib 데이터 폴더 안에 함께 저장되어, 하위 단계가 편향 플래그를 자동으로 읽는다.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
META_FILE = "research_meta.json"


def hash_dir(path, exclude=(META_FILE,)) -> str:
    """폴더 내 모든 파일의 (상대경로, 내용) sha256. 파일 순서·OS 에 무관."""
    path = Path(path)
    h = hashlib.sha256()
    for fp in sorted(p for p in path.rglob("*") if p.is_file() and p.name not in exclude):
        h.update(fp.relative_to(path).as_posix().encode())
        h.update(b"\0")
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, encoding="utf-8"
        ).stdout.strip()
    except Exception as e:  # git 없음 등
        return f"unavailable: {e}"


def _versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("pyqlib", "pandas", "numpy", "yahooquery", "lightgbm"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def build_manifest(config: dict, source, membership, raw_dir, qlib_dir, counts: dict) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "name": config["name"],
        "config": config,
        "source": {"name": source.name, **source.capabilities.to_dict()},
        "universe": {"name": membership.name, "kind": membership.kind},
        "flags": {
            "survivorship_bias": bool(source.capabilities.survivorship_bias or membership.survivorship_bias),
            "pit_membership": membership.kind == "point_in_time",
            "includes_delisted": source.capabilities.includes_delisted,
        },
        "data_hash": {"raw": hash_dir(raw_dir), "qlib": hash_dir(qlib_dir)},
        "git": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))},
        "versions": _versions(),
        "counts": counts,
    }


def write_manifest(manifest: dict, *dirs) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {"flags": manifest["flags"], "data_hash": manifest["data_hash"], "source": manifest["source"]["name"]}
    # qlib 폴더는 dirs[0] 의 하위 `qlib`
    (Path(dirs[0]) / "qlib" / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_data_flags(qlib_dir) -> dict:
    """하위 단계(백테스트/리포트)에서 호출. 메타가 없으면 안전하지 않은 쪽으로 가정한다."""
    fp = Path(qlib_dir) / META_FILE
    if not fp.exists():
        return {"flags": {"survivorship_bias": True, "pit_membership": False, "includes_delisted": False}, "unknown": True}
    return json.loads(fp.read_text(encoding="utf-8"))
