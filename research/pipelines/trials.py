"""시도(trial) 기록. 평가해 본 모든 팩터/모델/하이퍼파라미터/포트폴리오 구성을 남긴다.

- 같은 config 를 다시 돌리는 것은 새 시도가 아니다(config 해시로 구분, runs 만 증가).
- 보고서의 '총 시도 횟수'와 Deflated Sharpe 의 N 은 이 파일의 고유 시도 수다.
- 파일은 git 에 커밋한다(research/reports/trials.jsonl) — 시도 이력 자체가 연구 결과의 일부다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from research.data.manifest import REPO_ROOT

TRIALS_FILE = REPO_ROOT / "research" / "reports" / "trials.jsonl"


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_trials(path: Path = TRIALS_FILE) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                out[t["id"]] = t
    return out


def record_trial(kind: str, name: str, config: dict, metrics: dict, path: Path = TRIALS_FILE) -> str:
    """kind: factor | model | portfolio"""
    from filelock import FileLock

    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):  # 여러 프로세스가 동시에 기록해도 유실되지 않게
        trials = load_trials(path)
        tid = f"{kind}:{config_hash({'name': name, **config})}"
        prev = trials.get(tid, {})
        trials[tid] = {
            "id": tid,
            "kind": kind,
            "name": name,
            "config": config,
            "metrics": metrics,
            "first_seen_utc": prev.get("first_seen_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
            "runs": prev.get("runs", 0) + 1,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(t, ensure_ascii=False, default=str) for t in trials.values()) + "\n", encoding="utf-8"
        )
    return tid


def trial_counts(path: Path = TRIALS_FILE) -> dict:
    trials = load_trials(path)
    by_kind = {}
    for t in trials.values():
        by_kind[t["kind"]] = by_kind.get(t["kind"], 0) + 1
    return {"total": len(trials), "by_kind": by_kind}
