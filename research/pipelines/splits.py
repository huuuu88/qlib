"""시간순 Walk-forward 분할 + Purge/Embargo.

t 일 라벨은 t+span 일 가격까지 쓰므로, 앞 구간의 마지막 날과 뒤 구간의 첫 날 사이에
span 거래일(+ 선택적 embargo)을 비워야 앞 구간 라벨이 뒤 구간과 겹치지 않는다.

    train ........| purge |  valid ....| purge |  test ....
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd

Seg = Tuple[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class Fold:
    k: int
    train: Seg
    valid: Seg
    test: Seg

    def segments(self) -> dict:
        return {"train": self.train, "valid": self.valid, "test": self.test}


def walk_forward_folds(
    calendar: Sequence[pd.Timestamp],
    label_span: int,
    train_days: int = 1008,
    valid_days: int = 252,
    test_days: int = 252,
    step_days: int = 252,
    embargo_days: int = 0,
    start_index: int = 0,
    expanding: bool = False,
    usable_end_index: int = None,
) -> List[Fold]:
    """재학습 주기 = step_days. test 구간은 서로 겹치지 않고 이어진다.
    usable_end_index: 라벨이 확정되는 마지막 신호일 인덱스(기본: len(cal)-1-label_span)."""
    cal = pd.DatetimeIndex(calendar)
    gap = label_span + embargo_days
    last = (len(cal) - 1 - label_span) if usable_end_index is None else usable_end_index
    folds, k = [], 0
    while True:
        tr0 = start_index if expanding else start_index + k * step_days
        tr1 = start_index + k * step_days + train_days - 1
        va0 = tr1 + gap + 1
        va1 = va0 + valid_days - 1
        te0 = va1 + gap + 1
        te1 = min(te0 + test_days - 1, last)
        if te0 > last or te1 - te0 + 1 < min(test_days, 20):
            break
        folds.append(Fold(k, (cal[tr0], cal[tr1]), (cal[va0], cal[va1]), (cal[te0], cal[te1])))
        k += 1
    return folds


def assert_no_label_overlap(folds: List[Fold], calendar: Sequence[pd.Timestamp], label_span: int) -> None:
    """앞 구간 마지막 라벨의 종료일이 뒤 구간 시작일보다 앞서는지 검사."""
    cal = pd.DatetimeIndex(calendar)
    for f in folds:
        for a, b in ((f.train, f.valid), (f.valid, f.test)):
            end_of_label = cal.get_loc(a[1]) + label_span
            if end_of_label >= cal.get_loc(b[0]):
                raise AssertionError(f"fold {f.k}: label of {a[1].date()} overlaps segment starting {b[0].date()}")
        if not (f.train[0] <= f.train[1] < f.valid[0] <= f.valid[1] < f.test[0] <= f.test[1]):
            raise AssertionError(f"fold {f.k}: segments are not in chronological order")
    for p, n in zip(folds, folds[1:]):
        if n.test[0] <= p.test[1]:
            raise AssertionError("test segments overlap")
