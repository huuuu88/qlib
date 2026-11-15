"""데이터 핸들러. upstream Alpha158/Alpha360 의 피처 정의를 그대로 가져다 쓴다(코드 수정 없음).

- Yahoo 에는 VWAP 이 없으므로 $vwap 피처를 뺀다(Alpha158: 158 -> 157개, Alpha360: 360 -> 300개).
- 라벨은 LabelSpec 으로만 지정한다.
- $tradable 을 meta 그룹으로 불러와 TradableFilter 로 거래 불가 행을 제거한다.
- 전처리는 횡단면 변환만 허용한다(assert_cross_sectional_only).
"""

from __future__ import annotations

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from qlib.data.dataset.handler import DataHandlerLP

from .labels import LabelSpec
from .processors import assert_cross_sectional_only

_TRADABLE = {"class": "TradableFilter", "module_path": "research.pipelines.processors"}
_LEARN = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
TREE_INFER = [_TRADABLE]
# 선형 모델용: 피처를 일자별 순위로 바꾸고 결측을 0(= 횡단면 중앙)으로 채운다
LINEAR_INFER = [
    _TRADABLE,
    {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]


def alpha158_fields():
    conf = {"kbar": {}, "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW"]}, "rolling": {}}
    return Alpha158DL.get_feature_config(conf)


def alpha360_fields():
    fields, names = Alpha360DL.get_feature_config()
    keep = [(f, n) for f, n in zip(fields, names) if "$vwap" not in f]
    return [f for f, _ in keep], [n for _, n in keep]


def factor_fields():
    from research.factors.library import factor_fields as ff

    return ff()


FEATURE_SETS = {"alpha158": alpha158_fields, "alpha360": alpha360_fields, "factors": factor_fields}


class USHandler(DataHandlerLP):
    def __init__(
        self,
        instruments,
        start_time,
        end_time,
        feature_set="alpha158",
        horizon=5,
        entry="next_open",
        infer_processors=None,
        learn_processors=None,
        **kwargs,
    ):
        infer_processors = TREE_INFER if infer_processors is None else infer_processors
        learn_processors = _LEARN if learn_processors is None else learn_processors
        assert_cross_sectional_only(infer_processors, learn_processors)
        self.label_spec = LabelSpec(horizon=horizon, entry=entry)
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": FEATURE_SETS[feature_set](),
                    "label": self.label_spec.qlib_label(),
                    "meta": (["$tradable"], ["tradable"]),
                },
                "freq": "day",
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            **kwargs,
        )


class USAlpha158(USHandler):
    """qrun yaml 에서 쓰기 위한 이름 고정 버전."""

    def __init__(self, *args, **kwargs):
        kwargs["feature_set"] = "alpha158"
        super().__init__(*args, **kwargs)
