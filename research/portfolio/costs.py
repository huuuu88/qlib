"""거래비용 모델.

한 번의 거래(편도)에 대한 비용 = |거래금액| x (수수료 + 반스프레드 + 슬리피지 + 시장충격)

    시장충격(bps) = impact_coef x 일변동성 x sqrt(거래금액 / ADV)        (square-root 모델)

- 모든 값은 가정이다. 실제 체결 데이터로 보정한 값이 아니므로 반드시 민감도 분석과 함께 본다.
- ADV, 변동성은 체결일 전날까지의 값만 쓴다.
- 공매도 포지션에는 대차비용(연율)을 일할 계산해 붙인다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.5  # 수수료 (편도)
    half_spread_bps: float = 1.5  # 호가 스프레드의 절반
    slippage_bps: float = 1.0  # 시가 단일가 대비 체결가 이탈
    impact_coef: float = 0.1  # square-root 충격 계수 (변동성 배수)
    borrow_bps_per_year: float = 50.0  # 공매도 대차비용
    max_participation: float = 0.05  # 하루 주문금액 / ADV 상한. 초과분은 체결하지 않는다

    def scaled(self, mult: float) -> "CostModel":
        """비용 파라미터 전체를 mult 배 (참여율 상한은 그대로)."""
        return replace(
            self,
            commission_bps=self.commission_bps * mult,
            half_spread_bps=self.half_spread_bps * mult,
            slippage_bps=self.slippage_bps * mult,
            impact_coef=self.impact_coef * mult,
            borrow_bps_per_year=self.borrow_bps_per_year * mult,
        )

    @property
    def linear_rate(self) -> float:
        return (self.commission_bps + self.half_spread_bps + self.slippage_bps) * 1e-4

    def trade_cost(self, trade_dollar: np.ndarray, adv_dollar: np.ndarray, daily_vol: np.ndarray) -> dict:
        """종목별 거래금액(부호 무관) -> 비용 금액(구성요소별 합계)."""
        td = np.abs(np.nan_to_num(trade_dollar))
        adv = np.where(np.nan_to_num(adv_dollar) > 0, adv_dollar, np.nan)
        part = np.nan_to_num(td / adv, nan=0.0)
        impact_rate = self.impact_coef * np.nan_to_num(daily_vol, nan=0.02) * np.sqrt(part)
        return {
            "commission": float(td.sum() * self.commission_bps * 1e-4),
            "spread": float(td.sum() * self.half_spread_bps * 1e-4),
            "slippage": float(td.sum() * self.slippage_bps * 1e-4),
            "impact": float((td * impact_rate).sum()),
        }


ZERO_COST = CostModel(0.0, 0.0, 0.0, 0.0, 0.0, max_participation=np.inf)
