"""커스텀 팩터 라이브러리 (qlib expression).

규칙
- 모든 식은 t 일 종가까지의 정보만 쓴다(Ref 의 인자는 항상 양수). 누수 테스트가 이를 강제한다.
- qlib 0.9.8 표현식 파서는 단항 마이너스를 지원하지 않으므로 부호 반전은 `0-(...)` 로 쓴다.
- 부호는 '값이 클수록 기대수익이 높다'는 가설 방향으로 맞춘다.
- 가격 수준이 아니라 비율만 쓴다(수정주가 수준에는 미래 배당/분할이 섞여 있음).
  거래대금은 $close*$volume 으로 수정·스케일에 불변이다.

가치/퀄리티 팩터는 공시일 기준(PIT) 펀더멘털이 필요하다. Yahoo 소스에는 없으므로 구현하지 않고
UNAVAILABLE 에 사유와 함께 기록한다(가짜 대용치를 만들지 않는다).
"""

from __future__ import annotations

_RET = "($close/Ref($close,1)-1)"

FACTORS = {
    # 모멘텀
    "mom_12_1": {"family": "momentum", "expr": "Ref($close,21)/Ref($close,252)-1", "desc": "12개월 수익률(최근 1개월 제외)"},
    "mom_6_1": {"family": "momentum", "expr": "Ref($close,21)/Ref($close,126)-1", "desc": "6개월 수익률(최근 1개월 제외)"},
    "high52w": {"family": "momentum", "expr": "$close/Max($high,252)", "desc": "52주 고가 대비 위치"},
    # 단기 반전
    "rev_5": {"family": "reversal", "expr": "0-($close/Ref($close,5)-1)", "desc": "5일 수익률의 반대"},
    "rev_21": {"family": "reversal", "expr": "0-($close/Ref($close,21)-1)", "desc": "21일 수익률의 반대"},
    # 저변동
    "lowvol_60": {"family": "low_vol", "expr": f"0-Std({_RET},60)", "desc": "60일 일수익률 표준편차의 반대"},
    "lowvol_252": {"family": "low_vol", "expr": f"0-Std({_RET},252)", "desc": "252일 일수익률 표준편차의 반대"},
    "maxret_21": {"family": "low_vol", "expr": f"0-Max({_RET},21)", "desc": "21일 최대 일수익률의 반대(복권 성향)"},
    # 유동성
    "illiq_21": {
        "family": "liquidity",
        "expr": f"Mean(Abs({_RET})/($close*$volume+1e-12),21)",
        "desc": "Amihud 비유동성(21일)",
    },
    "dollarvol_60": {"family": "liquidity", "expr": "0-Log(Mean($close*$volume,60)+1)", "desc": "60일 평균 거래대금의 반대"},
}

UNAVAILABLE = {
    "value (B/P, E/P 등)": "공시일 기준 펀더멘털 필요 — Yahoo 소스에 없음",
    "quality (ROE, 총이익률, 발생액 등)": "공시일 기준 펀더멘털 필요 — Yahoo 소스에 없음",
    "size (시가총액)": "PIT 발행주식수 필요 — Yahoo 소스에 없음(거래대금으로 대체하지 않고 유동성 팩터로 분류)",
}


def factor_fields():
    names = list(FACTORS)
    return [FACTORS[n]["expr"] for n in names], names
