"""데이터 품질 리포트(markdown + json) 자동 생성. 모든 수치는 입력 데이터에서 직접 계산한다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .cleaning import FLAG_COLUMNS

FIELDS = ["open", "high", "low", "close", "adjclose", "volume"]


def _md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    if df.empty:
        return "_없음_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            elif isinstance(v, pd.Timestamp):
                cells.append(f"{v:%Y-%m-%d}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def membership_counts(table: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.Series:
    """각 거래일의 멤버 수(구간 테이블 -> 일별 카운트)."""
    s = cal.searchsorted(pd.DatetimeIndex(table["start"]), side="left")
    e = cal.searchsorted(pd.DatetimeIndex(table["end"]), side="right")
    delta = pd.Series(0, index=range(len(cal) + 1))
    delta = delta.add(pd.Series(s).value_counts(), fill_value=0).sub(pd.Series(e).value_counts(), fill_value=0)
    return pd.Series(delta.cumsum().iloc[: len(cal)].values, index=cal).astype(int)


def compute_quality(cleaned, events, stats, fetch, probes, membership, calendar, manifest) -> dict:
    cal = pd.DatetimeIndex(calendar)
    valid = cleaned.dropna(subset=["close"])

    per_day = valid.groupby("date")["symbol"].nunique().reindex(cal, fill_value=0)
    by_year = per_day.groupby(per_day.index.year).agg(["min", "max", "mean"]).round(1)
    by_year.index.name = "year"

    missing_by_field = cleaned[FIELDS].isna().mean().rename("missing_rate").rename_axis("field").reset_index()
    missing_by_symbol = (
        cleaned.groupby("symbol")["close"]
        .apply(lambda s: s.isna().mean())
        .rename("close_missing_rate")
        .sort_values(ascending=False)
        .reset_index()
    )

    span = valid.groupby("symbol")["date"].agg(first="min", last="max", n_days="count").reset_index()
    cal_end = cal.max()
    ended_early = span[span["last"] < cal_end - pd.Timedelta(days=10)]
    started_late = span[span["first"] > cal.min() + pd.Timedelta(days=10)]

    n_all = membership_counts(membership.unclipped, cal)
    n_data = membership_counts(membership.table, cal)
    cov = pd.DataFrame({"members": n_all, "members_with_data": n_data})
    coverage = cov.groupby(cov.index.year).mean().round(1)
    coverage["coverage"] = (coverage["members_with_data"] / coverage["members"]).round(4)
    coverage.index.name = "year"

    tradable_by_year = (
        cleaned.groupby(cleaned["date"].dt.year)["tradable"].mean().rename("tradable_fraction").rename_axis("year")
    )

    return {
        "flags": manifest["flags"],
        "source": manifest["source"],
        "universe": manifest["universe"],
        "data_hash": manifest["data_hash"],
        "period": {"start": f"{cal.min():%Y-%m-%d}", "end": f"{cal.max():%Y-%m-%d}", "trading_days": len(cal)},
        "symbols": {
            "requested": len(fetch.requested),
            "obtained": int(valid["symbol"].nunique()),
            "failed": fetch.failed,
        },
        "delisted_probes": probes,
        "cleaning_stats": stats,
        "tables": {
            "count_by_year": by_year.reset_index(),
            "universe_coverage": coverage.reset_index(),
            "missing_by_field": missing_by_field,
            "missing_by_symbol_top10": missing_by_symbol.head(10),
            "started_late": started_late,
            "ended_early": ended_early,
            "tradable_by_year": tradable_by_year.reset_index(),
            **{
                f"events_{f}": events.loc[events["flag"] == f, ["symbol", "date", "close", "adjclose", "adj_return"]]
                for f in FLAG_COLUMNS
            },
        },
    }


def write_report(q: dict, report_dir, max_rows: int = 60) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    t, fl = q["tables"], q["flags"]
    L = [f"# 데이터 품질 리포트 — {q['universe']['name']}", ""]
    if fl["survivorship_bias"]:
        L += [
            "> **⚠ 생존편향 존재 (survivorship_bias = true)**",
            f"> 소스 `{q['source']['name']}` / 유니버스 종류 `{q['universe']['kind']}`. "
            "이 데이터로 만든 모든 성과 수치는 상향 편향되어 있다.",
            "",
        ]
    L += [
        "## 1. 요약",
        f"- 기간: {q['period']['start']} ~ {q['period']['end']} ({q['period']['trading_days']} 거래일)",
        f"- 종목: 요청 {q['symbols']['requested']} / 수집 {q['symbols']['obtained']} / 실패 {len(q['symbols']['failed'])}"
        + (f" ({', '.join(q['symbols']['failed'][:40])}{' …' if len(q['symbols']['failed']) > 40 else ''})" if q["symbols"]["failed"] else ""),
        f"- PIT 멤버십: {fl['pit_membership']} · 상장폐지 종목 포함: {fl['includes_delisted']}",
        f"- 데이터 해시(raw): `{q['data_hash']['raw']}`",
        f"- 데이터 해시(qlib): `{q['data_hash']['qlib']}`",
        "",
        "## 2. 상장폐지 종목 포함 여부",
        "소스가 상장폐지 종목을 주는지 실제로 요청해 확인한 결과(유니버스에는 미포함):",
        "",
        _md_table(pd.DataFrame(q["delisted_probes"])),
        "## 3. 종목 수 추이 (연도별 일일 유효 종목 수)",
        _md_table(t["count_by_year"], "{:.1f}"),
        "### 유니버스 커버리지 (멤버십상 종목 수 대비 시세가 있는 종목 수, 연평균)",
        "커버리지가 1 보다 낮은 만큼이 '멤버였지만 시세가 없는 종목'이며 생존편향의 직접적인 크기다.",
        "",
        _md_table(t["universe_coverage"]),
        "### 기간 중간에 시작한 종목 (상장 등)",
        _md_table(t["started_late"].head(max_rows)),
        "### 캘린더 종료 전에 끝난 종목 (상장폐지/수집 중단 후보)",
        _md_table(t["ended_early"].head(max_rows)),
        "## 4. 결측률 (종목별 생존 구간 내, 캘린더 대비)",
        _md_table(t["missing_by_field"], "{:.5f}"),
        "### 종가 결측률 상위 10 종목",
        _md_table(t["missing_by_symbol_top10"], "{:.5f}"),
        "## 5. 정제 통계",
        _md_table(pd.DataFrame([q["cleaning_stats"]]).T.reset_index().set_axis(["항목", "값"], axis=1)),
        "## 6. 이상 수익률 목록",
    ]
    names = {
        "extreme_return": "이상 수익률 (|수정 일수익률| 임계 초과)",
        "suspected_split": "분할 미반영 의심",
        "invalid_price": "무효 가격",
        "halted": "거래정지/빠진 거래일",
    }
    for f, title in names.items():
        df = t[f"events_{f}"]
        L += [f"### {title} — {len(df)}건" + (f" (상위 {max_rows}건 표시)" if len(df) > max_rows else ""), ""]
        L += [_md_table(df.head(max_rows))]
    L += ["## 7. 유동성 필터 통과 비율 (연도별)", _md_table(t["tradable_by_year"]), "## 8. 소스 한계", ""]
    L += [f"- {n}" for n in q["source"]["notes"]]
    path = report_dir / "data_quality.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")

    js = {k: v for k, v in q.items() if k != "tables"}
    js["tables"] = {k: json.loads(v.to_json(orient="records", date_format="iso")) for k, v in t.items()}
    (report_dir / "data_quality.json").write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
