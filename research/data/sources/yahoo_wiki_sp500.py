"""Yahoo 시세 + 위키피디아 S&P 500 편입/편출 이력으로 만든 '근사' PIT 유니버스.

- 멤버십: 현재 구성종목 표에서 출발해 변경 이력을 최신 -> 과거 순으로 되감아 (symbol, start, end) 구간을 복원한다.
- 한계(리포트에 자동 표기):
  * 편출 종목 대부분은 Yahoo 시세가 없다 -> 멤버십은 PIT 에 가깝지만 '시세가 있는 멤버'는 생존 종목 쪽으로 치우친다.
    따라서 survivorship_bias 는 여전히 True 이고, 연도별 커버리지(시세 보유 멤버 / 전체 멤버)를 수치로 보고한다.
  * 티커 변경(예: FB -> META)은 추적하지 못한다. 되감기 중 앞뒤가 맞지 않는 이벤트는 anomalies 로 기록한다.
  * 위키피디아는 공식 소스가 아니며 편집으로 내용이 바뀐다 -> 표 스냅샷을 raw_meta 에 저장하고 해시한다.
  * 섹터는 '현재' GICS 섹터다(PIT 아님). 편출 종목은 Unknown.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from .base import SourceCapabilities, register_source
from .yahoo_qlib import YahooQlibSource, make_collector_class

URL_CURRENT = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
URL_CHANGES = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}
FAR_PAST, FAR_FUTURE = pd.Timestamp("1900-01-01"), pd.Timestamp("2200-01-01")


def _norm(t) -> str:
    return str(t).strip().upper().replace(".", "-")


def rebuild_intervals(current: Sequence[str], changes: pd.DataFrame, start: str):
    """changes: columns [date, added, removed] (티커, 없으면 NaN). `start` 이후 이벤트만 되감는다.
    반환: (intervals DataFrame[symbol,start,end], anomalies list)"""
    members = {_norm(s) for s in current}
    open_end = {s: FAR_FUTURE for s in members}
    rows, anomalies = [], []
    ch = changes[changes["date"] >= pd.Timestamp(start)].sort_values("date", ascending=False, kind="stable")
    for d, a, r in zip(ch["date"], ch["added"], ch["removed"]):
        if isinstance(a, str) and a.strip():
            a = _norm(a)
            if a in members:
                rows.append((a, d, open_end.pop(a)))
                members.discard(a)
            else:
                anomalies.append({"date": f"{d:%Y-%m-%d}", "ticker": a, "issue": "added_but_not_member_afterwards"})
        if isinstance(r, str) and r.strip():
            r = _norm(r)
            if r not in members:
                members.add(r)
                open_end[r] = d - pd.Timedelta(days=1)
            else:
                anomalies.append({"date": f"{d:%Y-%m-%d}", "ticker": r, "issue": "removed_but_still_member_afterwards"})
    rows += [(s, FAR_PAST, open_end[s]) for s in sorted(members)]
    table = pd.DataFrame(rows, columns=["symbol", "start", "end"])
    table = table[table["start"] <= table["end"]].sort_values(["symbol", "start"]).reset_index(drop=True)
    return table, anomalies


@register_source("yahoo_wiki_sp500")
class YahooWikiSP500Source(YahooQlibSource):
    name = "yahoo_wiki_sp500"
    capabilities = SourceCapabilities(
        includes_delisted=False,
        pit_membership=False,  # 근사치이므로 PIT 로 인정하지 않는다
        pit_fundamentals=False,
        notes=YahooQlibSource.capabilities.notes
        + (
            "S&P 500 멤버십은 위키피디아 변경 이력으로 복원한 근사치(티커 변경 미추적, 비공식 소스)",
            "편출 종목 다수의 시세가 없어 '시세 보유 멤버'는 생존 종목 쪽으로 편향",
            "섹터는 현재 GICS 기준(PIT 아님)",
        ),
    )
    retry = 2

    # ---- 메타 스냅샷 ----
    def snapshot_meta(self, meta_dir) -> None:
        import requests

        meta_dir = Path(meta_dir)
        meta_dir.mkdir(parents=True, exist_ok=True)
        cur = pd.read_html(io.StringIO(requests.get(URL_CURRENT, headers=UA, timeout=60).text))[0]
        cur = cur.rename(columns={"Symbol": "symbol", "GICS Sector": "sector", "Date added": "date_added"})
        cur[["symbol", "Security", "sector", "date_added"]].to_csv(meta_dir / "sp500_current.csv", index=False)
        tabs = pd.read_html(io.StringIO(requests.get(URL_CHANGES, headers=UA, timeout=60).text))
        ch = max(tabs, key=len)
        ch.columns = ["_".join(dict.fromkeys(map(str, c))) for c in ch.columns]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(ch["Effective Date"], errors="coerce"),
                "added": ch["Added_Ticker"],
                "removed": ch["Removed_Ticker"],
            }
        ).dropna(subset=["date"])
        out.to_csv(meta_dir / "sp500_changes.csv", index=False)

    def _load_meta(self, meta_dir):
        meta_dir = Path(meta_dir)
        cur = pd.read_csv(meta_dir / "sp500_current.csv", keep_default_na=False)
        ch = pd.read_csv(meta_dir / "sp500_changes.csv", parse_dates=["date"], keep_default_na=False)
        return cur, ch

    def membership_table(self, meta_dir, start: str):
        cur, ch = self._load_meta(meta_dir)
        return rebuild_intervals(cur["symbol"], ch, start)

    def list_symbols(self, start: str, end: str, meta_dir) -> List[str]:
        table, anomalies = self.membership_table(meta_dir, start)
        (Path(meta_dir) / "membership_anomalies.json").write_text(json.dumps(anomalies, indent=1), encoding="utf-8")
        t = table[(table["end"] >= pd.Timestamp(start)) & (table["start"] < pd.Timestamp(end))]
        return sorted(t["symbol"].unique())

    def sector_map(self, meta_dir) -> pd.Series:
        cur, _ = self._load_meta(meta_dir)
        return pd.Series(cur["sector"].values, index=cur["symbol"].map(_norm)).rename("sector")

    # ---- DataSource ----
    def fetch_membership(self, universe, symbols, prices) -> pd.DataFrame:
        table, _ = self.membership_table(self._meta_dir, self._start)
        return table

    def bind(self, meta_dir, start: str) -> None:
        self._meta_dir, self._start = meta_dir, start

    def _collector_class(self, symbols):
        cls = make_collector_class(symbols)
        cls.retry = self.retry  # 편출 종목 실패가 많아 재시도를 줄인다
        return cls
