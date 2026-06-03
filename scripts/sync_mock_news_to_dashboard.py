#!/usr/bin/env python3
"""Sync articles_with_sentiment.csv → dashboard mock-news-data.json for offline fallback."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT / "data/raw/sentiment/articles_with_sentiment.csv"
OUT_PATH = PROJECT / "dashboard/src/lib/mock-news-data.json"

PHARMA_TERMS = ("pharma", "pharmaceutical", "medicine", "drug", "biotech", "vaccine", "api")
TRADE_TERMS = ("trade", "export", "import", "shipment", "tariff", "market access", "customs")


def _format_date(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    if re.match(r"^\d{8}T\d{6}Z$", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.isna(ts):
        return s[:10]
    return ts.strftime("%Y-%m-%d")


def _is_relevant(title: str, domain: str, url: str) -> bool:
    text = " ".join([title or "", domain or "", url or ""]).lower()
    return any(k in text for k in PHARMA_TERMS + TRADE_TERMS)


def sync_mock_news(limit: int = 50) -> int:
    if not CSV_PATH.exists():
        print(f"Missing {CSV_PATH}", file=sys.stderr)
        return 0

    df = pd.read_csv(CSV_PATH)
    required = ["country_1_iso3", "country_2_iso3", "title", "url", "date"]
    if any(c not in df.columns for c in required):
        print("CSV missing required columns", file=sys.stderr)
        return 0

    filtered = df[
        (df["country_1_iso3"] == "IND") | (df["country_2_iso3"] == "IND")
    ].copy()
    filtered = filtered[
        filtered.apply(
            lambda r: _is_relevant(
                str(r.get("title", "")),
                str(r.get("domain", "")),
                str(r.get("url", "")),
            ),
            axis=1,
        )
    ]
    filtered["_date_dt"] = pd.to_datetime(filtered["date"], errors="coerce", utc=True)
    filtered = filtered.sort_values("_date_dt", ascending=False, na_position="last")
    filtered = filtered.drop_duplicates(subset=["url"], keep="first").head(limit)

    articles = []
    for idx, row in filtered.iterrows():
        url = str(row.get("url", "")).strip()
        if not url or url == "nan":
            continue
        if not url.startswith("http"):
            url = f"https://{url}"

        partner = None
        if pd.notna(row.get("country_2_iso3")) and row["country_2_iso3"] != "IND":
            partner = str(row["country_2_iso3"])
        elif pd.notna(row.get("country_1_iso3")) and row["country_1_iso3"] != "IND":
            partner = str(row["country_1_iso3"])

        title = str(row.get("title", ""))[:200]
        domain = str(row.get("domain", "Unknown")) if pd.notna(row.get("domain")) else "Unknown"
        sentiment = float(row["sentiment_score"]) if pd.notna(row.get("sentiment_score")) else 0.0
        relevance = float(row["trade_relevance"]) if pd.notna(row.get("trade_relevance")) else 0.8

        articles.append(
            {
                "id": f"news_{idx}",
                "title": title,
                "snippet": (title[:150] + "...") if len(title) > 150 else title,
                "source": domain,
                "url": url,
                "date": _format_date(row.get("date")),
                "sentiment": round(sentiment, 4),
                "relevance_score": round(relevance, 2),
                "country_code": partner,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_from": str(CSV_PATH.relative_to(PROJECT)),
        "article_count": len(articles),
        "articles": articles,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(articles)} articles → {OUT_PATH}")
    return len(articles)


if __name__ == "__main__":
    n = sync_mock_news()
    sys.exit(0 if n > 0 else 1)
