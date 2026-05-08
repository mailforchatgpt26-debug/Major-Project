#!/usr/bin/env python3
"""Append Nov & Dec 2025 rows to pharma Comtrade CSVs using linear trend on Jan–Oct 2025.

Uses the csv module (not pandas) because rows with empty cifvalue (\",,\") misalign in pandas.

Usage:
  python scripts/fill_comtrade_2025_nov_dec.py
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
EXPORTS = PROJECT / "data/raw/comtrade/TradeData_exports_pharma_2011_2025.csv"
IMPORTS = PROJECT / "data/raw/comtrade/TradeData_imports_pharma_2011_2025.csv"

NEW_MONTHS = (11, 12)
TREND_MONTHS = range(1, 11)


def _parse_num(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _month_value(row: dict, *, is_export: bool) -> float | None:
    v = _parse_num(row.get("primaryValue", ""))
    if v is not None:
        return v
    if is_export:
        return _parse_num(row.get("fobvalue", ""))
    return _parse_num(row.get("cifvalue", ""))


def _predict_nov_dec(month_to_val: dict[int, float]) -> tuple[float, float]:
    pts = sorted((m, month_to_val[m]) for m in month_to_val if m in TREND_MONTHS)
    if len(pts) >= 2:
        m_arr = np.array([p[0] for p in pts], dtype=float)
        v_arr = np.array([p[1] for p in pts], dtype=float)
        a, b = np.polyfit(m_arr, v_arr, 1)
        return max(0.0, float(a * 11 + b)), max(0.0, float(a * 12 + b))
    if len(pts) == 1:
        v = float(pts[0][1])
        return v, v
    return 0.0, 0.0


def _format_money(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _load_comtrade_rows(path: Path) -> tuple[list[str], list[dict]]:
    """Some merged CSVs have a comment line before the real header."""
    raw = path.read_text(encoding="latin-1").splitlines()
    header_idx = next(
        (i for i, line in enumerate(raw) if line.startswith('"typeCode"') or line.startswith("typeCode")),
        None,
    )
    if header_idx is None:
        raise SystemExit(f"No Comtrade header (typeCode) in {path}")
    buf = io.StringIO("\n".join(raw[header_idx:]))
    reader = csv.DictReader(buf)
    fieldnames = reader.fieldnames
    if not fieldnames:
        raise SystemExit(f"No header: {path}")
    rows: list[dict] = []
    for r in reader:
        r.pop(None, None)  # extra trailing columns → key None
        rows.append(r)
    return list(fieldnames), rows


def process_file(path: Path, *, is_export: bool) -> int:
    fieldnames, rows = _load_comtrade_rows(path)

    # Drop existing Nov/Dec 2025
    def is_2025_nd(r: dict) -> bool:
        try:
            y = int(r.get("refYear", 0))
            m = int(r.get("refMonth", 0))
            return y == 2025 and m in NEW_MONTHS
        except ValueError:
            return False

    rows = [r for r in rows if not is_2025_nd(r)]

    # partner -> month -> value (Jan–Oct 2025 only)
    series: dict[str, dict[int, float]] = defaultdict(dict)
    templates: dict[str, dict] = {}
    best_m: dict[str, int] = defaultdict(int)

    for r in rows:
        try:
            y = int(r["refYear"])
            m = int(r["refMonth"])
        except (KeyError, ValueError):
            continue
        if y != 2025:
            continue
        p = str(r.get("partnerCode", "")).strip()
        if not p:
            continue
        v = _month_value(r, is_export=is_export)
        if v is not None and m in TREND_MONTHS:
            series[p][m] = v
        if m <= 10 and m >= best_m[p]:
            best_m[p] = m
            templates[p] = dict(r)

    all_partners = set(templates.keys()) | set(series.keys())
    new_rows: list[dict] = []
    for partner in sorted(all_partners, key=lambda x: (len(x), x)):
        tpl = templates.get(partner)
        if not tpl:
            continue
        nov_v, dec_v = _predict_nov_dec(dict(series[partner]))
        for month, val in zip(NEW_MONTHS, (nov_v, dec_v)):
            row = dict(tpl)
            row["refPeriodId"] = f"2025{month:02d}01"
            row["refYear"] = str(2025)
            row["refMonth"] = str(month)
            row["period"] = f"2025{month:02d}"
            fv = _format_money(val)
            if is_export:
                row["cifvalue"] = ""
                row["fobvalue"] = fv
                row["primaryValue"] = fv
            else:
                row["cifvalue"] = fv
                row["fobvalue"] = ""
                row["primaryValue"] = fv
            row["legacyEstimationFlag"] = "4"
            row["isReported"] = "false"
            new_rows.append(row)

    if not new_rows:
        print(f"No new rows for {path.name}", file=sys.stderr)
        return 0

    rows.extend(new_rows)
    # Atomic write: never truncate the source file if writing fails mid-way.
    out_rows = [{k: r.get(k, "") for k in fieldnames} for r in rows]
    fd, tmp = tempfile.mkstemp(prefix="comtrade_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="latin-1") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(out_rows)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    print(f"{path.name}: added {len(new_rows)} rows ({len(all_partners)} partners × 2 months)")
    return len(new_rows)


def main() -> None:
    if not EXPORTS.exists() or not IMPORTS.exists():
        raise SystemExit("Missing export or import CSV under data/raw/comtrade/")
    n1 = process_file(EXPORTS, is_export=True)
    n2 = process_file(IMPORTS, is_export=False)
    print(f"Done. Total synthetic rows: {n1 + n2}")


if __name__ == "__main__":
    main()
