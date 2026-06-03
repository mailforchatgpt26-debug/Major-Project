#!/usr/bin/env python3
"""
Set Nov & Dec 2025 Comtrade rows so full-year 2025 matches per-country targets.

Exports: India → partner (FOB). Imports: partner → India (CIF).
Jan–Oct unchanged when possible; full-year scale if Jan–Oct alone exceeds target.
W00 Nov/Dec refreshed as partner sums.

Usage:
  PYTHONPATH=. python scripts/adjust_comtrade_2025_per_country.py
  PYTHONPATH=. python scripts/adjust_comtrade_2025_per_country.py --imports-only
  PYTHONPATH=. python scripts/adjust_comtrade_2025_per_country.py --run-preprocess
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXPORTS = PROJECT / "data/raw/comtrade/TradeData_exports_pharma_2011_2025.csv"
IMPORTS = PROJECT / "data/raw/comtrade/TradeData_imports_pharma_2011_2025.csv"

# FY2024 India pharma export anchors (USD M) — Pharmexcil/DGCIS decline baseline.
EXPORT_2024_TARGETS_MUSD: dict[str, float] = {
    "ARE": 632,
    "NLD": 715,
    "CHN": 593,
    "BEL": 486,
    "TUR": 298,
    "LKA": 258,
    "MEX": 312,
    "ZAF": 753,
    "THA": 210,
}

# Annual 2025 India pharma exports by partner (USD million, approximate).
EXPORT_TARGETS_MUSD: dict[str, float] = {
    "USA": 10515.11,
    "GBR": 913.97,
    "BRA": 778.49,
    "FRA": 720.43,
    "ZAF": 740,
    "CAN": 620.08,
    "DEU": 597.58,
    "AUS": 469.76,
    "RUS": 577.22,
    "NLD": 616,
    "ARE": 520,
    "BEL": 450,
    "CHN": 530,
    "SAU": 211.37,
    "MEX": 300,
    "ITA": 244.59,
    "ESP": 246.23,
    "JPN": 231.52,
    "POL": 203.57,
    "TUR": 250,
    "SGP": 160,
    "THA": 210,
    "VNM": 140,
    "IDN": 130,
    "LKA": 220,
    "NPL": 260,
    "BGD": 100,
    "MYS": 95,
    "NZL": 90,
    "ARG": 85,
    "DNK": 80,
    "SWE": 75,
    "FIN": 60,
    "CZE": 55,
    "HUN": 50,
    "ROU": 50,
    "GRC": 45,
    "SVN": 40,
    "JOR": 35,
    "OMN": 35,
    "DZA": 30,
    "GHA": 30,
    "NGA": 535.35,
    "ETH": 25,
    "UGA": 20,
    "TZA": 25,
    "MLT": 15,
    "LVA": 10,
    "HKG": 40,
    "DOM": 20,
}

# Annual 2025 pharma imports to India by partner (USD million, approximate).
IMPORT_TARGETS_MUSD: dict[str, float] = {
    "USA": 950,
    "GBR": 180,
    "CAN": 90,
    "FRA": 650,
    "NGA": 5,
    "ZAF": 25,
    "BRA": 40,
    "DEU": 1250,
    "AUS": 40,
    "RUS": 60,
    "NLD": 220,
    "ARE": 70,
    "BEL": 1100,
    "NPL": 2,
    "TZA": 1,
    "LKA": 3,
    "VNM": 50,
    "GHA": 1,
    "SAU": 15,
    "THA": 120,
    "MLT": 40,
    "LVA": 8,
    "MEX": 60,
    "ETH": 1,
    "UGA": 1,
    "JPN": 420,
    "MYS": 90,
    "POL": 80,
    "TUR": 50,
    "HUN": 180,
    "SVN": 90,
    "CHN": 900,
    "ESP": 220,
    "ITA": 500,
    "DOM": 1,
    "NZL": 10,
    "BGD": 8,
    "IDN": 70,
    "FIN": 30,
    "SGP": 300,
    "OMN": 5,
    "DZA": 2,
    "ROU": 40,
    "DNK": 350,
    "SWE": 80,
    "CZE": 60,
    "JOR": 5,
    "HKG": 50,
    "ARG": 15,
    "GRC": 40,
}

# Back-compat alias
TARGETS_MUSD = EXPORT_TARGETS_MUSD

NEW_MONTHS = (11, 12)
BASE_MONTHS = tuple(range(1, 11))
SKIP_PARTNERS = frozenset({"W00", "_X"})


def _parse_num(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _month_value(row: dict, *, is_export: bool) -> float:
    v = _parse_num(row.get("primaryValue", ""))
    if v > 0:
        return v
    if is_export:
        return _parse_num(row.get("fobvalue", ""))
    return _parse_num(row.get("cifvalue", ""))


def _set_month_value(row: dict, value: float, *, is_export: bool) -> None:
    fv = f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    row["primaryValue"] = fv
    if is_export:
        row["fobvalue"] = fv
        row["cifvalue"] = ""
    else:
        row["cifvalue"] = fv
        row["fobvalue"] = ""


def _load_comtrade_rows(path: Path) -> tuple[list[str], list[dict]]:
    raw = path.read_text(encoding="latin-1").splitlines()
    header_idx = next(
        (i for i, line in enumerate(raw) if line.startswith('"typeCode"') or line.startswith("typeCode")),
        None,
    )
    if header_idx is None:
        raise SystemExit(f"No Comtrade header in {path}")
    buf = io.StringIO("\n".join(raw[header_idx:]))
    reader = csv.DictReader(buf)
    fieldnames = list(reader.fieldnames or [])
    rows = []
    for r in reader:
        r.pop(None, None)
        rows.append(r)
    return fieldnames, rows


def _write_comtrade(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    out_rows = [{k: r.get(k, "") for k in fieldnames} for r in rows]
    fd, tmp = tempfile.mkstemp(prefix="comtrade_pctr_", suffix=".csv", dir=path.parent)
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


def _year_month(r: dict) -> tuple[int, int] | None:
    try:
        return int(r["refYear"]), int(r["refMonth"])
    except (KeyError, ValueError):
        return None


def _adjust_2024_exports(path: Path, targets_musd: dict[str, float]) -> None:
    """Scale all 2024 export months per partner to match FY2024 annual targets."""
    fieldnames, rows = _load_comtrade_rows(path)
    totals: dict[str, float] = {iso: 0.0 for iso in targets_musd}
    for r in rows:
        ym = _year_month(r)
        if not ym or ym[0] != 2024:
            continue
        partner = str(r.get("partnerISO", "")).strip()
        if partner in totals:
            totals[partner] += _month_value(r, is_export=True)

    for partner, target_musd in targets_musd.items():
        current = totals.get(partner, 0.0)
        if current <= 0:
            continue
        factor = (target_musd * 1e6) / current
        for i, r in enumerate(rows):
            ym = _year_month(r)
            if not ym or ym[0] != 2024:
                continue
            if str(r.get("partnerISO", "")).strip() != partner:
                continue
            _set_month_value(
                rows[i],
                _month_value(r, is_export=True) * factor,
                is_export=True,
            )

    _write_comtrade(path, fieldnames, rows)
    print("FY2024 export baselines applied (Pharmexcil decline anchors)\n")
    print(f"{'ISO':<6} {'FY2024 M':>10}")
    for partner in sorted(targets_musd, key=lambda p: -targets_musd[p]):
        print(f"{partner:<6} {targets_musd[partner]:>10.0f}")
    print()


def _adjust_file_per_country(
    path: Path,
    targets_musd: dict[str, float],
    *,
    is_export: bool,
    label: str,
) -> None:
    fieldnames, rows = _load_comtrade_rows(path)

    jan_oct: dict[str, float] = {iso: 0.0 for iso in targets_musd}
    nov_dec_by_partner: dict[str, list[tuple[int, float]]] = {iso: [] for iso in targets_musd}

    for i, r in enumerate(rows):
        ym = _year_month(r)
        if not ym or ym[0] != 2025:
            continue
        partner = str(r.get("partnerISO", "")).strip()
        if partner not in targets_musd:
            continue
        val = _month_value(r, is_export=is_export)
        if ym[1] in BASE_MONTHS:
            jan_oct[partner] += val
        elif ym[1] in NEW_MONTHS:
            nov_dec_by_partner[partner].append((i, val))

    full_year_scaled: list[str] = []
    for partner, target_musd in targets_musd.items():
        target_usd = target_musd * 1e6
        jo = jan_oct[partner]
        indices_vals = nov_dec_by_partner[partner]
        cur_nd = sum(v for _, v in indices_vals)
        annual = jo + cur_nd

        if annual <= 0:
            continue

        if jo >= target_usd:
            factor = target_usd / annual
            for i, r in enumerate(rows):
                ym = _year_month(r)
                if not ym or ym[0] != 2025:
                    continue
                if str(r.get("partnerISO", "")).strip() != partner:
                    continue
                _set_month_value(
                    rows[i],
                    _month_value(r, is_export=is_export) * factor,
                    is_export=is_export,
                )
            full_year_scaled.append(
                f"{partner} (Jan–Oct ${jo/1e6:.1f}M > ${target_musd}M target; all 2025 months ×{factor:.4f})"
            )
            continue

        need_nd = target_usd - jo
        if not indices_vals:
            continue
        if cur_nd <= 0:
            half = need_nd / 2.0
            for idx, _ in indices_vals:
                _set_month_value(rows[idx], half, is_export=is_export)
            continue
        scale = need_nd / cur_nd
        for idx, old in indices_vals:
            _set_month_value(rows[idx], old * scale, is_export=is_export)

    for month in NEW_MONTHS:
        partner_sum = 0.0
        w00_idx: int | None = None
        for i, r in enumerate(rows):
            ym = _year_month(r)
            if not ym or ym != (2025, month):
                continue
            partner = str(r.get("partnerISO", "")).strip()
            if partner in SKIP_PARTNERS:
                if partner == "W00":
                    w00_idx = i
                continue
            partner_sum += _month_value(rows[i], is_export=is_export)
        if w00_idx is not None:
            _set_month_value(rows[w00_idx], partner_sum, is_export=is_export)

    _write_comtrade(path, fieldnames, rows)

    print(f"Per-country 2025 {label} targets applied\n")
    if full_year_scaled:
        print("Note: Jan–Oct exceeded target; scaled all 2025 months for:")
        for line in full_year_scaled:
            print(f"  {line}")
        print()

    listed = 0.0
    print(f"{'ISO':<6} {'Target M':>10} {'Actual M':>10} {'Jan-Oct M':>10} {'Nov-Dec M':>10}")
    for partner in sorted(targets_musd, key=lambda p: -targets_musd[p]):
        act = 0.0
        jo = 0.0
        nd = 0.0
        for r in rows:
            ym = _year_month(r)
            if not ym or ym[0] != 2025 or str(r.get("partnerISO", "")).strip() != partner:
                continue
            v = _month_value(r, is_export=is_export)
            act += v
            if ym[1] in BASE_MONTHS:
                jo += v
            elif ym[1] in NEW_MONTHS:
                nd += v
        listed += act
        tgt = targets_musd[partner]
        print(f"{partner:<6} {tgt:>10.0f} {act/1e6:>10.1f} {jo/1e6:>10.1f} {nd/1e6:>10.1f}")

    grand = 0.0
    for r in rows:
        ym = _year_month(r)
        if not ym or ym[0] != 2025:
            continue
        if str(r.get("partnerISO", "")).strip() in SKIP_PARTNERS:
            continue
        grand += _month_value(r, is_export=is_export)
    other = grand - listed
    print(
        f"\nListed partners annual total: ${listed/1e9:.3f} B "
        f"(target table ~${sum(targets_musd.values())/1000:.3f} B)"
    )
    print(f"Other partners annual total:  ${other/1e9:.3f} B")
    print(f"All partners annual total:    ${grand/1e9:.3f} B\n")


def adjust_exports_per_country() -> None:
    _adjust_2024_exports(EXPORTS, EXPORT_2024_TARGETS_MUSD)
    _adjust_file_per_country(EXPORTS, EXPORT_TARGETS_MUSD, is_export=True, label="export")


def adjust_imports_per_country() -> None:
    _adjust_file_per_country(IMPORTS, IMPORT_TARGETS_MUSD, is_export=False, label="import")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-only", action="store_true")
    parser.add_argument("--imports-only", action="store_true")
    parser.add_argument("--run-preprocess", action="store_true")
    args = parser.parse_args()

    do_exports = not args.imports_only
    do_imports = not args.exports_only

    if do_exports:
        if not EXPORTS.exists():
            raise SystemExit(f"Missing {EXPORTS}")
        adjust_exports_per_country()
    if do_imports:
        if not IMPORTS.exists():
            raise SystemExit(f"Missing {IMPORTS}")
        adjust_imports_per_country()

    if args.run_preprocess:
        import subprocess

        print("\nRunning preprocess_data.py …")
        subprocess.run(
            [sys.executable, str(PROJECT / "scripts/preprocess_data.py")],
            cwd=str(PROJECT),
            env={**os.environ, "PYTHONPATH": str(PROJECT)},
            check=True,
        )
        print("Preprocessing complete.")


if __name__ == "__main__":
    main()
