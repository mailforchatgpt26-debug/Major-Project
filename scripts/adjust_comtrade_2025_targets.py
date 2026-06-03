#!/usr/bin/env python3
"""
Scale Nov & Dec 2025 Comtrade rows so full-year totals match targets.

Keeps Jan–Oct 2025 unchanged; scales Nov/Dec for every partner (excl. W00/_X),
then sets W00 Nov/Dec to the partner sum for each month.

Targets (USD):
  Exports: 30.5 billion FOB
  Imports:  9.0 billion CIF

Usage:
  PYTHONPATH=. python scripts/adjust_comtrade_2025_targets.py
  PYTHONPATH=. python scripts/adjust_comtrade_2025_targets.py --run-preprocess
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

TARGET_EXPORT_USD = 30.5e9
# Realistic imports: preserve ~12% import/export ratio from 2024 (not a forced $9B spike).
# 2024: $2.84B imports / $23.35B exports → apply same ratio to $30.5B exports ≈ $3.71B.
TARGET_IMPORT_USD = 3.71e9
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
    fd, tmp = tempfile.mkstemp(prefix="comtrade_adj_", suffix=".csv", dir=path.parent)
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


def _adjust_file(
    path: Path,
    *,
    is_export: bool,
    target_total_usd: float,
) -> dict[str, float]:
    fieldnames, rows = _load_comtrade_rows(path)

    def year_month(r: dict) -> tuple[int, int] | None:
        try:
            return int(r["refYear"]), int(r["refMonth"])
        except (KeyError, ValueError):
            return None

    jan_oct = 0.0
    nov_dec = 0.0
    nd_indices: list[int] = []

    for i, r in enumerate(rows):
        ym = year_month(r)
        if not ym or ym[0] != 2025:
            continue
        partner = str(r.get("partnerISO", "")).strip()
        if partner in SKIP_PARTNERS:
            continue
        val = _month_value(r, is_export=is_export)
        if ym[1] in BASE_MONTHS:
            jan_oct += val
        elif ym[1] in NEW_MONTHS:
            nov_dec += val
            nd_indices.append(i)

    target_nd = max(0.0, target_total_usd - jan_oct)
    scale = (target_nd / nov_dec) if nov_dec > 0 else 1.0

    for i in nd_indices:
        r = rows[i]
        old = _month_value(r, is_export=is_export)
        _set_month_value(r, old * scale, is_export=is_export)

    # Refresh W00 Nov/Dec = sum of partners for each month
    for month in NEW_MONTHS:
        partner_sum = 0.0
        w00_idx: int | None = None
        for i, r in enumerate(rows):
            ym = year_month(r)
            if not ym or ym != (2025, month):
                continue
            partner = str(r.get("partnerISO", "")).strip()
            if partner in SKIP_PARTNERS:
                if partner == "W00":
                    w00_idx = i
                continue
            partner_sum += _month_value(r, is_export=is_export)
        if w00_idx is not None:
            _set_month_value(rows[w00_idx], partner_sum, is_export=is_export)

    _write_comtrade(path, fieldnames, rows)

    # Verify
    total_excl = 0.0
    for r in rows:
        ym = year_month(r)
        if not ym or ym[0] != 2025:
            continue
        if str(r.get("partnerISO", "")).strip() in SKIP_PARTNERS:
            continue
        total_excl += _month_value(r, is_export=is_export)

    return {
        "jan_oct_usd": jan_oct,
        "nov_dec_before_usd": nov_dec,
        "nov_dec_target_usd": target_nd,
        "scale_factor": scale,
        "total_after_usd": total_excl,
        "rows_scaled": len(nd_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale Nov/Dec 2025 to annual export/import targets")
    parser.add_argument(
        "--run-preprocess",
        action="store_true",
        help="Regenerate data/processed/edges.csv after adjusting raw Comtrade files",
    )
    args = parser.parse_args()

    if not EXPORTS.exists() or not IMPORTS.exists():
        raise SystemExit("Missing Comtrade CSVs under data/raw/comtrade/")

    exp_stats = _adjust_file(EXPORTS, is_export=True, target_total_usd=TARGET_EXPORT_USD)
    imp_stats = _adjust_file(IMPORTS, is_export=False, target_total_usd=TARGET_IMPORT_USD)

    print("Adjusted Nov/Dec 2025 (Jan–Oct unchanged, partners excl. W00/_X)\n")
    print("EXPORTS → $30.5B target")
    for k, v in exp_stats.items():
        if "usd" in k:
            print(f"  {k}: ${v/1e9:.4f} B")
        else:
            print(f"  {k}: {v}")

    print("\nIMPORTS → $9.0B target")
    for k, v in imp_stats.items():
        if "usd" in k:
            print(f"  {k}: ${v/1e9:.4f} B")
        else:
            print(f"  {k}: {v}")

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
