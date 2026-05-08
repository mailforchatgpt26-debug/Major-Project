"""
Download real sector-disaggregated trade data from CEPII BACI database.

BACI HS17 V202401b contains bilateral trade flows at 6-digit HS level for 2017-2022.
We stream-decompress only pharma (HS chapter 30) and textile (HS chapters 50-63) rows
directly from the remote zip without downloading the full 647MB file.

Output: data/raw/comtrade/TradeData_sectors.csv
  Columns: reporterCode, partnerCode, cmdCode, refYear, primaryValue, dist
  - cmdCode 30  = Pharmaceuticals
  - cmdCode 60  = Textiles (aggregated chapters 50-63, labelled with 60 as sentinel)
  - primaryValue in millions USD (matching original TradeData.csv units)
  - dist merged from existing TradeData.csv for matching pairs

Usage:
    python scripts/download_baci_sector_data.py
    python scripts/download_baci_sector_data.py --years 2017 2018 2019 2020
"""

import sys
import struct
import zlib
import io
import csv
import argparse
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

BACI_ZIP_URL = "https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS17_V202401b.zip"

# Byte offsets of each year's local file header inside the zip (determined from central dir)
YEAR_OFFSETS = {
    2017: 0,
    2018: 98_711_815,
    2019: 205_479_116,
    2020: 315_436_922,
    2021: 422_664_599,
    2022: 535_372_125,
}

# Country codes file offset
COUNTRY_CODES_OFFSET = 646_828_208

PHARMA_CHAPTER   = 30         # HS chapter 30 = pharmaceutical products
TEXTILE_CHAPTERS = range(50, 64)  # HS chapters 50–63 = textiles & apparel

# Sentinel cmdCode values written to output (loader maps these to sector names)
PHARMA_CMDCODE  = 30
TEXTILE_CMDCODE = 60


def _fetch_bytes(url: str, start: int, end: int, retries: int = 3) -> bytes:
    import urllib.request
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt+1}/{retries} after error: {e}")
            time.sleep(2 ** attempt)


def _read_local_header(url: str, file_offset: int) -> tuple[int, int]:
    """Return (data_start_offset, compressed_size) for a file in the zip."""
    hdr = _fetch_bytes(url, file_offset, file_offset + 31)
    assert hdr[:4] == b"PK\x03\x04", "Bad local file header signature"
    fname_len = struct.unpack_from("<H", hdr, 26)[0]
    extra_len = struct.unpack_from("<H", hdr, 28)[0]
    # comp_size in local header may be 0 when data descriptor is used;
    # we'll stream until zlib signals end-of-stream instead.
    data_start = file_offset + 30 + fname_len + extra_len
    return data_start


def _stream_decompress_csv(url: str, data_start: int, chunk_size: int = 1 << 20):
    """
    Stream-decompress a deflate-compressed CSV entry from the zip.
    Yields decoded text lines one at a time.
    """
    import urllib.request
    decompressor = zlib.decompressobj(-15)
    buf = ""
    offset = data_start

    while True:
        try:
            raw = _fetch_bytes(url, offset, offset + chunk_size - 1)
        except Exception:
            break
        if not raw:
            break

        try:
            chunk = decompressor.decompress(raw)
        except zlib.error:
            # End of compressed stream reached mid-chunk; flush remaining
            chunk = decompressor.flush()
            buf += chunk.decode("utf-8", errors="replace")
            lines = buf.split("\n")
            yield from lines[:-1]
            break

        buf += chunk.decode("utf-8", errors="replace")
        lines = buf.split("\n")
        # Yield all complete lines, keep the last partial one in buf
        yield from lines[:-1]
        buf = lines[-1]
        offset += len(raw)

        if decompressor.unused_data:
            # zlib signals stream end
            if buf:
                yield buf
            break

    if buf:
        yield buf


def load_baci_country_codes(url: str) -> dict[int, str]:
    """Return {baci_code -> iso3} mapping from the country codes CSV in the zip."""
    print("Loading BACI country codes...")
    data_start = _read_local_header(url, COUNTRY_CODES_OFFSET)
    raw = _fetch_bytes(url, data_start, data_start + 30_000)
    content = zlib.decompress(raw, -15).decode("utf-8")

    mapping = {}
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        try:
            code = int(row["country_code"])
            iso3 = row["country_iso3"].strip()
            if iso3:
                mapping[code] = iso3
        except (ValueError, KeyError):
            pass
    print(f"  Loaded {len(mapping)} country codes")
    return mapping


def stream_year(url: str, year: int, baci_to_iso3: dict, valid_pairs: set) -> list[dict]:
    """
    Stream one year's BACI CSV and aggregate pharma + textile trade.
    Returns list of {reporterCode, partnerCode, cmdCode, refYear, primaryValue}.
    """
    file_offset = YEAR_OFFSETS[year]
    data_start = _read_local_header(url, file_offset)

    # Accumulate sums by (reporter, partner, sector_cmd)
    sums: dict[tuple, float] = defaultdict(float)

    print(f"  Streaming {year}...", end=" ", flush=True)
    row_count = 0
    match_count = 0
    header_skipped = False

    for line in _stream_decompress_csv(url, data_start):
        line = line.strip()
        if not line:
            continue
        if not header_skipped:
            header_skipped = True
            continue  # skip header row

        # BACI format: t,i,j,k,v,q
        parts = line.split(",")
        if len(parts) < 5:
            continue

        try:
            i = int(float(parts[1]))   # exporter BACI code
            j = int(float(parts[2]))   # importer BACI code
            k = int(float(parts[3]))   # HS6 code
            v = float(parts[4])        # value in 1000 USD
        except (ValueError, IndexError):
            continue

        chapter = k // 10000

        if chapter == PHARMA_CHAPTER:
            cmd = PHARMA_CMDCODE
        elif chapter in TEXTILE_CHAPTERS:
            cmd = TEXTILE_CMDCODE
        else:
            row_count += 1
            continue

        # Map to ISO3 and check against valid pairs
        iso_i = baci_to_iso3.get(i)
        iso_j = baci_to_iso3.get(j)
        if iso_i is None or iso_j is None:
            row_count += 1
            continue

        if valid_pairs and (iso_i, iso_j) not in valid_pairs:
            row_count += 1
            continue

        # BACI v is in thousands USD; existing TradeData uses millions USD
        sums[(i, j, cmd)] += v / 1000.0
        match_count += 1
        row_count += 1

    print(f"{row_count:,} rows, {match_count:,} pharma/textile matches")

    return [
        {"reporterCode": i, "partnerCode": j, "cmdCode": cmd,
         "refYear": year, "primaryValue": val}
        for (i, j, cmd), val in sums.items()
        if val > 0
    ]


def main():
    parser = argparse.ArgumentParser(description="Download BACI sector trade data")
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2017, 2018, 2019, 2020],
                        help="Years to download (default: 2017-2020)")
    parser.add_argument("--include-2021-2022", action="store_true",
                        help="Also download 2021 and 2022 for richer training data")
    args = parser.parse_args()

    years = args.years
    if args.include_2021_2022:
        years = sorted(set(years) | {2021, 2022})

    project_root = Path(__file__).parent.parent
    old_trade_path = project_root / "data/raw/comtrade/TradeData.csv"
    out_path = project_root / "data/raw/comtrade/TradeData_sectors.csv"

    # Build the set of valid country pairs from the existing dataset so we only
    # keep pairs the rest of the pipeline already knows about.
    valid_pairs: set[tuple[str, str]] = set()
    dist_map: dict[tuple[int, int], float] = {}

    if old_trade_path.exists():
        print("Reading existing TradeData.csv for valid pairs and distances...")
        old = pd.read_csv(old_trade_path)
        import pycountry

        def num_to_iso3(val):
            try:
                c = pycountry.countries.get(numeric=str(int(float(val))).zfill(3))
                return c.alpha_3 if c else None
            except Exception:
                return None

        for _, row in old.iterrows():
            iso_r = num_to_iso3(row["reporterCode"])
            iso_p = num_to_iso3(row["partnerCode"])
            if iso_r and iso_p:
                valid_pairs.add((iso_r, iso_p))
            if "dist" in old.columns and pd.notna(row.get("dist")):
                dist_map[(int(row["reporterCode"]), int(row["partnerCode"]))] = float(row["dist"])
        print(f"  {len(valid_pairs):,} valid pairs, {len(dist_map):,} distance entries")
    else:
        print("No existing TradeData.csv — will download all pairs")

    # Load BACI country codes
    baci_to_iso3 = load_baci_country_codes(BACI_ZIP_URL)

    # Invert valid_pairs to BACI numeric codes for fast lookup
    # (BACI codes = standard ISO numeric, same as reporterCode in old file)
    import pycountry as _pc
    iso3_to_baci: dict[str, int] = {}
    for baci_code, iso3 in baci_to_iso3.items():
        iso3_to_baci[iso3] = baci_code

    valid_baci_pairs: set[tuple[int, int]] = set()
    if valid_pairs:
        for iso_r, iso_p in valid_pairs:
            bi = iso3_to_baci.get(iso_r)
            bj = iso3_to_baci.get(iso_p)
            if bi and bj:
                valid_baci_pairs.add((bi, bj))
        print(f"  {len(valid_baci_pairs):,} valid BACI-code pairs")

    # Stream each year
    all_rows = []
    for year in years:
        if year not in YEAR_OFFSETS:
            print(f"Year {year} not available in BACI HS17 V202401b, skipping")
            continue
        rows = stream_year(BACI_ZIP_URL, year, baci_to_iso3, valid_baci_pairs)
        # Attach distance
        for r in rows:
            r["dist"] = dist_map.get((r["reporterCode"], r["partnerCode"]), None)
        all_rows.extend(rows)
        print(f"    → {len(rows):,} aggregated trade flows for {year}")

    if not all_rows:
        print("ERROR: No data downloaded. Check connectivity and year offsets.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["refYear", "reporterCode", "partnerCode", "cmdCode"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    pharma = (df["cmdCode"] == PHARMA_CMDCODE).sum()
    textile = (df["cmdCode"] == TEXTILE_CMDCODE).sum()
    print(f"\nSaved {len(df):,} rows to {out_path}")
    print(f"  Pharmaceuticals (HS 30): {pharma:,} flows")
    print(f"  Textiles (HS 50-63):     {textile:,} flows")
    print(f"\nSample IND→USA rows:")
    import pycountry as _pc2
    ind_code = next((c for c, iso in baci_to_iso3.items() if iso == "IND"), None)
    usa_code = next((c for c, iso in baci_to_iso3.items() if iso == "USA"), None)
    sample = df[(df["reporterCode"] == ind_code) & (df["partnerCode"] == usa_code)]
    print(sample.to_string(index=False))

    print("\nNext step: run   python scripts/preprocess_data.py --trade-file TradeData_sectors.csv")


if __name__ == "__main__":
    main()
