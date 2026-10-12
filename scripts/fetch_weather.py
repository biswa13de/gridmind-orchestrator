"""Download real historical weather for every generator site from the Open-Meteo archive API.

Usage:  python scripts/fetch_weather.py [--start 2025-01-01] [--end 2025-12-31]

Writes data/weather/<asset_id>.csv (hourly, UTC). The files are committed to the repo so the
project runs offline; this script exists so anyone can reproduce or extend the dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402

from gridmind.config import ROOT, load_portfolio  # noqa: E402

API = "https://archive-api.open-meteo.com/v1/archive"


def fetch(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url = (
        f"{API}?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
        "&hourly=shortwave_radiation,wind_speed_100m,temperature_2m,cloud_cover&wind_speed_unit=ms"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.load(resp)
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()

    out_dir = ROOT / "data" / "weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    for g in load_portfolio().generators:
        df = fetch(g.lat, g.lon, args.start, args.end)
        df.to_csv(out_dir / f"{g.id}.csv", index=False)
        print(f"{g.id:6s} {g.name:18s} {len(df)} rows")


if __name__ == "__main__":
    main()
