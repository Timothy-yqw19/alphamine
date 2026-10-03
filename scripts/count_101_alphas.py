#!/usr/bin/env python
"""Classify the 101 Formulaic Alphas by the inputs they need.

The project can only reproduce alphas that are expressible in the fields it has
(OHLCV plus `returns`).  Its documentation used to quote "35 reproducible, 48
needing vwap / market cap / industry" without a way to check it.  This script
re-derives the split by *statically* parsing two open-source implementations, so
the number is reproducible rather than remembered.

It downloads two files into a cache directory (``/tmp`` by default) and never
imports or executes them - the code is untrusted, and nothing here needs to run
it.

Usage::

    python scripts/count_101_alphas.py
    python scripts/count_101_alphas.py --cache ~/.cache/alphamine
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import urllib.request

BASE = "https://raw.githubusercontent.com/yli188/WorldQuant_alpha101_code/master"
FILES = ("101Alpha_code_1.py", "101Alpha_code_2.py")

#: Fields the project actually has.  Anything outside this set is a blocker.
HAVE = {"open", "high", "low", "close", "volume", "returns", "vwap_proxy", "adv20"}

OHLCV = {"open", "high", "low", "close", "volume", "returns", "adv20"}


def fetch(cache: pathlib.Path) -> dict[str, str]:
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in FILES:
        local = cache / name
        if not local.is_file():
            urllib.request.urlretrieve(f"{BASE}/{name}", local)  # noqa: S310
        out[name] = local.read_text()
    return out


def parse_zipline(text: str) -> dict[int, str]:
    """`code_2`: one ``class AlphaN(CustomFactor)`` with an explicit ``inputs`` list."""

    parts = re.split(r"\n\s*class Alpha(\d+)\b", text)
    found = {}
    for i in range(1, len(parts), 2):
        body = parts[i + 1]
        m = re.search(r"inputs\s*=\s*\[(.*?)\]", body, re.S)
        found[int(parts[i])] = m.group(1) if m else ""
    return found


def parse_pandas(text: str) -> dict[int, str]:
    """`code_1`: one indented ``def alphaN(self)`` per alpha."""

    parts = re.split(r"\n\s+def\s+alpha0*(\d+)\s*\(", text)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def classify(body: str) -> str:
    """What does this alpha need beyond OHLCV?"""

    if re.search(r"IndClass|indneutralize", body):
        return "industry"
    if re.search(r"\bvwap\b", body, re.I):
        return "vwap"
    if re.search(r"market_cap|\bcap\b(?!e)", body, re.I):
        return "market cap"
    return "ohlcv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/tmp/alpha101_src", type=pathlib.Path)
    args = ap.parse_args()

    src = fetch(args.cache)
    pand = parse_pandas(src["101Alpha_code_1.py"])
    zip_ = parse_zipline(src["101Alpha_code_2.py"])

    buckets = collections.Counter(classify(b) for b in pand.values())
    ohlcv_1 = {n for n, b in pand.items() if classify(b) == "ohlcv"}
    ohlcv_2 = {n for n, b in zip_.items() if classify(b) == "ohlcv"}
    safe = ohlcv_1 & ohlcv_2
    unimplemented = sorted(set(range(1, 102)) - set(pand))

    print(f"parsed {args.cache}")
    print(f"  code_1 implements {len(pand)} alphas")
    print(f"  code_2 implements {len(zip_)} alphas")
    print()
    print(f"  OHLCV-expressible          {buckets['ohlcv']}")
    print(f"  needs vwap                 {buckets['vwap']}")
    print(f"  needs an industry class    {buckets['industry']}")
    print(f"  needs market cap           {buckets['market cap']}")
    print(f"  not implemented in code_1  {len(unimplemented)}")
    print()
    covered = sum(buckets.values()) + len(unimplemented)
    print(f"  {buckets['ohlcv']} + {len(pand) - buckets['ohlcv']} + "
          f"{len(unimplemented)} = {covered}")
    print(f"  OHLCV-only in *both* files: {len(safe)}  <- safe to transcribe")
    print(f"  ids: {sorted(safe)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
