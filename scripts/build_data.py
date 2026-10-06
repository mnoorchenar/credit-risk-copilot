"""Generate the SYNTHETIC residential-secured-lending portfolio used by the demo.

Nothing here comes from a real lender. Values are drawn from simple, plausible relationships (expensive markets carry
higher LTV/DTI and more payment-shock exposure, resource regions carry higher unemployment, ...) with a fixed seed so the
file is reproducible. The risk-score formula below MUST match static/js/engine.js (tests/engine.test.mjs checks it).

Run:  python scripts/build_data.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "portfolio.json"

# id: (short name, portfolio size weight, price index 0-1, unemployment %, trend shape)
# trend shape = share of today's delinquency level that existed 24 months ago (lower = deteriorating faster)
REGIONS = {
    2226: ("Algoma", 0.5, 0.15, 7.4, 0.80),
    2230: ("Durham", 3.0, 0.62, 6.0, 0.70),
    2233: ("Grey Bruce", 0.7, 0.35, 5.2, 0.90),
    2236: ("Halton", 2.6, 0.85, 5.1, 0.78),
    2237: ("Hamilton", 2.2, 0.60, 6.4, 0.66),
    2240: ("Chatham-Kent", 0.5, 0.25, 6.8, 0.84),
    2242: ("Lambton", 0.5, 0.25, 6.6, 0.88),
    2244: ("London", 1.8, 0.45, 6.7, 0.74),
    2246: ("Niagara", 1.9, 0.50, 7.1, 0.64),
    2247: ("North Bay", 0.5, 0.25, 6.9, 0.86),
    2249: ("Northwest", 0.35, 0.15, 5.9, 0.92),
    2251: ("Ottawa", 3.6, 0.60, 5.3, 0.90),
    2253: ("Peel", 6.2, 0.78, 7.0, 0.52),
    2257: ("Renfrew", 0.4, 0.25, 5.2, 0.95),
    2258: ("Eastern Ontario", 0.9, 0.30, 5.9, 0.90),
    2260: ("Simcoe Muskoka", 2.0, 0.55, 6.1, 0.72),
    2261: ("Sudbury", 0.7, 0.25, 6.5, 0.88),
    2262: ("Thunder Bay", 0.6, 0.20, 6.6, 0.85),
    2265: ("Waterloo", 2.0, 0.60, 6.8, 0.68),
    2266: ("Guelph-Wellington", 1.4, 0.55, 5.0, 0.86),
    2268: ("Windsor-Essex", 1.3, 0.35, 8.6, 0.58),
    2270: ("York", 4.2, 0.85, 5.4, 0.74),
    3895: ("Toronto", 9.0, 0.95, 7.8, 0.56),
    4913: ("Southwestern", 0.8, 0.30, 4.9, 0.94),
    5183: ("Huron Perth", 0.5, 0.30, 4.4, 0.96),
    7652: ("Grand Erie", 1.0, 0.40, 6.2, 0.80),
    7653: ("Lakelands", 0.9, 0.35, 6.5, 0.82),
    7654: ("Northeast", 0.7, 0.20, 7.2, 0.84),
    7655: ("Southeast", 1.5, 0.40, 5.8, 0.88),
}

# Region-level overrides that make the demo story interesting (an adjudication outlier, a data-quality outlier, ...).
OVERRIDE_RATE = {2268: 9.4, 2262: 8.9, 2253: 7.9}
DATA_QUALITY = {2249: 93.8, 2253: 95.6, 2261: 95.2}

# (driver, low anchor, high anchor, weight, higher_is_worse)
DRIVERS = [
    ("dpd90", 0.05, 0.60, 0.30, True),
    ("ltv", 50, 75, 0.15, True),
    ("dti", 30, 46, 0.15, True),
    ("renew", 18, 42, 0.12, True),
    ("unemp", 4.0, 9.5, 0.13, True),
    ("hpi", -8.0, 4.0, 0.08, False),
    ("score", 700, 770, 0.07, False),
]
MONTHS = 24
AS_OF = "2026-09-30"


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def risk_score(v):
    total = 0.0
    for key, lo, hi, w, worse_high in DRIVERS:
        t = clip((v[key] - lo) / (hi - lo))
        total += w * (t if worse_high else 1 - t)
    return round(100 * total, 1)


def month_labels():
    y, m = 2026, 9
    out = []
    for _ in range(MONTHS):
        out.append(f"{y}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out[::-1]


def build_region(rid, spec):
    short, size, price, unemp, shape = spec
    rng = random.Random(rid)
    n = lambda s: rng.gauss(0, s)  # noqa: E731

    ltv = 57 + 15 * price + n(1.6)
    dti = 33 + 9.5 * price + n(1.2)
    score = 748 - 16 * price - 3.0 * (unemp - 6) + n(3)
    renew = 21 + 17 * price + n(2.4)
    hpi = 2.6 - 6.8 * price + n(1.0)
    dpd90 = 0.09 + 0.05 * max(0.0, unemp - 4.5) + 0.12 * price + 0.0009 * (dti - 35) * 10 + n(0.02)
    dpd90 = max(0.06, dpd90)
    override = OVERRIDE_RATE.get(rid, 3.2 + 2.8 * price + abs(n(0.9)))
    dq = DATA_QUALITY.get(rid, 98.9 - 1.6 * rng.random() - 0.8 * price)
    heloc = 7 + 13 * price + n(1.5)
    balance_m = size * 14000 / 9.0 * 0.62  # CAD millions, Toronto ~ $13.6B
    avg_loan = 255 + 340 * price  # CAD thousands
    accounts = int(balance_m * 1000 / avg_loan)

    cur = dict(dpd90=dpd90, ltv=ltv, dti=dti, renew=renew, unemp=unemp, hpi=hpi, score=score)

    # 24 months of history: delinquency climbs from `shape * today` to today; other drivers drift into place.
    dpd_hist, risk_hist = [], []
    for t in range(MONTHS):
        f = t / (MONTHS - 1)
        d = dpd90 * (shape + (1 - shape) * f) * (1 + n(0.025))
        d = max(0.03, d)
        past = dict(cur)
        past.update(
            dpd90=d,
            ltv=ltv - 2.5 * (1 - f) * (0.4 + price),
            dti=dti - 1.8 * (1 - f) * renew / 30,
            renew=renew * (0.70 + 0.30 * f),
            unemp=unemp - 1.2 * (1 - f),
        )
        dpd_hist.append(round(d, 3))
        risk_hist.append(risk_score(past))
    dpd_hist[-1] = round(dpd90, 3)
    risk_hist[-1] = risk_score(cur)

    return {
        "id": rid,
        "short": short,
        "balance_m": round(balance_m),
        "accounts": accounts,
        "score": round(score),
        "ltv": round(ltv, 1),
        "dti": round(dti, 1),
        "renew": round(renew, 1),
        "unemp": round(unemp, 1),
        "hpi": round(hpi, 1),
        "dpd90": round(dpd90, 3),
        "dpd30": round(dpd90 * (2.0 + rng.random() * 0.5), 3),
        "override": round(override, 1),
        "dq": round(dq, 1),
        "heloc": round(heloc, 1),
        "dpd90_hist": dpd_hist,
        "risk_hist": risk_hist,
    }


def main():
    regions = [build_region(rid, spec) for rid, spec in REGIONS.items()]
    for r in regions:
        r["risk"] = risk_score(r | {"hpi": r["hpi"]})
    payload = {
        "meta": {
            "as_of": AS_OF,
            "synthetic": True,
            "note": "Synthetic demo data. Not from any lender. Region polygons: Ontario public health unit boundaries (Open Government Licence - Ontario), used only as a stand-in for credit-risk regions.",
            "months": month_labels(),
            "drivers": [
                {"key": k, "lo": lo, "hi": hi, "weight": w, "worse_high": wh} for k, lo, hi, w, wh in DRIVERS
            ],
        },
        "regions": regions,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(regions)} regions)")
    for r in sorted(regions, key=lambda x: -x["risk"]):
        trend = (r["dpd90_hist"][-1] - r["dpd90_hist"][-7]) * 100
        print(f"{r['short']:<18}{r['risk']:>6}  dpd90={r['dpd90']:.2f}  ltv={r['ltv']}  dti={r['dti']}  6m={trend:+.1f}bps  bal={r['balance_m']}")


if __name__ == "__main__":
    main()
