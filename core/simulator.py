"""Synthetic food-bank data simulator for NourishLink AI.

Generates a reproducible, clearly-labelled *simulation* dataset:
  - daily per-organisation demand for each product (ground truth for ML training)
  - partner requests assembled from that demand (twice-weekly ordering days)
  - donation batches with received dates, distribution cutoffs and blocked flags
  - disruption events: demand spikes, donation gaps, blocked batches

Pure functions only — no Django imports — so the forecasting code (Phase 5)
and evaluation harness (Phase 7) can import this module directly.

Realism notes (documented limitations for the final report):
  - demand is Poisson noise around a seasonal mean; real food-bank demand has
    messier drivers (referrals, funding cycles) that we do not model
  - requested quantities equal true demand (no censoring by stockouts)
  - synthetic performance must NOT be presented as real-world effectiveness
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Static catalogue
# ---------------------------------------------------------------------------

PRODUCTS = [
    # name, unit, category, base weekly demand (units across all partners)
    ("Rice", "1kg pack", "GRAIN", 120),
    ("Beans", "400g pack", "LEGUME", 80),
    ("Pasta", "500g pack", "PASTA", 70),
    ("Canned Tomatoes", "400g can", "CANNED", 90),
    ("Cooking Oil", "1L bottle", "OIL", 40),
    ("Sugar", "1kg pack", "OTHER", 60),
    ("Tea", "250g box", "BEVERAGE", 35),
    ("Biscuits", "200g pack", "SNACK", 55),
    ("Lentils", "500g pack", "LEGUME", 65),
    ("Powdered Milk", "400g tin", "OTHER", 45),
    ("Oats", "500g pack", "GRAIN", 50),
    ("Canned Beans", "400g can", "CANNED", 60),
]

# Products deliberately supplied below demand so evaluation scenarios
# contain genuine shortages (documented in metadata.json).
SCARCE_PRODUCTS = {"Rice", "Cooking Oil", "Powdered Milk"}

ORGANIZATIONS = [
    ("Hope Shelter", "SHELTER"),
    ("City Kitchen", "COMMUNITY_KITCHEN"),
    ("Care NGO", "NGO"),
    ("Sunrise Shelter", "SHELTER"),
    ("Community Kitchen", "COMMUNITY_KITCHEN"),
    ("Relief Foundation", "NGO"),
]

DONORS = ["Harvest Donors", "City Grocers", "Community Farms", "Goodwill Foods"]

# Day-of-week demand multipliers (Mon=0..Sun=6) per organisation type.
DOW_MULTIPLIERS = {
    "SHELTER": [0.9, 0.9, 1.0, 1.0, 1.1, 1.3, 1.3],
    "COMMUNITY_KITCHEN": [1.2, 1.1, 1.1, 1.1, 1.0, 0.6, 0.5],
    "NGO": [1.0, 1.0, 1.0, 1.0, 1.1, 0.9, 0.9],
}

CUTOFF_CHOICES = [7, 14, 21, 30, 45, 60, 90]
CUTOFF_WEIGHTS = [0.05, 0.10, 0.20, 0.25, 0.20, 0.12, 0.08]

N_SPIKES = 10
N_DONATION_GAPS = 2
BLOCKED_FRACTION = 0.08


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(seed: int = 42, days: int = 365, start: date | None = None) -> dict:
    """Run the full simulation. Returns dict of DataFrames + metadata."""
    if start is None:
        start = date.today() - timedelta(days=days - 1)
    rng = np.random.default_rng(seed)
    dates = [start + timedelta(days=i) for i in range(days)]

    product_names = [p[0] for p in PRODUCTS]
    org_names = [o[0] for o in ORGANIZATIONS]
    org_type = dict(ORGANIZATIONS)

    # Distinct demand patterns: per-product Dirichlet shares across orgs.
    shares = {
        p: rng.dirichlet(np.full(len(org_names), 2.0)) for p in product_names
    }
    # Gentle per-product trend (+/-0.08% per day).
    trends = {p: rng.uniform(-0.0008, 0.0008) for p in product_names}

    # --- disruption: demand spikes --------------------------------------
    spikes = []
    for _ in range(N_SPIKES):
        spikes.append({
            "start_day": int(rng.integers(0, days - 8)),
            "duration": int(rng.integers(3, 8)),
            "products": sorted(rng.choice(
                product_names,
                size=int(rng.integers(1, 4)),
                replace=False,
            ).tolist()),
            "multiplier": round(float(rng.uniform(2.0, 3.0)), 2),
        })

    def spike_factor(day_idx: int, product: str) -> float:
        factor = 1.0
        for s in spikes:
            if s["start_day"] <= day_idx < s["start_day"] + s["duration"]:
                if product in s["products"]:
                    factor *= s["multiplier"]
        return factor

    # --- disruption: donation gaps --------------------------------------
    gaps = []
    for _ in range(N_DONATION_GAPS):
        gap_start = int(rng.integers(30, days - 30))
        gap_len = int(rng.integers(10, 15))
        gaps.append({"start_day": gap_start, "end_day": gap_start + gap_len})
    in_gap = np.zeros(days, dtype=bool)
    for g in gaps:
        in_gap[g["start_day"]:g["end_day"]] = True

    # --- daily demand ----------------------------------------------------
    demand_rows = []
    for day_idx, day in enumerate(dates):
        dow = day.weekday()
        for p_idx, (name, _unit, _cat, weekly) in enumerate(PRODUCTS):
            for o_idx, org in enumerate(org_names):
                lam = (
                    weekly / 7.0
                    * shares[name][o_idx]
                    * DOW_MULTIPLIERS[org_type[org]][dow]
                    * (1.0 + trends[name] * day_idx)
                    * spike_factor(day_idx, name)
                )
                qty = int(rng.poisson(max(lam, 0.0)))
                if qty:
                    demand_rows.append({
                        "date": day.isoformat(),
                        "product": name,
                        "organization": org,
                        "quantity": qty,
                    })
    daily_demand = pd.DataFrame(
        demand_rows, columns=["date", "product", "organization", "quantity"]
    )

    # --- donation batches ------------------------------------------------
    # Supply is target-driven: each product aims for a fixed supply/demand
    # ratio (0.9 scarce, 1.2 normal) so evaluation scenarios deterministically
    # contain both shortages and surplus regardless of seed. Arrival days,
    # quantities (jittered, normalised to target) and blocked flags stay random.
    batch_rows = []
    counter: dict[str, int] = {}
    for name, _unit, _cat, weekly in PRODUCTS:
        target_ratio = 0.85 if name in SCARCE_PRODUCTS else 1.25
        target_supply = target_ratio * weekly * 52 * 1.05  # 5% spike allowance
        arrival_p = 0.00133 * weekly
        n_arrivals = max(8, int(round(days * arrival_p)))
        raw = rng.uniform(0.8, 1.2, size=n_arrivals)
        qtys = np.maximum(10, np.round(target_supply * raw / raw.sum())).astype(int)
        arrival_days = sorted(
            rng.choice(days, size=n_arrivals, replace=False).tolist()
        )
        tag = name.upper().replace(" ", "-")
        for day_idx, qty in zip(arrival_days, qtys):
            if in_gap[day_idx] and rng.random() < 0.85:
                continue  # donation gap: this delivery never arrives
            day = dates[day_idx]
            counter[tag] = counter.get(tag, 0) + 1
            cutoff_days = int(rng.choice(CUTOFF_CHOICES, p=CUTOFF_WEIGHTS))
            blocked = bool(rng.random() < BLOCKED_FRACTION)
            batch_rows.append({
                "batch_code": f"SIM-{tag}-{counter[tag]:03d}",
                "product": name,
                "donor": str(rng.choice(DONORS)),
                "quantity_received": int(qty),
                "received_date": day.isoformat(),
                "distribution_cutoff": (
                    day + timedelta(days=cutoff_days)
                ).isoformat(),
                "status": "BLOCKED" if blocked else "AVAILABLE",
            })
    batches = pd.DataFrame(
        batch_rows,
        columns=["batch_code", "product", "donor", "quantity_received",
                 "received_date", "distribution_cutoff", "status"],
    )

    # --- partner requests (Mon & Thu ordering days) ----------------------
    demand_by_day = {}
    for _, r in daily_demand.iterrows():
        demand_by_day.setdefault(
            (r["date"], r["organization"], r["product"]), 0
        )
        demand_by_day[(r["date"], r["organization"], r["product"])] += int(
            r["quantity"]
        )

    request_rows, item_rows = [], []
    request_id = 0
    for org in org_names:
        order_days = [d for d in dates if d.weekday() in (0, 3)]
        if order_days[-1] != dates[-1]:
            # Final flush: pack demand after the last ordering day so
            # requested totals reconcile exactly with daily demand.
            order_days.append(dates[-1])
        prev = dates[0] - timedelta(days=1)
        for req_day in order_days:
            accumulated: dict[str, int] = {}
            d = prev + timedelta(days=1)
            while d <= req_day:
                for p in product_names:
                    q = demand_by_day.get((d.isoformat(), org, p), 0)
                    if q:
                        accumulated[p] = accumulated.get(p, 0) + q
                d += timedelta(days=1)
            prev = req_day
            if not accumulated:
                continue
            request_id += 1
            required_by = req_day + timedelta(
                days=int(rng.integers(3, 8))
            )
            request_rows.append({
                "request_id": request_id,
                "organization": org,
                "request_date": req_day.isoformat(),
                "required_by_date": required_by.isoformat(),
            })
            for product, qty in sorted(accumulated.items()):
                item_rows.append({
                    "request_id": request_id,
                    "product": product,
                    "quantity_requested": qty,
                })

    requests = pd.DataFrame(
        request_rows,
        columns=["request_id", "organization", "request_date",
                 "required_by_date"],
    )
    request_items = pd.DataFrame(
        item_rows, columns=["request_id", "product", "quantity_requested"]
    )

    # --- supply/demand balance per product --------------------------------
    demand_total = daily_demand.groupby("product")["quantity"].sum()
    supply_total = batches.groupby("product")["quantity_received"].sum()
    balance = {
        p: round(float(supply_total.get(p, 0)) / max(float(demand_total.get(p, 0)), 1), 3)
        for p in product_names
    }

    metadata = {
        "seed": seed,
        "days": days,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "n_products": len(PRODUCTS),
        "n_organizations": len(ORGANIZATIONS),
        "n_spikes": N_SPIKES,
        "spikes": spikes,
        "donation_gaps": gaps,
        "blocked_fraction_target": BLOCKED_FRACTION,
        "scarce_products": sorted(SCARCE_PRODUCTS),
        "supply_demand_ratio": balance,
        "n_demand_rows": len(daily_demand),
        "n_batches": len(batches),
        "n_requests": len(requests),
        "n_request_items": len(request_items),
        "total_demand_units": int(daily_demand["quantity"].sum()),
        "total_supply_units": int(batches["quantity_received"].sum()),
        "note": "SYNTHETIC simulation study data — not real food-bank records.",
    }
    return {
        "daily_demand": daily_demand,
        "batches": batches,
        "requests": requests,
        "request_items": request_items,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Validation + export
# ---------------------------------------------------------------------------

def validate(frames: dict) -> tuple[list[str], dict]:
    """Integrity checks. Returns (errors, summary). Empty errors == pass."""
    errors: list[str] = []
    dd, batches, reqs, items = (
        frames["daily_demand"], frames["batches"],
        frames["requests"], frames["request_items"],
    )
    products = {p[0] for p in PRODUCTS}
    orgs = {o[0] for o in ORGANIZATIONS}

    if (dd["quantity"] < 0).any():
        errors.append("negative quantity in daily_demand")
    if (items["quantity_requested"] <= 0).any():
        errors.append("non-positive quantity_requested in request_items")
    if (batches["quantity_received"] <= 0).any():
        errors.append("non-positive quantity_received in batches")
    if set(dd["product"].unique()) - products:
        errors.append("unknown product in daily_demand")
    if set(items["product"].unique()) - products:
        errors.append("unknown product in request_items")
    if set(reqs["organization"].unique()) - orgs:
        errors.append("unknown organization in requests")

    b = batches.copy()
    b["received_date"] = pd.to_datetime(b["received_date"])
    b["distribution_cutoff"] = pd.to_datetime(b["distribution_cutoff"])
    if not (b["distribution_cutoff"] > b["received_date"]).all():
        errors.append("some batches have cutoff <= received_date")

    r = reqs.copy()
    r["request_date"] = pd.to_datetime(r["request_date"])
    r["required_by_date"] = pd.to_datetime(r["required_by_date"])
    if not (r["required_by_date"] >= r["request_date"]).all():
        errors.append("some requests have required_by < request_date")

    orphan_items = set(items["request_id"].unique()) - set(reqs["request_id"].unique())
    if orphan_items:
        errors.append(f"{len(orphan_items)} request_items reference missing requests")

    # Requested totals must reconcile with daily demand accumulated into
    # ordering windows (guards against generation bugs, not a model claim).
    requested_total = int(items["quantity_requested"].sum())
    demand_total = int(dd["quantity"].sum())
    if requested_total != demand_total:
        errors.append(
            f"requested total ({requested_total}) != demand total ({demand_total})"
        )

    blocked_pct = float((batches["status"] == "BLOCKED").mean() * 100)
    summary = {
        "date_range": f"{dd['date'].min()}..{dd['date'].max()}",
        "demand_rows": len(dd),
        "demand_units": demand_total,
        "batches": len(batches),
        "supply_units": int(batches["quantity_received"].sum()),
        "blocked_pct": round(blocked_pct, 1),
        "requests": len(reqs),
        "request_items": len(items),
        "supply_demand_ratio": frames["metadata"]["supply_demand_ratio"],
    }
    return errors, summary


def export(frames: dict, out_dir: str | Path) -> Path:
    """Write CSVs + metadata.json. Returns output directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames["daily_demand"].to_csv(out / "daily_demand.csv", index=False)
    frames["batches"].to_csv(out / "batches.csv", index=False)
    frames["requests"].to_csv(out / "requests.csv", index=False)
    frames["request_items"].to_csv(out / "request_items.csv", index=False)
    (out / "metadata.json").write_text(
        json.dumps(frames["metadata"], indent=2), encoding="utf-8"
    )
    return out
