"""Generate the synthetic food-bank simulation and optionally load it into the DB.

Usage:
    python manage.py simulate                        # seed 42, 365 days, load DB
    python manage.py simulate --seed 7 --days 365
    python manage.py simulate --no-load-db            # CSVs only
    python manage.py simulate --out-dir data/sim_v2

CSVs + metadata.json go to <out-dir> (ML training input for Phase 5).
DB load is idempotent: volatile data (requests, batches, plans, forecasts)
is cleared first; catalogue rows (products, orgs, users, donors) are reused.
"""

from datetime import date, datetime, timedelta, timezone as dt_timezone

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import (
    ActivityLog,
    AllocationItem,
    AllocationPlan,
    DemandForecast,
    Donor,
    FoodBatch,
    Organization,
    Product,
    Profile,
    Request,
    RequestItem,
)
from core.simulator import PRODUCTS, ORGANIZATIONS, DONORS, export, generate, validate


class Command(BaseCommand):
    help = "Generate synthetic simulation data, export CSVs, load into DB."

    def add_arguments(self, parser):
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--days", type=int, default=365)
        parser.add_argument("--start", type=str, default=None,
                            help="Start date YYYY-MM-DD (default: today - days + 1)")
        parser.add_argument("--out-dir", type=str, default="data/simulation")
        parser.add_argument("--no-load-db", action="store_true",
                            help="Export CSVs only, skip DB load")

    def handle(self, *args, **options):
        seed, days = options["seed"], options["days"]
        start = (date.fromisoformat(options["start"]) if options["start"]
                 else date.today() - timedelta(days=days - 1))

        self.stdout.write(f"Simulating {days} days from {start} (seed={seed})...")
        frames = generate(seed=seed, days=days, start=start)

        errors, summary = validate(frames)
        if errors:
            for e in errors:
                self.stderr.write(self.style.ERROR(f"  VALIDATION FAIL: {e}"))
            raise SystemExit(1)
        self.stdout.write(self.style.SUCCESS("  Validation passed."))

        out = export(frames, options["out_dir"])
        self.stdout.write(f"  CSVs + metadata.json written to {out}/")

        if not options["no_load_db"]:
            self._load_db(frames)

        self.stdout.write(self.style.SUCCESS("\n--- Simulation Summary ---"))
        for key, value in summary.items():
            if key == "supply_demand_ratio":
                continue
            self.stdout.write(f"  {key:18s} {value}")
        self.stdout.write("  per-product supply/demand:")
        for product, ratio in summary["supply_demand_ratio"].items():
            flag = "  <-- scarce" if ratio < 1.0 else ""
            self.stdout.write(f"    {product:18s} {ratio}{flag}")

    # ------------------------------------------------------------------

    def _load_db(self, frames):
        self.stdout.write("Loading simulation into database...")
        RequestItem.objects.all().delete()
        Request.objects.all().delete()
        AllocationItem.objects.all().delete()
        AllocationPlan.objects.all().delete()
        DemandForecast.objects.all().delete()
        FoodBatch.objects.all().delete()

        products = {}
        for name, unit, category, _weekly in PRODUCTS:
            product, _ = Product.objects.update_or_create(
                name=name, defaults={"unit": unit, "category": category})
            products[name] = product

        donors = {}
        for name in DONORS:
            donor, _ = Donor.objects.get_or_create(name=name)
            donors[name] = donor

        orgs, partners = {}, {}
        for idx, (name, org_type) in enumerate(ORGANIZATIONS, start=1):
            org, _ = Organization.objects.update_or_create(
                name=name, defaults={"org_type": org_type, "is_active": True})
            orgs[name] = org
            user, created = User.objects.get_or_create(
                username=f"partner{idx}",
                defaults={"email": f"partner{idx}@nourishlink.local",
                          "first_name": name.split()[0], "last_name": "Staff"})
            if created:
                user.set_password("partner123")
                user.save()
            Profile.objects.update_or_create(
                user=user, defaults={"role": "PARTNER", "organization": org})
            partners[name] = user

        if not User.objects.filter(
                username="coord1", profile__role="COORDINATOR").exists():
            coord = User.objects.create_user(
                username="coord1", email="coord1@nourishlink.local",
                password="coord123")
            Profile.objects.create(user=coord, role="COORDINATOR")

        today = timezone.now().date()
        batch_map = {}
        for _, row in frames["batches"].iterrows():
            cutoff = date.fromisoformat(row["distribution_cutoff"])
            if row["status"] == "BLOCKED":
                status = "BLOCKED"
            elif cutoff < today:
                status = "EXPIRED"
            else:
                status = "AVAILABLE"
            batch = FoodBatch.objects.create(
                product=products[row["product"]],
                donor=donors[row["donor"]],
                batch_code=row["batch_code"],
                quantity_received=int(row["quantity_received"]),
                quantity_available=int(row["quantity_received"]),
                quantity_reserved=0,
                received_date=date.fromisoformat(row["received_date"]),
                distribution_cutoff=cutoff,
                status=status,
            )
            batch_map[row["batch_code"]] = batch

        items_by_request = {}
        for _, row in frames["request_items"].iterrows():
            items_by_request.setdefault(int(row["request_id"]), []).append(row)

        n_requests = 0
        for _, row in frames["requests"].iterrows():
            org = orgs[row["organization"]]
            req = Request.objects.create(
                organization=org,
                requested_by=partners[row["organization"]],
                required_by_date=date.fromisoformat(row["required_by_date"]),
                status="PENDING",
                notes="Simulated request",
            )
            # backdate to the simulated request date (auto_now_add override)
            created_at = datetime.combine(
                date.fromisoformat(row["request_date"]),
                datetime.min.time(),
            ).replace(tzinfo=dt_timezone.utc)
            Request.objects.filter(pk=req.pk).update(created_at=created_at)
            for item in items_by_request.get(int(row["request_id"]), []):
                RequestItem.objects.create(
                    request=req,
                    product=products[item["product"]],
                    quantity_requested=int(item["quantity_requested"]),
                    quantity_allocated=0,
                )
            n_requests += 1

        ActivityLog.objects.create(
            user=None, action_type="SIMULATION_LOAD",
            description=(f"Loaded synthetic simulation: {len(batch_map)} "
                         f"batches, {n_requests} requests "
                         f"({frames['metadata']['start_date']}.."
                         f"{frames['metadata']['end_date']})"))

        self.stdout.write(
            f"  Loaded {len(batch_map)} batches, {n_requests} requests.")
