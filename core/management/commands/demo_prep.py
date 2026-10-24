"""Nudge the live database into a demo-friendly shortage state.

Scenario tool, same spirit as ``simulate --seed 42`` (but smaller): when
the Sourcing page reads "healthy week" because recent imports filled the
warehouse, this trims free stock of up to three forecast-covered products
to ~60% of their 14-day predicted demand — creating believable shortfall
recommendations for the presentation without touching reservations or
history. Idempotent: re-running only tops the existing deficit up.
"""

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db.models import Sum
from django.utils import timezone

from core import forecast as forecasting
from core import sourcing
from core.models import DemandForecast, FoodBatch

DEMO_TARGET_COVER = 0.6   # free stock ends at ~60% of 14-day forecast
DEMO_PRODUCTS = 3


class Command(BaseCommand):
    help = ("Nudge a few products' free stock below their forecast so the "
            "Sourcing page shows live recommendations (demo scenario tool).")

    def handle(self, *args, **options):
        today = timezone.now().date()
        mlp_version = forecasting.MODEL_NAME

        rows = sourcing.compute_sourcing_rows(mlp_version, top_n=100)
        if rows:
            self.stdout.write(self.style.WARNING(
                f'Sourcing already shows {len(rows)} recommendation(s) — '
                f'the page is demo-ready, nothing nudged.'))
            self._print_summary(rows)
            return

        deferred = []
        for forecast_row in (DemandForecast.objects
                             .filter(model_version=mlp_version,
                                     forecast_date__gte=today
                                     - timedelta(days=3))
                             .select_related('product')):
            product = forecast_row.product
            agg = FoodBatch.objects.filter(
                product=product, status='AVAILABLE',
                distribution_cutoff__gte=today).aggregate(
                available=Sum('quantity_available'),
                reserved=Sum('quantity_reserved'))
            stock_free = (agg['available'] or 0) - (agg['reserved'] or 0)
            if stock_free < 10:
                continue
            target = max(1, int(forecast_row.predicted_quantity
                                * DEMO_TARGET_COVER))
            if stock_free <= target:
                continue
            deferred.append((product, stock_free, target))

        if not deferred:
            self.stdout.write(self.style.NOTICE(
                'No product has both forecast rows and free stock — '
                'run `python manage.py forecast` (and `simulate`) first.'))
            return

        deferred.sort(key=lambda entry: -entry[1])
        for product, stock_free, target in deferred[:DEMO_PRODUCTS]:
            self._drain(product, stock_free - target)

        rows = sourcing.compute_sourcing_rows(mlp_version, top_n=6)
        self.stdout.write(self.style.SUCCESS(
            f'Demo state ready: drained {min(len(deferred), DEMO_PRODUCTS)} '
            f'product(s); sourcing now shows {len(rows)} recommendation(s).'))
        self._print_summary(rows)

    def _drain(self, product, reduce_by):
        """Reduce the earliest-cutoff batches first (FEFO-consistent —
        stock dying soonest is the first to go), never touching
        reserved units."""
        remaining = reduce_by
        for batch in FoodBatch.objects.filter(
                product=product, status='AVAILABLE').order_by(
                'distribution_cutoff'):
            free = batch.quantity_available - batch.quantity_reserved
            take = min(free, remaining)
            if take <= 0:
                continue
            batch.quantity_available -= take
            batch.save(update_fields=['quantity_available', 'updated_at'])
            remaining -= take
            if remaining <= 0:
                break

    def _print_summary(self, rows):
        for row in rows[:6]:
            self.stdout.write(
                f"  - {row['product_name']}: source {row['units_needed']} "
                f"units by {row['cover_by'] or 'later'} "
                f"({row['partner_count']} partner(s) exposed)")
