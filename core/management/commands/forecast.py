"""Train/evaluate the demand forecaster and persist its outputs.

Usage
-----
    python manage.py forecast                       # full run (default data)
    python manage.py forecast --epochs 150          # quick demo run
    python manage.py forecast --no-save             # console only

What it does
------------
1. Trains the 3-seed residual MLP on data/simulation/daily_demand.csv and
   evaluates it against the rolling-28-day weekly baseline on a 12-week
   trailing holdout (weekly MAE/WAPE).
2. Prints the comparison table.
3. Writes a JSON report (default data/forecast/report.json) that the
   "Forecasts" page uses.
4. Persists next-7-day totals per product as DemandForecast rows
   (model_version 'torch-mlp-v2-residual' and the baseline reference),
   replacing the previous run's rows.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from argparse import BooleanOptionalAction

from django.core.management.base import BaseCommand
from django.utils.dateparse import parse_date
from django.utils import timezone

from core import forecast
from core.models import DemandForecast, Product


class Command(BaseCommand):
    help = "Train the demand forecaster, evaluate vs baseline, save outputs."

    def add_arguments(self, parser):
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--epochs", type=int,
                            default=forecast.DEFAULT_EPOCHS)
        parser.add_argument("--test-weeks", type=int,
                            default=forecast.N_WEEKS_TEST)
        parser.add_argument("--val-weeks", type=int, default=0,
                            help="trailing train blocks for early stopping "
                                 "(0 = off; ablation showed it hurts here)")
        parser.add_argument("--source", type=str,
                            default="data/simulation/daily_demand.csv")
        parser.add_argument("--report", type=str,
                            default="data/forecast/report.json")
        parser.add_argument("--save", action=BooleanOptionalAction,
                            default=True, help="persist DemandForecast rows")

    def handle(self, *args, **opts):
        source = Path(opts["source"])
        if not source.is_absolute():
            source = Path.cwd() / source
        if not source.exists():
            self.stderr.write(self.style.ERROR(
                f"daily-demand data not found: {source}\n"
                "Run `python manage.py simulate` first."))
            raise SystemExit(1)

        report_path = Path(opts["report"])
        if not report_path.is_absolute():
            report_path = Path.cwd() / report_path

        self.stdout.write(
            f"Training {forecast.MODEL_NAME} (seeds "
            f"{forecast.ENSEMBLE_SEEDS}, {opts['epochs']} epochs) — "
            f"holdout {opts['test_weeks']} weekly windows...")
        report = forecast.train_and_evaluate(
            csv_path=source, seed=opts["seed"], epochs=opts["epochs"],
            test_weeks=opts["test_weeks"], val_weeks=opts["val_weeks"])
        forecast.save_report(report, report_path)
        self.stdout.write(f"Report written: {report_path}")

        ov = report["overall"]
        self.stdout.write(self.style.MIGRATE_HEADING(
            "\nForecast quality (7-day totals, units/week)"))
        self.stdout.write(
            f"{'Product':<24}{'NN MAE':>9}{'base MAE':>10}{'NN WAPE':>10}"
            f"{'base WAPE':>11}  verdict")
        for name, m in report["per_product"].items():
            verdict = "WIN" if m["model_wins"] else "loss"
            self.stdout.write(
                f"{name:<24}{m['model_mae']:>9}{m['baseline_mae']:>10}"
                f"{m['model_wape']:>10}{m['baseline_wape']:>11}  {verdict}")
        self.stdout.write(self.style.SUCCESS(
            f"\nOVERALL  NN MAE {ov['model_mae']} vs baseline "
            f"{ov['baseline_mae']}  |  WAPE {ov['model_wape']} vs "
            f"{ov['baseline_wape']}  |  wins "
            f"{report['model_wins_overall']}/{report['n_products']}"))

        if not opts["save"]:
            self.stdout.write(self.style.WARNING("--no-save: DB left untouched"))
            return
        self._persist_forecasts(report)

    def _persist_forecasts(self, report):
        names = {p.name: p for p in Product.objects.all()}
        # Operational rows: the app consumes these directly, so they are
        # dated at least today even if the training data ended earlier —
        # a one-day-old "next 7 days" is still the coordinator's next 7 days.
        # The scientific vintage stays in report.json (data_end).
        from datetime import timedelta
        next_date = (parse_date(report["data_end"]) + timedelta(days=1)
                     if parse_date(report["data_end"]) else timezone.localdate())
        next_date = max(next_date, timezone.localdate())
        for version, future in (
                (report["model_name"], report["future_model"]),
                (report["baseline_name"], report["future_baseline"])):
            DemandForecast.objects.filter(model_version=version).delete()
            rows = []
            for product_name, total in future.items():
                product = names.get(product_name)
                if product is None:
                    self.stderr.write(self.style.WARNING(
                        f"skipping unknown product in report: {product_name}"))
                    continue
                rows.append(DemandForecast(
                    product=product, forecast_date=next_date, period_days=7,
                    predicted_quantity=float(total),
                    model_version=version))
            DemandForecast.objects.bulk_create(rows)
            self.stdout.write(
                f"Saved {len(rows)} {version} forecasts "
                f"(7-day totals starting {next_date}).")
