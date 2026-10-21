"""Run the A/B/C evaluation and save results to data/evaluation/results.json."""

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from ... import evaluation


class Command(BaseCommand):
    help = 'Evaluate Config A (FEFO), B (forecast-ordered FEFO) and C (agents).'

    def add_arguments(self, parser):
        parser.add_argument('--no-agents', action='store_true',
                            help='Skip Config C (agents)')
        parser.add_argument('--out',
                            help='Output JSON path (default data/evaluation/results.json)')

    def handle(self, *args, **options):
        results = evaluation.evaluate_configs(
            include_agents=not options['no_agents'])

        default_out = (Path(__file__).resolve().parents[3]
                       / 'data' / 'evaluation' / 'results.json')
        out = Path(options['out']) if options['out'] else default_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str),
                       encoding='utf-8')

        self.stdout.write(self.style.SUCCESS(f'Results written: {out}'))
        self._print_table(results)

    def _print_table(self, r):
        def fmt(x, fallback='-'):
            return '-' if x is None else x

        self.stdout.write('\nForecast quality (7-day totals, holdout MAE)')
        f = r.get('forecast') or {}
        if f:
            wins, n = f.get('wins_of', (0, 0))
            self.stdout.write(
                f"  {f.get('model_name')}: MAE {fmt(f.get('model_mae')):.2f} "
                f"WAPE {fmt(f.get('model_wape')):.3f}")
            self.stdout.write(
                f"  {f.get('baseline_name')}: MAE {fmt(f.get('baseline_mae')):.2f} "
                f"WAPE {fmt(f.get('baseline_wape')):.3f}")
            self.stdout.write(f'  wins {wins}/{n}')

        self.stdout.write('\nAllocation configs (same pending pool, drafts only)')
        self.stdout.write(
            '  %-9s %12s %12s %10s %8s' %
            ('config', 'allocated', 'unmet', 'fulfil%', 'items'))
        for key, label in (('config_a', 'A  FEFO'),
                           ('config_b', 'B  +forecast'),
                           ('config_c', 'C  +agents')):
            row = r.get(key)
            if not row:
                continue
            self.stdout.write(
                '  %-9s %12s %12s %10s %8s' % (
                    label,
                    fmt(row.get('allocated_units'), 0),
                    fmt(row.get('unmet_units_after_plan'), 0),
                    f"{fmt(row.get('fulfilment_rate'), 0):.1%}"
                    if row.get('fulfilment_rate') is not None else '-',
                    fmt(row.get('n_plan_items'), 0)))
            validation = row.get('validation') or {}
            if validation.get('errors'):
                for err in validation['errors'][:3]:
                    self.stdout.write(f'      ! {err}')
            audit = row.get('agent_audit')
            if audit:
                self.stdout.write(
                    f"      agent mode: {row.get('agent_cost', {}).get('agent_mode')}"
                    f", tool_calls {row.get('agent_cost', {}).get('tool_calls')}"
                    f", tokens {row.get('agent_cost', {}).get('tokens_prompt')}"
                    f"+{row.get('agent_cost', {}).get('tokens_completion')}"
                    f", latency {row.get('agent_cost', {}).get('latency_ms')} ms"
                    f", id-checks {audit.get('checks')}"
                    f", violations {len(audit.get('violations', []))}")
        self.stdout.write(
            f"\nRuntime: {fmt(r.get('runtime_seconds'))} s | "
            'drafts deleted; live stock untouched')
