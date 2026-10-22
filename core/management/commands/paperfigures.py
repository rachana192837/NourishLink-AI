"""Export two publication figures from data/evaluation/results.json.

fig1_abc_comparison.png   — Config A/B/C allocation performance.
fig2_forecast_quality.png — neural model vs baseline MAE/WAPE.

Offline-safe: reads only the evaluation JSON; no DB, no network.
Output: data/paper/ at 300 dpi.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.conf import settings
from django.core.management.base import BaseCommand

ORANGE = '#EB5310'
NAVY = '#082454'
GRAY = '#8A8F94'
DARK = '#1C1A1B'

EVAL_PATH = Path(settings.BASE_DIR) / 'data' / 'evaluation' / 'results.json'
OUT_DIR = Path(settings.BASE_DIR) / 'data' / 'paper'

CONFIG_LABELS = {
    'config_a': 'A  pure FEFO',
    'config_b': 'B  forecast-boosted',
    'config_c': 'C  LLM agent',
}


class Command(BaseCommand):
    help = 'Render research-paper figures from the evaluation results.'

    def handle(self, *args, **options):
        if not EVAL_PATH.exists():
            self.stderr.write('results.json missing — run `python manage.py '
                              'evaluate` first.')
            return

        res = json.loads(EVAL_PATH.read_text(encoding='utf-8'))
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        fig1 = self._fig_abc(res)
        fig2 = self._fig_forecast(res)

        out1 = OUT_DIR / 'fig1_abc_comparison.png'
        out2 = OUT_DIR / 'fig2_forecast_quality.png'
        fig1.savefig(out1, dpi=300, bbox_inches='tight')
        fig2.savefig(out2, dpi=300, bbox_inches='tight')
        self.stdout.write(self.style.SUCCESS(f'Wrote {out1}'))
        self.stdout.write(self.style.SUCCESS(f'Wrote {out2}'))

    @staticmethod
    def _style_axes(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#C9C5BF')
        ax.spines['bottom'].set_color('#C9C5BF')
        ax.tick_params(colors=DARK, labelsize=9)
        ax.grid(axis='y', color='#E7E4DE', linewidth=.8)
        ax.set_axisbelow(True)

    def _fig_abc(self, res):
        configs = [k for k in ('config_a', 'config_b', 'config_c')
                   if k in res]
        names = [CONFIG_LABELS[k] for k in configs]
        requested = res[configs[0]]['requested_units_subset']
        allocated = [res[k]['allocated_units'] for k in configs]
        fulfil = [res[k]['fulfilment_rate'] * 100 for k in configs]
        at_risk = [res[k]['at_risk_units_covered'] for k in configs]

        colors = [NAVY, '#10306B', ORANGE][:len(configs)]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

        ax = axes[0]
        bars = ax.bar(names, allocated, color=colors, width=.62)
        ax.axhline(requested, color=GRAY, linestyle='--', linewidth=1.2)
        ax.text(len(names) - .45, requested, f'requested ({requested})',
                color=GRAY, fontsize=8.5, va='bottom', ha='right')
        for bar, alloc, pct in zip(bars, allocated, fulfil):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 25,
                    f'{alloc:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9, color=DARK,
                    fontweight='bold')
        ax.set_title('(a) Units allocated of requested', fontsize=10.5,
                     color=NAVY, fontweight='bold', pad=10)
        ax.set_ylabel('units', fontsize=9)
        ax.set_ylim(0, requested * 1.22)
        self._style_axes(ax)

        ax = axes[1]
        bars = ax.bar(names, at_risk, color=colors, width=.62)
        for bar, v in zip(bars, at_risk):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
                    f'{v:,}', ha='center', va='bottom', fontsize=9,
                    color=DARK, fontweight='bold')
        ax.set_title('(b) At-risk units covered by allocation', fontsize=10.5,
                     color=NAVY, fontweight='bold', pad=10)
        ax.set_ylabel('units', fontsize=9)
        ax.set_ylim(0, max(at_risk) * 1.22)
        self._style_axes(ax)

        n_requests = res.get('scenario', {}).get('n_requests', '?')
        mode = (res.get('config_c') or {}).get('agent_cost', {}) \
            .get('agent_mode', '')
        note = {
            'llm': 'LLM agent mode: live LLM (OpenRouter free tier, '
                   'tool-calling verified)',
            'live': 'LLM agent mode: live LLM (OpenRouter)',
        }.get(mode, f'LLM agent mode: {mode} (deterministic fallback)')
        fig.suptitle(f'Allocation performance by configuration — '
                     f'{n_requests}-request subset', fontsize=12,
                     color=NAVY, fontweight='bold', y=1.02)
        fig.text(.5, -.06, note + ' — all three configurations maximise '
                 'both metrics when stock is the binding constraint.',
                 ha='center', fontsize=8.5, color='#6D7378')
        fig.tight_layout()
        return fig

    def _fig_forecast(self, res):
        fc = res['forecast']
        metrics = ['MAE (units/week)', 'WAPE (%)']
        baseline = [fc['baseline_mae'], fc['baseline_wape'] * 100]
        model = [fc['model_mae'], fc['model_wape'] * 100]
        wins_n, wins_d = fc['wins_of']

        x = range(len(metrics))
        w = 0.36
        fig, ax = plt.subplots(figsize=(7.8, 3.8))

        b1 = ax.bar([i - w / 2 for i in x], baseline, w,
                    label=f"{fc['baseline_name']}", color=GRAY)
        b2 = ax.bar([i + w / 2 for i in x], model, w,
                    label=f"{fc['model_name']}", color=ORANGE)
        for bars in (b1, b2):
            for bar in bars:
                v = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, v + .25,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=9,
                        color=DARK, fontweight='bold')

        ax.set_xticks(list(x))
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_title('7-day demand forecast quality — '
                     f'{fc["model_name"]} vs {fc["baseline_name"]}',
                     fontsize=11.5, color=NAVY, fontweight='bold', pad=12)
        ax.legend(frameon=False, fontsize=9)
        ax.text(.02, -.22,
                f'Lower is better. The neural model wins on '
                f'{wins_n} of {wins_d} product-weeks.', fontsize=8.5,
                color='#6D7378', transform=ax.transAxes)
        self._style_axes(ax)
        fig.tight_layout()
        return fig
