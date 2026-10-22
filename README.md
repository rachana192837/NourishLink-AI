# NourishLink AI

Final-year AIML project: a Django food-bank platform that combines a
rule-based FEFO allocation engine with a demand forecasting model and
LLM-backed AI agents, so coordinators can turn donated perishable stock
into distribution plans faster and waste food less.

## The three allocation configurations evaluated

| Config | Policy | Where it lives |
|--------|--------|----------------|
| **A** | Rule-based FEFO only — earliest-distribution-cutoff batches first, requests ordered by required-by date | `core/allocation.py` |
| **B** | FEFO + forecast-informed ordering — requests containing a product whose 7-day demand forecast exceeds free stock are prioritised by worst gap | `core/allocation.py` (`forecast_boost=True`) |
| **C** | FEFO + AI agents — an Inventory Analyst (reads stock/forecasts/requests) and a Distribution Planner (drafts a plan through tool calls). The LLM (OpenRouter) is optional; a deterministic fallback agent chain produces equivalent drafts when the API is unavailable | `core/agent_tools.py`, `core/agents.py` |

Safety principle: the LLM never mutates stock or gets free-form DB
access. Its only write-capable tool creates a **draft** allocation
through the same audited engine used by Config A, and no stock state
changes until a human coordinator approves the plan.

## Presentation layer (why the app is self-explanatory)

- Dashboard, Forecasts and Reports pages explain *what each number means
  and why it matters* in plain words, and include offline-capable
  Chart.js visualisations (4-week demand outlook, product gap bars,
  requested-vs-distributed bars).
- **Evaluation page** (`/reports/evaluation/`, sidebar → Evaluation):
  renders the A/B/C comparison and forecast quality as a one-page,
  plain-English summary with a "Save as PDF" button — the thesis artifact.
- **Dispatch board** (plan detail → "Open dispatch board"):
  fulfilment-control-plane view of one plan — Leaflet map (OpenStreetMap
  tiles online, automatic offline SVG schematic fallback) showing the
  warehouse → charity truck routes (nearest-neighbour grouping, indicative
  only), per-truck manifests with stop numbers and case counts, KPI chips
  (stops / cases / delivered / remaining) and the plan's execution record
  (activity log + AI agent runs) as a timeline. Read-only; trucks/stops
  routes are computed per view and change nothing.

- **Design system** (Bangalore Food Bank inspired, `templates/base.html`):
  orange `#eb5310` action palette + deep navy `#082454` sidebar/hero, warm
  gray background, pill buttons, Mulish display + Quicksand body type
  (variable woff2 files vendored under `static/vendor/fonts/` — no Google
  CDN needed at runtime; dashboard shows a mission hero, four animated
  count-up impact counters (all-time, honest zeros allowed) and a
  01–04 "how it works" strip; the dark dispatch board is navy-themed).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install django torch numpy scikit-learn

python manage.py migrate
python manage.py seed_data      # synthetic organisation, users, products
python manage.py simulate --seed 42    # synthetic 12-month history + live state
python manage.py forecast       # trains the NN forecaster + baseline, stores forecasts
python manage.py createsuperuser

python manage.py runserver
```

Demo credentials (from `seed_data`): `admin / admin123`,
`coord1 / coord123`, `partner1..partner6 / partner123`.

## Key commands

| Command | Purpose |
|---------|---------|
| `python manage.py seed_data` | Seed base orgs/users/products |
| `python manage.py simulate --seed 42` | Generate synthetic donation/request history + live pending pool |
| `python manage.py forecast` | Train PyTorch MLP forecaster (with synthetic-lag ablation) and the rolling-28 baseline; writes `data/forecast/report.json` |
| `python manage.py agent_proposal` | Run the analyst+planner cycle from the CLI (draft plan only) |
| `python manage.py evaluate` | Run the A/B/C comparison over the current scenario; writes `data/evaluation/results.json` |

## Optional: enabling the LLM (Config C "live" mode)

The agents work fully without any API key (deterministic fallback,
marked `FALLBACK` in the run records). To use the real OpenRouter API:

1. Create a (free) account at openrouter.ai.
2. Set the environment variable before `runserver`:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
# optional, default is the free auto-router:
$env:OPENROUTER_MODEL = "openrouter/free"
```

Approved LLM usage (per project plan): the LLM only *narrates and
proposes* (analyst report + planner draft through tools); all numeric
facts come from DB tools, all state changes remain behind the audited
approval flow, and any OpenRouter outage degrades gracefully to the
deterministic fallback.

## Test suite

```bash
python manage.py test core
```

75 tests cover the simulator, FEFO engine, approval workflow,
forecast orchestration, agent tool layer, agent loops (scripted
fake-LLM), agent security, and the forecast-boost ordering.

## Ethical notes baked into the design

- **Determinism under stress:** when the LLM is down or the API is
  misconfigured, the system continues at full capacity with the
  fallback agents (no stock lock-out during outages).
- **Human in the loop:** agents can only produce drafts; approval,
  record-keeping and stock changes are coordinator actions.
- **Data realism:** all records are synthetic; forecasts are replayed
  against hold-out weeks with the metric (MAE/WAPE) and a documented
  ablation (synthetic lag) to avoid over-claiming.
- **Waste visibility:** the Reports screen shows avoided-at-risk units
  and recorded waste so the intervention is measurable.

## Project layout

```
nourishlink/        Django settings
core/               models, engine, forecaster, agents, views, commands
templates/core/     server-rendered UI (no JS framework needed)
data/forecast/      hold-out report + future forecasts JSON
data/evaluation/    A/B/C results from `evaluate`
```
