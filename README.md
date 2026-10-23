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
  fulfilment-control-plane view of one plan — Leaflet map (resilient
  tile chain: Wikimedia → CARTO → OSM; automatic offline SVG schematic
  fallback) showing the warehouse → charity truck routes. Routes are a
  **stored record**: computed once per truck count and persisted on the
  plan (`?refresh=1` recomputes), drawn as **real driving roads with
  per-stop ETAs** via the public OSRM router (straight-line fallback
  when offline). Per-truck manifests carry stop status chips
  (planned/scheduled/delivered/cancelled); KPI chips, and the plan's
  execution record (activity log + AI agent runs) as a timeline.
  Read-only; nothing here changes data.
- **Weekly Ledger** (`/reports/ledger/`, sidebar → Weekly Ledger):
  government-style accountability record for any week — donations
  received per donor (batches, units, top products + bar chart), stock
  distributed per partner organisation (requests served, allocation
  lines, units + chart), and a net-flow footer (received in /
  distributed out / expired unrescued). Week navigation, Print/Save-as-
  PDF and CSV export (same week). 

- **Design system** (Bangalore Food Bank inspired, `templates/base.html`):
  orange `#eb5310` action palette + deep navy `#082454` sidebar/hero, warm
  gray background, pill buttons, Mulish display + Quicksand body type
  (variable woff2 files vendored under `static/vendor/fonts/` — no Google
  CDN needed at runtime; dashboard shows a mission hero, four animated
  count-up impact counters (all-time, honest zeros allowed) and a
  01–04 "how it works" strip; the dark dispatch board is navy-themed).
  Photo layer: dashboard hero photo, mission band and login visual use
  public-domain food-bank photographs vendored offline in `static/img/`
  (Wikimedia Commons; credited in the footer).

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
| `python manage.py paperfigures` | Export the two publication figures (`data/paper/fig1_abc_comparison.png`, `fig2_forecast_quality.png`) from `results.json` at 300 dpi |

## Optional: enabling the LLM (Config C "live" mode)

The agents work fully without any API key (deterministic fallback,
marked `FALLBACK` in the run records). To use the real OpenRouter API:

1. Create a (free) account at openrouter.ai.
2. Set the environment variable permanently (preferred):

```powershell
setx OPENROUTER_API_KEY "sk-or-..."
# then open a NEW terminal; optionally override the auto-router:
#   setx OPENROUTER_MODEL "openrouter/free"
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

75+ tests cover the simulator, FEFO engine, approval workflow,
forecast orchestration, agent tool layer, agent loops (scripted
fake-LLM), agent security, the forecast-boost ordering, dispatch-board
route persistence/OSRM fallback, and the weekly ledger.

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
