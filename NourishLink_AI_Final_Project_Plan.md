# NourishLink AI — Final Project Plan

**Demand Forecasting and Agent-Assisted Food Distribution**

A food-bank decision-support system that predicts upcoming food demand, identifies likely shortages and surplus, and uses LLM agents to recommend practical distribution actions under human approval.

**Scope:** One food bank, several partner organizations, packaged food with fixed units. No vehicle routing or complex cloud infrastructure.

---

## Research Question

> Can a forecast-informed, LLM-agent-assisted allocation workflow improve food-bank planning compared to rule-based FEFO allocation alone?

The answer must come from experiments—not assumed in advance.

---

## Three AI Components

| # | Component | Technique | Input → Output |
|---|-----------|-----------|----------------|
| 1 | **Demand Forecaster** | PyTorch feed-forward NN vs. rolling-average baseline | Historical requests → 7-day demand per product |
| 2 | **Inventory Analyst Agent** | Tool-using LLM (read-only tools) | Current state → shortage/surplus report with batch IDs |
| 3 | **Distribution Planner Agent** | Tool-using LLM (propose + validate tools) | Analyst report → ranked allocation recommendations |

Deterministic FEFO engine + validation sits *outside* the agents. Agents advise; coordinator approves.

---

## System Architecture

```
Coordinator / Partner
        |
   Django Web App
        |
        +-- Inventory & Batches
        +-- Partner Requests
        +-- FEFO Allocation Engine
        +-- Coordinator Approval
        +-- Dashboard & Reports
        |
    PostgreSQL
```

### Data Flow

```
Synthetic Simulator → PostgreSQL
                          ↓
                   Train Forecaster (PyTorch NN)
                          ↓
                   Predict 7-day Demand
                          ↓
              Inventory Analyst Agent (read-only tools)
                          ↓
              Distribution Planner Agent (propose + validate tools)
                          ↓
                   FEFO Allocation Engine
                          ↓
                   Coordinator Approval
                          ↓
                      PostgreSQL
                          ↓
                   Dashboard & Reports
```

---

## Data Strategy

**Build a synthetic food-bank simulator.** This is the most defensible choice.

| Simulator Parameter | Example Values |
|---|---|
| Products | 8–12 shelf-stable items (rice 1kg, beans 400g, pasta 500g, …) |
| Partners | 5–8 organizations with distinct weekly demand patterns |
| Donations | Random arrivals, varying batch sizes, cutoffs 7–90 days out |
| Demand patterns | Weekly seasonality, occasional spikes, gradual trends |
| Disruptions | Stockouts, blocked batches, demand surges |
| Duration | 365 simulated days, seeded for reproducibility |

Generate ~50k–100k request records. Split: first 280 days train, last 85 days test. No leakage across the 7-day forecast window.

### Why Synthetic Over Public Datasets

- The Food Demand Forecasting dataset (456k records) doesn't map cleanly to food-bank batches—bridging takes weeks.
- InvAgent used simulation and got published. A simulation study is academically valid when clearly labelled.
- You control the disruption scenarios, which strengthens evaluation.

---

## Evaluation Design (Three Configurations)

| Config | Allocation Method | Forecasting | Agents |
|---|---|---|---|
| **A** | FEFO rules only | ✗ | ✗ |
| **B** | FEFO + forecast-informed priority | ✓ | ✗ |
| **C** | FEFO + forecast + agent recommendations | ✓ | ✓ |

### Metrics

| Category | Metric |
|---|---|
| Forecast quality | MAE, WAPE vs. rolling-average baseline |
| Fulfilment | % of requested quantity actually allocated |
| Waste reduction | Units reaching cutoff undistributed |
| Agent accuracy | Factual consistency rate (IDs, quantities match ground truth) |
| Agent cost | API calls, tokens, latency per proposal |
| Validity | Zero over-allocations, zero blocked-batch usage |

---

## Tech Stack

| Layer | Choice |
|---|---|
| App | Python 3.11 + Django 5.x |
| DB | PostgreSQL 16 |
| ML | PyTorch + NumPy + pandas |
| Agents | AutoGen (pinned version, adapted from InvAgent patterns) |
| LLM | GPT-4o-mini or Gemini Flash (budget-conscious) |
| Frontend | Django templates + HTMX for interactivity |
| Testing | pytest + Django test client |

---

## Database Schema (Core Entities)

### Entities

- **User** — coordinator or partner representative
- **Organization** — partner org (shelter, kitchen, NGO)
- **Donor** — contact who provides donations
- **Product** — standardized food item (e.g., "Rice 1kg pack")
- **FoodBatch** — specific donated batch with quantity, received date, distribution cutoff, status
- **Request** — partner submission with required-by date and status
- **RequestItem** — individual product + quantity within a request
- **AllocationPlan** — proposed distribution plan with approval status
- **AllocationItem** — links a request item to a specific batch and quantity
- **DemandForecast** — predicted demand per product per period
- **ActivityLog** — audit trail of all actions

### Relationships

```
User ──belongs_to──> Organization
Organization ──submits──> Request
Donor ──donates──> FoodBatch
Product ──categorizes──> FoodBatch
FoodBatch ──supplies──> AllocationItem
Request ──contains──> RequestItem
RequestItem ──fulfilled_by──> AllocationItem
AllocationPlan ──groups──> AllocationItem
AllocationPlan ──approved_by──> User
Product ──predicted_for──> DemandForecast
AllocationPlan ──logs──> ActivityLog
```

### Approval Behavior

- Draft proposals do not reserve stock.
- Approval reserves stock (inside a DB transaction, rechecking eligibility).
- Completed pickup reduces physical stock and clears the reservation.
- Cancellation releases the reservation.
- A blocked batch prevents pickup and flags affected allocations for review.

---

## Agent Tool Definitions

### Inventory Analyst Agent (Read-Only)

| Tool | Signature | Returns |
|---|---|---|
| `get_inventory` | `(product_id?)` | `[{batch_id, product, qty_available, cutoff_date}]` |
| `get_pending_requests` | `()` | `[{request_id, org, product, qty, required_by}]` |
| `get_demand_forecast` | `(product_id?)` | `[{product, forecast_7d, current_stock, gap}]` |
| `get_batches_at_risk` | `(days=7)` | `[{batch_id, product, qty_unreserved, cutoff_date, forecast_demand}]` |

### Distribution Planner Agent (Propose + Validate)

| Tool | Signature | Returns |
|---|---|---|
| `generate_allocation` | `(request_ids)` | `{plan_id, items: [{request_item, batch, qty}], unmet: [...]}` |
| `validate_allocation` | `(plan_id)` | `{valid: bool, errors: [...]}` |

### Agent Constraints

- Both agents get ≤5 tool calls per invocation.
- Planner gets 1 retry after validation failure.
- No open-ended loops or autonomous conversations.
- Agents cannot modify inventory, approve allocations, or override batch blocks.
- If LLM API is unavailable, the app falls back to deterministic summaries.

---

## Application Screens

| Screen | Purpose |
|---|---|
| **Dashboard** | Available stock, approaching cutoffs, pending requests, forecast highlights |
| **Inventory & Batches** | Batch list, filters, donation entry, batch blocking |
| **Partner Requests** | Submissions, status tracking (partner sees only their own) |
| **Forecasts & Risk** | 7-day demand forecast, projected gaps, at-risk batches |
| **AI Proposal Review** | Agent recommendations, AI explanation, approve/reject controls |
| **Reports & History** | Distribution totals, recorded waste, activity log |

---

## 12-Week Development Schedule

| Week | Deliverable | Acceptance Criteria |
|---|---|---|
| **1** | Project setup, DB schema, Django models, admin | Models migrate; admin CRUD works |
| **2** | Synthetic simulator script | Generates 365 days of reproducible data; CSV + DB load |
| **3** | Inventory views, donation entry, batch management | Coordinator can add/view/block batches |
| **4** | Partner request submission and status views | Partner submits request; coordinator sees it |
| **5** | FEFO allocation engine (Config A) | Deterministic allocation passes correctness tests; no over-allocation |
| **6** | Demand forecaster: data prep, model training | NN beats rolling-average baseline on test split (MAE) |
| **7** | Forecast integration + surplus-risk calculation (Config B) | Dashboard shows forecast, projected gaps, at-risk batches |
| **8** | Two LLM agents with tools (Config C) | Analyst produces structured report; Planner produces valid allocation |
| **9** | Approval workflow, pickup tracking, activity log | Full cycle: donate → request → propose → approve → pickup → log |
| **10** | Run A/B/C evaluation, collect metrics | Results table for all metrics across 3 configs |
| **11** | Dashboard polish, reports screen, error handling | AI-unavailable fallback works; all screens functional |
| **12** | Report writing, demo prep, final testing | End-to-end demo of the golden path |

---

## Reuse & Attribution

| Source | What You Take | License | Your Modification |
|---|---|---|---|
| InvAgent | Agent-state prompting pattern, experiment structure | Apache-2.0 | New tools, food-bank domain, constrained tool budget |
| Food Pantry Inventory | Inventory model patterns | MIT | Batch cutoffs, reservations, FEFO engine |
| Pantry for Good | Workflow reference | — | No code reuse; design reference only |
| Full Shelf | Architectural inspiration: human approval, validated changes | — | Narrower, experimentally evaluated implementation |

### Reuse Guidelines

- Preserve applicable license and attribution notices; mark modifications as required.
- Clearly distinguish reused components from original work in the report.
- Follow your college's rules on open-source reuse.

---

## Report Structure

1. **Literature Review:** InvAgent, food-bank inventory systems, FEFO vs FIFO, LLM tool-use
2. **System Design:** Architecture, schema, agent design, tool contracts
3. **ML Methodology:** Feature engineering, model selection, train/test split, hyperparameters
4. **Experiments:** Configs A/B/C results table, forecast accuracy plots, agent accuracy audit
5. **Discussion:** What forecast adds over baseline, what agents add over forecast alone, limitations of simulation
6. **Honest Limitations:** Synthetic data ≠ real food bank; LLM costs; agent latency

---

## Explicit Non-Goals (v1)

- Vehicle routing and live maps
- Image recognition
- Reinforcement learning
- Real-time IoT / cold-chain monitoring
- Mobile application
- Multi-warehouse coordination
- Nutritional analysis
- LLM fine-tuning
- Demand prediction without a suitable dataset
- Autonomous food-safety decisions

---

## Golden Demo Path

```
Donation arrives
    → Request submitted by partner
    → Forecaster flags upcoming shortage
    → Analyst agent confirms with inventory data
    → Planner agent proposes allocation
    → Validator checks constraints
    → Coordinator approves plan
    → Pickup recorded
    → History visible in reports
```

This is the end-to-end workflow that demonstrates all three AI components working together under human oversight.

---

## References

- **InvAgent:** Yinzhu Quan and Zefang Liu, "InvAgent: A Large Language Model based Multi-Agent System for Inventory Management in Supply Chains," 2024. Apache-2.0 license.
- **Food Pantry Inventory:** Django/PostgreSQL warehouse inventory system. MIT license.
- **Pantry for Good:** Food-bank workflow reference (archived 2020).
- **Full Shelf:** Devpost — human approval patterns and operational record design.
- **Synthetic Inventory Forecast:** ARIMA vs LSTM comparison on synthetic data.
- **Hostel Food Waste AI:** Synthetic data generator + forecasting pipeline.
