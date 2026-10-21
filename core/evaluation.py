"""A/B/C evaluation for the final report (Config A vs B vs C).

What each configuration is:
  A — rule-based FEFO only (allocation.generate_plan)
  B — FEFO + forecast-informed demand ordering (forecast_boost=True)
  C — FEFO + agents: the Distribution Planner produces the draft via the
      same engine through its tools (the planner cannot invent stock the
      engine would refuse; correctness comes from the engine either way)

Replay design (publication-reproducible, no stock mutation):
  - Bounded scenario so all three configs are directly comparable: the
    earliest required-by pending requests (by default the same 25 the
    planner tool exposes) matched against the same live stock pool.
  - Drafts reserve nothing, so configs replay sequentially over
    identical state; drafts are deleted after measurement.
  - Metrics per config: allocated/unmet/fulfilment over the subset pool,
    at-risk rescue (units allocated from batches whose distribution
    cutoff is within 7 days), item counts, static-validation outcome.
  - Agent metrics (Config C): tokens/latency/tool-call counts, fallback
    flag, and a factual-consistency audit: every batch/request/
    request-item/plan id appearing in the agent's tool-trace results
    must exist in the database.

Forecast metrics come from the trained model's holdout report
(data/forecast/report.json): NN vs rolling-28 baseline MAE/WAPE wins.

This is a deterministic evaluation of the coded policies on simulated
data; it is not a guarantee about real-world food banks.
"""

from datetime import timedelta
from pathlib import Path

from django.contrib.auth.models import User
from django.db.models import Sum
from django.utils import timezone

from . import agent_tools, allocation, agents
from .forecast import load_report
from .models import (
    AgentRun,
    AllocationPlan,
    FoodBatch,
    Request,
    RequestItem,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def metrics_for(plan, subset_ids):
    """Fulfilment/coverage metrics of one draft plan on the subset pool."""
    today = timezone.now().date()
    cutoffs = dict(FoodBatch.objects.exclude(
        status__in=['BLOCKED', 'EXPIRED'])
        .values_list('pk', 'distribution_cutoff'))

    requested = sum(
        max(0, item.quantity_requested - item.quantity_allocated)
        for item in RequestItem.objects.filter(request_id__in=subset_ids))

    allocated = 0
    at_risk_units = 0
    requests_touched = set()
    for row in plan.items.select_related('request_item__request'):
        allocated += row.quantity_allocated
        row_cutoff = cutoffs.get(row.food_batch_id)
        if row_cutoff is not None and row_cutoff <= today + timedelta(days=7):
            at_risk_units += row.quantity_allocated
        requests_touched.add(row.request_item.request_id)

    return {
        'requested_units_subset': requested,
        'allocated_units': allocated,
        'unmet_units_after_plan': max(0, requested - allocated),
        'fulfilment_rate': (round(allocated / requested, 4)
                            if requested else 0.0),
        'n_plan_items': plan.items.count(),
        'requests_touched': len(requests_touched),
        'at_risk_units_covered': at_risk_units,
    }


# ---------------------------------------------------------------------------
# Config runners (drafts only — callers delete them)
# ---------------------------------------------------------------------------

def run_config_a(user, subset_ids, name='Eval A: FEFO only'):
    plan = allocation.generate_plan(user, name=name, request_ids=subset_ids)
    return {'plan_id': plan.pk,
            'validation': allocation.validate_plan(plan),
            **metrics_for(plan, subset_ids)}, plan


def run_config_b(user, subset_ids, name='Eval B: FEFO + forecast ordering'):
    plan = allocation.generate_plan(
        user, name=name, request_ids=subset_ids, forecast_boost=True)
    return {'plan_id': plan.pk,
            'validation': allocation.validate_plan(plan),
            **metrics_for(plan, subset_ids)}, plan


def run_config_c(user, subset_ids, client=None,
                 name='Eval C: agents + FEFO'):
    analyst_run, planner_run = agents.run_proposal_cycle(
        user, client=client, request_ids=subset_ids)
    fallback = (planner_run.status == AgentRun.Status.FALLBACK
                or analyst_run.status == AgentRun.Status.FALLBACK)
    cost = {
        'agent_mode': 'fallback' if fallback else 'llm',
        'tokens_prompt': planner_run.prompt_tokens + analyst_run.prompt_tokens,
        'tokens_completion': (planner_run.completion_tokens
                              + analyst_run.completion_tokens),
        'latency_ms': analyst_run.latency_ms + planner_run.latency_ms,
        'tool_calls': (len(planner_run.tool_calls)
                       + len(analyst_run.tool_calls)),
    }
    audit = _merge_audit(audit_agent_run(analyst_run),
                         audit_agent_run(planner_run))

    planner_draft = planner_run.plan
    if planner_draft and planner_draft.items.exists():
        plan = AllocationPlan.objects.get(pk=planner_draft.pk)
        result = {'plan_id': plan.pk,
                  'validation': allocation.validate_plan(plan),
                  'agent_cost': cost,
                  'agent_audit': audit,
                  **metrics_for(plan, subset_ids)}
        return result, plan
    return {'plan_id': None,
            'status': 'no allocation needed for the scenario pool',
            'validation': {'valid': True, 'errors': []},
            'agent_cost': cost,
            'agent_audit': audit}, None


# ---------------------------------------------------------------------------
# Agent factual-consistency audit
# ---------------------------------------------------------------------------

_ID_FIELDS = (
    ('batch_id', 'batches'),
    ('request_id', 'requests'),
    ('request_item_id', 'request_items'),
    ('plan_id', 'plans'),
)


def audit_agent_run(agent_run):
    """Factual grounding check: every id in the persisted tool-trace
    results must exist in the database. Returns
    {'checks': n, 'violations': [str], 'passes': bool}.
    """
    pools = {
        'batches': set(FoodBatch.objects.values_list('pk', flat=True)),
        'requests': set(Request.objects.values_list('pk', flat=True)),
        'request_items': set(RequestItem.objects.values_list(
            'pk', flat=True)),
        'plans': set(AllocationPlan.objects.values_list('pk', flat=True)),
    }
    checks = 0
    violations = []

    def scan(value):
        nonlocal checks, violations
        if isinstance(value, dict):
            for field, pool_name in _ID_FIELDS:
                if field in value:
                    checks += 1
                    if value[field] not in pools[pool_name]:
                        violations.append(
                            f"run #{agent_run.pk}: {field}="
                            f"{value[field]} not found in {pool_name}")
            for child in value.values():
                scan(child)
        elif isinstance(value, list):
            for child in value:
                scan(child)

    for entry in (agent_run.tool_calls or []):
        scan(entry.get('result') or {})

    return {'checks': checks,
            'violations': violations,
            'passes': checks > 0 and not violations}


def _merge_audit(*audits):
    merged = {'checks': sum(a['checks'] for a in audits)}
    violations = []
    for audit in audits:
        violations.extend(audit['violations'])
    merged['violations'] = violations
    merged['passes'] = merged['checks'] > 0 and not violations
    return merged


# ---------------------------------------------------------------------------
# Full A/B/C evaluation run
# ---------------------------------------------------------------------------

def evaluate_configs(user_id=None, include_agents=True, client=None):
    """Run all three configs over the identical bounded scenario.

    The scenario is the planner-tool bandwidth subset (the earliest
    required-by pending requests exposed by the tool — 25 by default).
    Draft plans reserve nothing and are deleted after measuring, so the
    live stock state is untouched.
    """
    user = (User.objects.filter(pk=user_id).first() if user_id else
            User.objects.filter(profile__role='COORDINATOR').first()
            or User.objects.filter(is_superuser=True).first())
    started = timezone.now()

    pending = agent_tools.get_pending_requests()
    subset_ids = [r['request_id'] for r in pending['requests']]
    results = {
        'generated_at': started.isoformat(),
        'scenario': {
            'n_requests': len(subset_ids),
            'open_backlog': pending['n_requests_open'],
            'selection': 'earliest required-by pending requests',
        },
        'forecast': _forecast_metrics(),
    }

    result_a, plan_a = run_config_a(user, subset_ids)
    results['config_a'] = result_a
    plan_a.delete()

    result_b, plan_b = run_config_b(user, subset_ids)
    results['config_b'] = result_b
    plan_b.delete()

    if include_agents:
        result_c, plan_c = run_config_c(user, subset_ids, client=client)
        results['config_c'] = result_c
        if plan_c is not None:
            plan_c.delete()

    results['runtime_seconds'] = round(
        (timezone.now() - started).total_seconds(), 1)
    return results


def _forecast_metrics():
    path = Path(__file__).resolve().parent.parent / 'data' / 'forecast' \
        / 'report.json'
    report = load_report(path)
    if not report:
        return None
    overall = report.get('overall', {})
    per_product = report.get('per_product', {})
    return {
        'model_name': report.get('model_name'),
        'baseline_name': report.get('baseline_name'),
        'model_mae': overall.get('model_mae'),
        'baseline_mae': overall.get('baseline_mae'),
        'model_wape': overall.get('model_wape'),
        'baseline_wape': overall.get('baseline_wape'),
        'wins_of': (sum(1 for value in per_product.values()
                        if value.get('model_wins')), len(per_product)),
    }
