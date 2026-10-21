"""Tool layer for the two LLM agents (Config C).

Every agent capability is a real, typed function here. The LLM only
*requests* tools; this module executes them against the same models and
the same FEFO engine the web UI uses, so no agent output can represent
state the database does not have.

Analyst tools (read-only):  get_inventory, get_pending_requests,
                            get_demand_forecast, get_batches_at_risk
Planner tools (effectful):  generate_allocation (creates a DRAFT plan —
                            reserves nothing until approval),
                            validate_allocation (static checks)

Design notes:
- Tool results are plain JSON-serialisable dicts (dates as ISO strings)
  so they can go straight into both LLM messages and AgentRun.tool_calls.
- Tool specs follow the OpenAI function-calling schema, which is also
  what OpenRouter exposes.
"""

from datetime import timedelta

from django.db.models import F, Sum
from django.utils import timezone

from . import allocation
from .models import (
    AllocationPlan,
    DemandForecast,
    FoodBatch,
    Product,
    Request,
    RequestItem,
)

FORECAST_VERSION = 'torch-mlp-v2-residual'


def _d(date_obj):
    return date_obj.isoformat() if date_obj else None


# ---------------------------------------------------------------------------
# Analyst tools (read-only)
# ---------------------------------------------------------------------------

def get_inventory(product_id=None):
    """Available batches with free (unreserved) quantities."""
    qs = (FoodBatch.objects
          .filter(status='AVAILABLE', distribution_cutoff__gte=timezone.now().date())
          .select_related('product')
          .order_by('distribution_cutoff', 'batch_code'))
    if product_id:
        qs = qs.filter(product_id=product_id)
    return [{'batch_id': b.pk, 'batch_code': b.batch_code,
             'product': b.product.name, 'product_id': b.product_id,
             'qty_available': b.quantity_available,
             'qty_unreserved': b.quantity_unreserved,
             'cutoff_date': _d(b.distribution_cutoff)}
            for b in qs]


def get_pending_requests(cap=25):
    """Pending or approved requests with unmet item quantities.

    The simulator leaves a large historical backlog, so this returns the
    earliest-dated `cap` requests plus summary totals for the remainder —
    budgets the LLM context while preserving the true queue size.
    """
    qs = (Request.objects
          .filter(status__in=['PENDING', 'APPROVED'])
          .order_by('required_by_date', 'created_at'))
    n_total = (
        RequestItem.objects
        .filter(request__status__in=['PENDING', 'APPROVED'],
                quantity_requested__gt=F('quantity_allocated'))
        .values('request_id').distinct().count())
    rows, total_unmet = [], 0
    for req in qs[:cap + 50]:
        items = []
        for item in req.items.select_related('product'):
            unmet = item.quantity_requested - item.quantity_allocated
            if unmet > 0:
                items.append({'request_item_id': item.pk,
                              'product': item.product.name,
                              'product_id': item.product_id,
                              'qty_requested': item.quantity_requested,
                              'qty_unmet': unmet})
                if len(rows) < cap:
                    total_unmet += unmet
        if items and len(rows) < cap:
            rows.append({'request_id': req.pk,
                         'organization': req.organization.name,
                         'required_by': _d(req.required_by_date),
                         'status': req.status,
                         'items': items})
    return {
        'n_requests_open': n_total,
        'listed_requests': len(rows),
        'truncated': n_total > len(rows),
        'total_unmet_units_listed': total_unmet,
        'requests': rows,
    }


def get_demand_forecast(product_id=None):
    """Latest 7-day forecast per product, plus free stock and the gap.

    Uses the newest forecast run per product (the stored forecast may be a
    few days old relative to today; the date is included so agents and
    readers can judge staleness).
    """
    today = timezone.now().date()
    latest = (DemandForecast.objects
              .filter(model_version=FORECAST_VERSION)
              .order_by('product_id', '-forecast_date', '-pk'))
    latest_per_product = {}
    for row in latest:
        latest_per_product.setdefault(row.product_id, row)

    rows = []
    for product in Product.objects.all():
        if product_id and product.pk != product_id:
            continue
        forecast_row = latest_per_product.get(product.pk)
        if forecast_row is None:
            continue
        agg = (FoodBatch.objects
               .filter(product=product, status='AVAILABLE',
                       distribution_cutoff__gte=today)
               .aggregate(available_sum=Sum('quantity_available'),
                          reserved_sum=Sum('quantity_reserved')))
        free = (agg['available_sum'] or 0) - (agg['reserved_sum'] or 0)
        gap = free - forecast_row.predicted_quantity
        rows.append({
            'product_id': product.pk,
            'product': product.name,
            'forecast_date': _d(forecast_row.forecast_date),
            'forecast_7d': round(forecast_row.predicted_quantity, 1),
            'stock_free': free,
            'gap': round(gap, 1),
        })
    return rows


def get_batches_at_risk(days=7):
    """Available batches whose distribution cutoff is still in the future
    but within `days` days, holding unreserved units — stock that can
    still be allocated today (passed cutoffs are waste, not risk)."""
    today = timezone.now().date()
    window_end = today + timedelta(days=int(days))
    forecasts = {}
    for row in (DemandForecast.objects
                .filter(model_version=FORECAST_VERSION)
                .order_by('product_id', '-forecast_date', '-pk')):
        forecasts.setdefault(row.product_id, row.predicted_quantity)
    qs = (FoodBatch.objects
          .filter(status='AVAILABLE', distribution_cutoff__gte=today,
                  distribution_cutoff__lte=window_end)
          .select_related('product')
          .order_by('distribution_cutoff', 'batch_code'))
    return [{'batch_id': b.pk, 'batch_code': b.batch_code,
             'product': b.product.name, 'product_id': b.product_id,
             'qty_unreserved': b.quantity_unreserved,
             'cutoff_date': _d(b.distribution_cutoff),
             'forecast_7d': (round(forecasts[b.product_id], 1)
                             if b.product_id in forecasts else None)}
            for b in qs if b.quantity_unreserved > 0]


# ---------------------------------------------------------------------------
# Planner tools (effectful, but approval still reserves stock)
# ---------------------------------------------------------------------------

def generate_allocation(user, request_ids):
    """Run the deterministic FEFO engine on the given pending requests.

    Creates a DRAFT AllocationPlan (reserves nothing). Returns the plan's
    items and any unmet quantities so the LLM can reason about them.
    """
    request_ids = [int(pk) for pk in request_ids]
    found = list(Request.objects.filter(pk__in=request_ids))
    if not found:
        return {'error': 'No matching requests found.'}
    plan = allocation.generate_plan(
        user, name=f'AI proposal {timezone.now().date().isoformat()}',
        request_ids=request_ids)
    items = []
    for item in (plan.items
                 .select_related('request_item__product',
                                 'request_item__request__organization',
                                 'food_batch')):
        items.append({
            'allocation_item_id': item.pk,
            'request_id': item.request_item.request_id,
            'organization': item.request_item.request.organization.name,
            'product': item.request_item.product.name,
            'batch_id': item.food_batch_id,
            'batch_code': item.food_batch.batch_code,
            'qty': item.quantity_allocated,
            'required_by': _d(item.request_item.request.required_by_date),
        })
    unmet = []
    # Unmet is relative to THIS plan's allocation (the engine does not touch
    # request items before approval, so deriving it from RequestItem would
    # overstate the gap).
    allocated_per_item = {}
    for item in plan.items.all():
        allocated_per_item[item.request_item_id] = (
            allocated_per_item.get(item.request_item_id, 0)
            + item.quantity_allocated)
    for item in RequestItem.objects.filter(request_id__in=request_ids):
        covered = allocated_per_item.get(item.pk, 0)
        if item.quantity_requested > covered:
            unmet.append({'request_item_id': item.pk,
                          'product': item.product.name,
                          'unmet': item.quantity_requested - covered})
    return {'plan_id': plan.pk, 'plan_name': plan.name,
            'n_items': len(items), 'items': items, 'unmet': unmet,
            'note': 'Draft plan created — nothing reserved until approval.'}


def validate_allocation(plan_id):
    """Static validation of a draft plan (the same checks approval re-runs)."""
    try:
        plan = AllocationPlan.objects.get(pk=int(plan_id))
    except (AllocationPlan.DoesNotExist, ValueError):
        return {'valid': False, 'errors': ['Unknown plan id.']}
    result = allocation.validate_plan(plan)
    result['plan_id'] = plan.pk
    result['plan_status'] = plan.status
    return result


# ---------------------------------------------------------------------------
# OpenAI-schema tool specs (also served to OpenRouter)
# ---------------------------------------------------------------------------

ANALYST_SPECS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_inventory',
            'description': 'List available food batches with their free '
                           '(unreserved) quantities and distribution cutoffs.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'product_id': {'type': 'integer',
                                   'description': 'Optional: filter to one product id'},
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_pending_requests',
            'description': 'List pending and approved partner requests that '
                           'still have unmet item quantities.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_demand_forecast',
            'description': 'Next-7-day demand forecast per product together '
                           'with current free stock and the gap (stock - forecast). '
                           'Negative gap means a projected shortage.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'product_id': {'type': 'integer',
                                   'description': 'Optional: filter to one product id'},
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_batches_at_risk',
            'description': 'List available batches whose distribution cutoff is '
                           'within the given number of days and that still have '
                           'unreserved units — stock about to become undistributable.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'days': {'type': 'integer',
                             'description': 'Window in days from today (default 7)'},
                },
                'required': [],
            },
        },
    },
]

PLANNER_SPECS = [
    {
        'type': 'function',
        'function': {
            'name': 'generate_allocation',
            'description': 'Generate a DRAFT allocation plan for the given pending '
                           'request ids using the FEFO engine. Nothing is reserved '
                           'until a coordinator approves the plan.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'request_ids': {
                        'type': 'array',
                        'items': {'type': 'integer'},
                        'description': 'Request ids to cover',
                    },
                },
                'required': ['request_ids'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'validate_allocation',
            'description': 'Validate a draft allocation plan: batch availability, '
                           'cutoffs in the future, no over-allocation of unreserved stock.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'plan_id': {'type': 'integer',
                                'description': 'Id returned by generate_allocation'},
                },
                'required': ['plan_id'],
            },
        },
    },
]


def analyst_tools(user=None):
    """Map of tool name -> zero-risk callable (bound user irrelevant)."""
    return {
        'get_inventory': lambda args: get_inventory(args.get('product_id')),
        'get_pending_requests': lambda args: get_pending_requests(),
        'get_demand_forecast': lambda args: get_demand_forecast(
            args.get('product_id')),
        'get_batches_at_risk': lambda args: get_batches_at_risk(
            args.get('days') or 7),
    }


def planner_tools(user):
    """Map of tool name -> callable for the planner (user owns any draft)."""
    return {
        'get_pending_requests': lambda args: get_pending_requests(),
        'generate_allocation': lambda args: generate_allocation(
            user, args.get('request_ids') or []),
        'validate_allocation': lambda args: validate_allocation(
            args.get('plan_id')),
    }
