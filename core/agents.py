"""The two LLM agents (Config C) plus their deterministic fallbacks.

Hard constraints (documented in the project plan):
- Analyst: read-only tools only, at most 5 tool calls per invocation.
- Planner: propose + validate tools only, at most 5 tool calls, ONE retry
  after a failed validation.
- Neither agent can modify inventory, approve allocations, or override
  batch blocks — the FEFO engine and the approval transaction enforce that.
- If the LLM is unavailable (no key, rate-limited, provider down), both
  agents fall back to deterministic summaries built from the same tools,
  so the workflow always completes.

Every invocation is persisted as an AgentRun (tool trace, token counts,
latency, model id, status) for the factual-consistency audit and cost
metrics evaluated in the A/B/C comparison.
"""

import json
import time

from . import agent_tools, llm
from .models import AgentRun, AllocationPlan

MAX_TOOL_CALLS = 5
MAX_TOOL_RESULT_CHARS = 16000


def _truncate_result(result):
    text = json.dumps(result, default=str)
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return text[:MAX_TOOL_RESULT_CHARS] + ' ...[truncated]'


def run_tool_loop(messages, specs, tools, client, max_calls=MAX_TOOL_CALLS,
                  planner_mode=False):
    """Agentic loop: LLM requests tools, we execute, feed results back.

    Returns (messages, trace, usage, served_model, final_content).
    Raises llm.LLMUnavailable on the first failed round-trip — the caller
    (not this loop) decides to fall back.
    """
    trace, ptoks, ctoks, model = [], 0, 0, ''
    for _ in range(max_calls):
        response = client.chat(messages, tools=specs)
        model = response['model'] or model
        ptoks += response['prompt_tokens']
        ctoks += response['completion_tokens']

        assistant = {'role': 'assistant', 'content': response['content']}
        if response['tool_calls']:
            assistant['tool_calls'] = response['tool_calls']
        messages.append(assistant)

        if not response['tool_calls']:
            return messages, trace, (ptoks, ctoks), model, response['content']

        for call in response['tool_calls']:
            name = call.get('function', {}).get('name', '')
            raw_args = call.get('function', {}).get('arguments') or '{}'
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            handler = tools.get(name)
            if handler is None:
                result = {'error': f'Unknown tool: {name}'}
            else:
                result = handler(args)
            trace.append({'tool': name, 'args': args,
                          'result': result if isinstance(result, dict)
                          else {'value': result}})
            messages.append({'role': 'tool', 'tool_call_id': call.get('id', ''),
                             'name': name, 'content': _truncate_result(result)})

        if planner_mode and name == 'validate_allocation' and isinstance(result, dict):
            if result.get('valid') is False:
                # Documented planner behaviour: exactly one retry is possible
                # within the remaining budget; the prompt says to stop on a
                # second failure. A soft nudge keeps weak models on track.
                messages.append({'role': 'user', 'content':
                    'Validation reported errors. You may regenerate the '
                    'allocation ONCE with corrected request ids — otherwise '
                    'report the errors in your final answer.'})

    # Tool budget exhausted without a final answer: force one.
    response = client.chat(messages, tools=specs, tool_choice='none')
    ptoks += response['prompt_tokens']
    ctoks += response['completion_tokens']
    messages.append({'role': 'assistant', 'content': response['content']})
    return messages, trace, (ptoks, ctoks), response['model'], response['content']


# ---------------------------------------------------------------------------
# Inventory Analyst
# ---------------------------------------------------------------------------

ANALYST_SYSTEM = """You are the Inventory Analyst for a community food bank.
You answer ONLY from tool data — never invent batch ids, quantities or
product names. Call the read-only tools to ground every claim, then write
a short structured report with these sections:

1. Stock snapshot: total free units, notable batch observations.
2. Projected 7-day demand vs stock: name the products with the most
   negative gap (projected shortages) first, and any large surpluses.
3. Batches at risk: which batches hit their distribution cutoff within
   7 days while units sit undistributed.
4. Pending demand: the oldest required-by requests still unmet.
5. Recommended coordinator focus: at most 3 bullet points.

Be concise, quantitative, and cite the ids you saw in tool results."""

ANALYST_INIT_TASK = """Produce the inventory analyst report for today.
Start by calling get_pending_requests, get_demand_forecast, and
get_batches_at_risk, then summarise."""


def fallback_analyst():
    """Deterministic analyst report built from the same read-only tools."""
    as_of = time.strftime('%Y-%m-%d')
    inventory = agent_tools.get_inventory()
    forecasts = agent_tools.get_demand_forecast()
    at_risk = agent_tools.get_batches_at_risk(7)
    pending = agent_tools.get_pending_requests()

    free_total = sum(b['qty_unreserved'] for b in inventory)
    shortages = sorted((f for f in forecasts if f['gap'] < 0),
                       key=lambda f: f['gap'])
    surplus = sorted((f for f in forecasts if f['gap'] > 0),
                     key=lambda f: -f['gap'])
    risk_units = sum(b['qty_unreserved'] for b in at_risk)

    lines = [f'Inventory analyst report — {as_of} (deterministic fallback, LLM unavailable)',
             '',
             '1. Stock snapshot:',
             f'   - {len(inventory)} available batches, {free_total} free (unreserved) units.']
    for b in at_risk[:5]:
        lines.append(f'   - Batch {b["batch_code"]} ({b["product"]}): '
                     f'{b["qty_unreserved"]} units, cutoff {b["cutoff_date"]}.')
    lines.append('')
    lines.append('2. Projected 7-day demand vs stock (most negative gap first):')
    for f in shortages:
        lines.append(f'   - {f["product"]}: forecast {f["forecast_7d"]}, '
                     f'free stock {f["stock_free"]}, gap {f["gap"]}.')
    if not shortages:
        lines.append('   - No projected shortages.')
    for f in surplus[:3]:
        lines.append(f'   - Surplus: {f["product"]} gap +{f["gap"]}.')
    lines.append('')
    lines.append('3. Batches at risk (cutoff within 7 days):')
    lines.append(f'   - {len(at_risk)} batches holding {risk_units} unreserved units.')
    lines.append('')
    lines.append('4. Pending demand:')
    lines.append(f'   - {pending["n_requests_open"]} open requests'
                 + (f' (showing earliest {pending["listed_requests"]})'
                    if pending['truncated'] else '') + '.')
    if pending['requests']:
        first = pending['requests'][0]
        lines.append(f'   - Oldest: request #{first["request_id"]} '
                     f'({first["organization"]}) required by {first["required_by"]}.')
    lines.append('')
    lines.append('5. Recommended coordinator focus:')
    for f in shortages[:3]:
        lines.append(f'   - Cover the {f["product"]} shortfall before more stock '
                     'hits its cutoff.')

    return {
        'report': '\n'.join(lines),
        'stock_free_units': free_total,
        'shortages': shortages,
        'at_risk_count': len(at_risk),
        'at_risk_units': risk_units,
        'pending_requests_open': pending['n_requests_open'],
    }


def run_analyst(user, client=None):
    """Run the Inventory Analyst once; persist and return the AgentRun."""
    client = client or llm
    started = time.monotonic()
    tools = agent_tools.analyst_tools()
    status, model, error = AgentRun.Status.FALLBACK, '', ''
    usage = (0, 0)
    output = {}
    trace = []
    try:
        messages = [
            {'role': 'system', 'content': ANALYST_SYSTEM},
            {'role': 'user', 'content': ANALYST_INIT_TASK},
        ]
        _, trace, usage, model, report_text = run_tool_loop(
            messages, agent_tools.ANALYST_SPECS, tools, client)
        status = AgentRun.Status.OK
        output = {'report': report_text}
    except llm.LLMUnavailable as exc:
        error = str(exc)
        output = fallback_analyst()

    return AgentRun.objects.create(
        role=AgentRun.Role.ANALYST,
        status=status,
        model_name=model,
        user=user,
        tool_calls=trace,
        output=output,
        prompt_tokens=usage[0],
        completion_tokens=usage[1],
        latency_ms=int((time.monotonic() - started) * 1000),
        error_message=error,
    )


# ---------------------------------------------------------------------------
# Distribution Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the Distribution Planner for a community food bank.
Your job: propose an allocation covering the oldest pending partner
requests using the tools. Rules:
- Call get_pending_requests first; use the ids you actually see.
- Call generate_allocation with those request ids (this runs the
  food bank's deterministic FEFO engine and creates a DRAFT plan —
  nothing is reserved until a human coordinator approves it).
- Call validate_allocation on the returned plan id. If validation
  fails you may regenerate ONCE; after a second failure, report the
  errors instead of trying again.
- Never invent batch ids or quantities: quote only what the tools
  returned.
Finish with a short proposal summary: plan id, number of items,
total units, unmet quantities, validation result, and one sentence
on why this plan is reasonable."""

PLANNER_INIT_TASK = """Propose an allocation for today's unmet demand.
Use get_pending_requests and cover the earliest required-by requests."""


def fallback_planner(user):
    """Deterministic planner: FEFO over the oldest pending requests."""
    started = time.monotonic()
    pending = agent_tools.get_pending_requests()
    request_ids = [r['request_id'] for r in pending['requests']]
    if not request_ids:
        return AgentRun.objects.create(
            role=AgentRun.Role.PLANNER, status=AgentRun.Status.FALLBACK,
            user=user, tool_calls=[], output={
                'report': 'No pending requests with unmet quantity — '
                          'nothing to propose.'},
            latency_ms=int((time.monotonic() - started) * 1000))

    generated = agent_tools.generate_allocation(user, request_ids)
    validation = agent_tools.validate_allocation(generated['plan_id'])
    plan = AllocationPlan.objects.get(pk=generated['plan_id'])
    items = generated['items']
    total_units = sum(i['qty'] for i in items)
    orgs = sorted({i['organization'] for i in items})

    lines = [f'Allocation proposal — {time.strftime("%Y-%m-%d")} '
             '(deterministic fallback, LLM unavailable)', '',
             f'Plan #{plan.pk} "{plan.name}" (DRAFT — awaiting approval).',
             f'- Covers {plan.items.count()} allocation items, '
             f'{total_units} units for {len(orgs)} organizations.',
             f'- Requests covered: the {len(request_ids)} oldest required-by '
             f'of {pending["n_requests_open"]} open requests.']
    if generated['unmet']:
        lines.append('- Unmet quantities remain for: '
                     + ', '.join(f"{u['product']} ({u['unmet']} units)"
                                 for u in generated['unmet'][:6]) + '.')
    lines.append(f'- Static validation: '
                 f'{"passed" if validation["valid"] else "errors found"}.')
    for err in validation['errors'][:5]:
        lines.append(f'    * {err}')
    lines.append('- Nothing is reserved until a coordinator approves the plan.')

    return AgentRun.objects.create(
        role=AgentRun.Role.PLANNER,
        status=AgentRun.Status.FALLBACK,
        model_name='',
        user=user,
        plan=plan,
        tool_calls=[{'tool': 'get_pending_requests',
                     'args': {}, 'result': {'n_requests_open':
                                            pending['n_requests_open']}},
                    {'tool': 'generate_allocation',
                     'args': {'request_ids': request_ids},
                     'result': {'plan_id': plan.pk,
                                'n_items': len(items)}},
                    {'tool': 'validate_allocation',
                     'args': {'plan_id': plan.pk}, 'result': validation}],
        output={'report': '\n'.join(lines),
                'plan_id': plan.pk,
                'validation': validation},
        latency_ms=int((time.monotonic() - started) * 1000))


def run_planner(user, client=None, request_ids=None):
    """Run the Distribution Planner once; persist and return the AgentRun."""
    client = client or llm
    started = time.monotonic()
    tools = dict(agent_tools.planner_tools(user))
    plan_state = {'plan_id': None}

    original_generate = tools['generate_allocation']

    def generate_and_track(args):
        result = original_generate(args)
        if isinstance(result, dict) and result.get('plan_id'):
            plan_state['plan_id'] = result['plan_id']
        return result

    tools['generate_allocation'] = generate_and_track

    status, model, error = AgentRun.Status.FALLBACK, '', ''
    usage = (0, 0)
    output, trace = {}, []
    task = PLANNER_INIT_TASK
    if request_ids:
        task += (f'\nCover exactly these request ids: '
                 f"{', '.join(str(pk) for pk in request_ids)}.")
    try:
        messages = [
            {'role': 'system', 'content': PLANNER_SYSTEM},
            {'role': 'user', 'content': task},
        ]
        _, trace, usage, model, report_text = run_tool_loop(
            messages, agent_tools.PLANNER_SPECS, tools, client,
            planner_mode=True)
        status = AgentRun.Status.OK
        output = {'report': report_text}
    except llm.LLMUnavailable as exc:
        error = str(exc)
        # Note: if the LLM had already created a draft before the API failed,
        # that draft remains in Allocation plans as a valid FEFO draft.
        fallback_run = fallback_planner(user)
        fallback_run.error_message = error
        fallback_run.save(update_fields=['error_message'])
        return fallback_run

    plan = (AllocationPlan.objects.filter(pk=plan_state['plan_id']).first()
            if plan_state['plan_id'] else None)
    return AgentRun.objects.create(
        role=AgentRun.Role.PLANNER,
        status=status,
        model_name=model,
        user=user,
        plan=plan,
        tool_calls=trace,
        output=output,
        prompt_tokens=usage[0],
        completion_tokens=usage[1],
        latency_ms=int((time.monotonic() - started) * 1000),
        error_message=error,
    )


# ---------------------------------------------------------------------------
# Combined proposal cycle (what the "Run AI proposal" button does)
# ---------------------------------------------------------------------------

def run_proposal_cycle(user, client=None):
    """Analyst first, then planner informed by the analyst's report."""
    analyst_run = run_analyst(user, client=client)
    planner_run = run_planner(user, client=client)
    return analyst_run, planner_run
