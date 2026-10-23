"""Sourcing recommendations for NourishLink AI.

Turns the forecast gap (predicted 14-day demand vs free stock) into an
actionable, visible instruction for the coordinator: which product, how
many units, by when (cover-by date), and which partners are exposed.

Deterministic-first: rows and the fallback brief are computed from DB
data only. The optional LLM "explain" pass (OpenRouter) rephrases the
same numbers into a demo-friendly narrative; any failure degrades to
the deterministic brief and is recorded as a FALLBACK AgentRun.
"""

import hashlib
import json
import math
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.db.models import Sum
from django.utils import timezone

from .llm import chat as llm_chat
from .models import DemandForecast, FoodBatch, Product, RequestItem

BRIEF_PATH = Path(settings.BASE_DIR) / 'data' / 'sourcing' / 'brief.json'
HORIZON_DAYS = 14


def compute_sourcing_rows(mlp_version, top_n=6):
    """Rows for products whose forecast demand outruns free stock.

    Returns list of dicts sorted by urgency:
    {product_id, product_name, stock_free, predicted_14, units_needed,
     cover_by (date|None), partners: [names], partner_count, reason}
    """
    today = timezone.now().date()
    horizon_end = today + timedelta(days=HORIZON_DAYS)

    day_rows = {}
    for row in (DemandForecast.objects
                .filter(model_version=mlp_version,
                        forecast_date__gte=today)
                .order_by('product_id', 'forecast_date')):
        day_rows.setdefault(row.product_id, []).append(row)

    rows = []
    for product in Product.objects.filter(pk__in=day_rows):
        agg = FoodBatch.objects.filter(
            product=product, status='AVAILABLE',
            distribution_cutoff__gte=today).aggregate(
            available=Sum('quantity_available'),
            reserved=Sum('quantity_reserved'))
        stock_free = (agg['available'] or 0) - (agg['reserved'] or 0)

        predicted_14 = 0.0
        cumulative = 0.0
        cover_by = None
        for row in day_rows[product.pk]:
            if row.forecast_date > horizon_end:
                cumulative += row.predicted_quantity  # keeps crossing check honest
                continue
            predicted_14 += row.predicted_quantity
            cumulative += row.predicted_quantity
            if cover_by is None and cumulative > stock_free:
                cover_by = row.forecast_date

        shortfall = predicted_14 - stock_free
        if shortfall < 1:
            continue

        partner_names = _exposed_partners(product, today, horizon_end)
        earliest = (FoodBatch.objects
                    .filter(product=product, status='AVAILABLE',
                            distribution_cutoff__gte=today)
                    .order_by('distribution_cutoff').first())
        if earliest is None:
            reason = 'No usable stock left for this product.'
        elif earliest.days_until_cutoff <= 7:
            reason = (f'Earliest batch {earliest.batch_code} passes its '
                      f'distribution cutoff in {earliest.days_until_cutoff} day(s).')
        else:
            reason = (f'Forecast demand ({round(predicted_14)} units in '
                      f'{HORIZON_DAYS} days) exceeds free stock ({stock_free}).')

        rows.append({
            'product_id': product.pk,
            'product_name': product.name,
            'stock_free': stock_free,
            'predicted_14': round(predicted_14, 1),
            'units_needed': int(math.ceil(shortfall)),
            'cover_by': cover_by,
            'partners': partner_names[:3],
            'partner_count': len(partner_names),
            'reason': reason,
        })

    rows.sort(key=lambda r: (r['cover_by'] or horizon_end + timedelta(days=365),
                             -r['units_needed']))
    return rows[:top_n]


def _exposed_partners(product, today, horizon_end):
    """Orgs with open requests for this product inside the horizon
    (falls back to their recent request history)."""
    near = (RequestItem.objects
            .filter(product=product,
                    request__status__in=['PENDING', 'APPROVED'],
                    request__required_by_date__range=(today, horizon_end))
            .values_list('request__organization__name', flat=True)
            .distinct())
    names = list(near)
    if not names:
        names = list(
            RequestItem.objects
            .filter(product=product,
                    request__created_at__gte=timezone.now() - timedelta(days=30))
            .values_list('request__organization__name', flat=True)
            .distinct()[:3])
    return names


def _fmt_date(day):
    if day is None:
        return 'beyond the 2-week horizon'
    return day.strftime('%b') + f' {day.day}'


def deterministic_brief(rows):
    """Plain-English brief without any API — always available."""
    if not rows:
        return ('No projected shortfalls: free stock covers the next '
                f'{HORIZON_DAYS} days of forecast demand for every product.')
    parts = []
    for row in rows[:4]:
        parts.append(
            f"Source {row['units_needed']} units of {row['product_name']} "
            f"by {_fmt_date(row['cover_by'])} to cover "
            f"{row['partner_count']} partner request(s).")
    return ' '.join(parts) + ' Approved plans and logged donations will reduce these gaps.'


def rows_signature(rows):
    digest = hashlib.sha256(json.dumps(
        [[r['product_name'], r['units_needed'],
          str(r['cover_by'])] for r in rows],
        sort_keys=True).encode()).hexdigest()[:12]
    return digest


def load_cached_brief(rows):
    """Previous LLM brief if it matches the current recommendation set."""
    try:
        payload = json.loads(BRIEF_PATH.read_text(encoding='utf-8'))
    except Exception:
        return None
    if payload.get('signature') != rows_signature(rows):
        return None
    return payload


def phase_brief_with_llm(rows, user):
    """LLM narrative pass over the deterministic rows (button-triggered).

    Returns (text, model_name|None, agentrun_status). Persists the brief
    keyed to the row signature and records an AgentRun (ANALYST).
    """
    from .models import AgentRun

    payload = [{
        'product': r['product_name'],
        'units_to_source': r['units_needed'],
        'run_out_by': str(r['cover_by'] or 'later'),
        'stock_free': r['stock_free'],
        'forecast_14d': r['predicted_14'],
        'exposed_partners': r['partners'],
        'why': r['reason'],
    } for r in rows]

    messages = [
        {'role': 'system',
         'content': (
             'You write sourcing briefs for a community food-bank '
             'coordinator in Bengaluru. Using ONLY the JSON facts given, '
             'write one short urgent paragraph (max 90 words): what to '
             'source, how many units, by when, and who is exposed. '
             'Plain, warm, direct. No bullet lists, no invented numbers.')},
        {'role': 'user',
         'content': json.dumps(payload, indent=1)},
    ]

    status = AgentRun.Status.OK
    model_name = None
    text = None
    try:
        response = llm_chat(messages, temperature=0.3, max_tokens=350)
        text = (response.get('content') or '').strip()
        model_name = response.get('model')
        if not text:
            raise ValueError('empty completion')
    except Exception:
        status = AgentRun.Status.FALLBACK
        model_name = ''
        text = None

    if status == AgentRun.Status.FALLBACK:
        text = deterministic_brief(rows)

    BRIEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRIEF_PATH.write_text(json.dumps({
        'signature': rows_signature(rows),
        'created': timezone.now().isoformat(),
        'brief': text,
        'status': status,
        'model': model_name or '',
    }, indent=1), encoding='utf-8')

    AgentRun.objects.create(
        role=AgentRun.Role.ANALYST,
        status=status,
        model_name=model_name or '',
        user=user,
        tool_calls=[],
        output={'brief': text, 'signature': rows_signature(rows)},
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0,
        error_message='' if status == AgentRun.Status.OK else 'LLM unavailable, deterministic brief shown',
    )
    return text, model_name, status
