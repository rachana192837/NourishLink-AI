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
import re
import statistics
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
    # Operational freshness: forecast rows issued up to 3 days ago still
    # describe the current cycle (a row never re-dates itself; the
    # `forecast` command re-issues rows dated at least today).
    fresh_from = today - timedelta(days=3)

    day_rows = {}
    for row in (DemandForecast.objects
                .filter(model_version=mlp_version,
                        forecast_date__gte=fresh_from)
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
            effective = row.forecast_date
            if row.forecast_date > horizon_end:
                cumulative += row.predicted_quantity  # keeps crossing check honest
                continue
            predicted_14 += row.predicted_quantity
            cumulative += row.predicted_quantity
            if cover_by is None and cumulative > stock_free:
                cover_by = max(effective, today)

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
            reason = (f'Forecast demand ({round(predicted_14)} units over '
                      f'the coming cycles) exceeds free stock ({stock_free}).')

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


def _pattern_rows(min_occurrences=3):
    """Full detection: recurring partner-order patterns that need attention now.

    Two row kinds, learned purely from request history (median cycle
    days + median quantity per organisation×product pair):

    kind='unmet' — the partner orders this product on a visible rhythm
        (e.g. every ~4 days) and their newest request is still pending
        with unmet quantity; warehouse stock is thin relative to the
        usual order. Recommendation: approve and let the AI draft a
        scoped allocation for that existing request.
    kind='due'   — the historical cycle is due/overdue (days since the
        last request ≥ 80% of the usual cycle) and there is NO open
        request; proactively create the refill request and draft it.

    Deterministic; nothing is proposed without a human approving it.
    """
    from .models import RequestItem

    today = timezone.now().date()

    pairs = {}
    for item in (RequestItem.objects
                 .exclude(request__status='CANCELLED')
                 .select_related('request__organization', 'product')
                 .order_by('request__created_at')):
        key = (item.request.organization_id, item.product_id)
        entry = pairs.setdefault(key, {
            'org_id': key[0],
            'org_name': item.request.organization.name,
            'product_id': key[1],
            'product_name': item.product.name,
            'dates': [], 'qtys': [], 'unmet_item': None})
        entry['dates'].append(item.request.created_at.date())
        entry['qtys'].append(item.quantity_requested)
        if (item.request.status == 'PENDING'
                and item.quantity_unmet > 0
                and (entry['unmet_item'] is None
                     or item.request.created_at
                     > entry['unmet_item'][0])):
            entry['unmet_item'] = (item.request.created_at,
                                  item.request_id,
                                  item.quantity_unmet)

    product_ids = {key[1] for key in pairs}
    stock = {}
    for product in Product.objects.filter(pk__in=product_ids):
        agg = FoodBatch.objects.filter(
            product=product, status='AVAILABLE',
            distribution_cutoff__gte=today).aggregate(
            available=Sum('quantity_available'),
            reserved=Sum('quantity_reserved'))
        stock[product.pk] = (agg['available'] or 0) - (agg['reserved'] or 0)

    rows = []
    for key, entry in pairs.items():
        if len(entry['dates']) < min_occurrences:
            continue
        dates, qtys = entry['dates'], sorted(entry['qtys'])
        intervals = [(dates[i + 1] - dates[i]).days
                     for i in range(len(dates) - 1)]
        intervals = [d for d in intervals if d > 0]
        if not intervals:
            continue
        cycle_days = max(1, int(statistics.median(intervals)))
        usual_qty = int(statistics.median(qtys))
        days_since = (today - dates[-1]).days
        stock_free = stock.get(key[1], 0)
        cycles_covered = (stock_free // usual_qty if usual_qty else 0)

        base = dict(entry)
        base.update({'cycle_days': cycle_days, 'usual_qty': usual_qty,
                     'days_since': days_since, 'stock_free': stock_free,
                     'cycles_covered': cycles_covered})

        if entry['unmet_item'] is not None and cycle_days <= 31:
            created, request_id, unmet = entry['unmet_item']
            rows.append(dict(base, kind='unmet', request_id=request_id,
                             last_ago=(today - created.date()).days,
                             unmet_qty=unmet))
        elif (entry['unmet_item'] is None
                and days_since >= 0.8 * cycle_days):
            rows.append(dict(base, kind='due', request_id=None,
                             last_sent=dates[-1],
                             due_in=cycle_days - days_since,
                             overdue=max(0, days_since - cycle_days)))

    def urgency(row):
        thin = row['cycles_covered'] <= 2
        return (0 if row['kind'] == 'unmet' and thin
                else 1 if row['kind'] == 'unmet' else 2,
                row['cycles_covered'], row['org_name'])

    rows.sort(key=urgency)
    return rows


def compute_refill_patterns(max_rows=6):
    """Public shortlist: full pattern list + partner-variety caps
    (at most two rows per product so the page shows variety, not six
    flavours of one stock-out)."""
    seen = {}
    selected = []
    for row in _pattern_rows():
        count = seen.get(row['product_id'], 0)
        if count >= 2:
            continue
        seen[row['product_id']] = count + 1
        selected.append(row)
        if len(selected) >= max_rows:
            break
    return selected


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


# Lines that are the model's visible reasoning/meta-commentary, never the
# answer a coordinator should read (reasoning models leak these into the
# content when not tagged cleanly).
_META_LINE = re.compile(
    r"^(the user|user wants|need to include|need to write|let's|lets|okay|"
    r"ok\b|i'll|i will|word count|count words|words:|count:|paragraph:|"
    r"json|prompt|max \d+ words|step \d|analysis:|thinking|note:|"
    r"output:|format:)\b",
    re.IGNORECASE)


def clean_narrative(text, fallback):
    """Return only the coordinator-facing paragraph from an LLM answer.

    1. Drop <think>…</think> reasoning blocks (reasoning models).
    2. Prefer the explicit <FINAL>…</FINAL> block the prompt asks for.
    3. Otherwise strip meta-commentary lines; if too little plausible
       narrative survives, return the deterministic fallback.
    """
    text = (text or '').strip()
    text = re.sub(r'<think>.*?</think>', '', text,
                  flags=re.DOTALL | re.IGNORECASE).strip()
    tagged = re.search(r'<FINAL>\s*(.*?)\s*</FINAL>', text,
                       flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        text = tagged.group(1).strip()
    else:
        kept = [line for line in text.splitlines()
                if line.strip() and not _META_LINE.match(line.strip())]
        text = ' '.join(line.strip() for line in kept).strip()
        # Collapse "Let's count..." style tails glued inside one line.
        text = re.sub(
            r"(Count words?:.*|Word count.*|Let's craft.*|Let's count.*)$",
            '', text, flags=re.IGNORECASE).strip()
    if len(text) < 16 or _META_LINE.match(text) or 'JSON' in text[:80]:
        return fallback
    return text


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
             'Plain, warm, direct. No bullet lists, no invented numbers. '
             'Think silently if you must, then reply with ONLY the final '
             'paragraph wrapped exactly in <FINAL> and </FINAL> tags — '
             'no planning text, no word counts, nothing outside the tags.')},
        {'role': 'user',
         'content': json.dumps(payload, indent=1)},
    ]

    status = AgentRun.Status.OK
    model_name = None
    text = None
    try:
        response = llm_chat(messages, temperature=0.3, max_tokens=800)
        raw = (response.get('content') or '').strip()
        model_name = response.get('model')
        if not raw:
            raise ValueError('empty completion')
        text = clean_narrative(raw, deterministic_brief(rows))
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
