import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Count, Q, Sum
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from . import allocation, forecast as forecasting, routing
from . import sourcing as sourcing_engine
from . import llm as llm_module
from .activity import log_activity
from .decorators import coordinator_required, partner_required
from .forms import BatchForm, ProductForm, RequestForm, RequestItemFormSet
from .models import (
    ActivityLog,
    AgentRun,
    AllocationItem,
    AllocationPlan,
    DemandForecast,
    Donor,
    FoodBatch,
    Organization,
    Product,
    Request,
    RequestItem,
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = BASE_DIR / 'data' / 'forecast' / 'report.json'
MLP_VERSION = forecasting.MODEL_NAME     # DemandForecast row version for the NN
BASELINE_VERSION = forecasting.BASELINE_NAME


@login_required
def dashboard(request):
    """Route to role-specific dashboard."""
    if not hasattr(request.user, 'profile'):
        return render(request, 'core/no_profile.html')

    profile = request.user.profile
    today = timezone.now().date()

    if profile.is_coordinator:
        # Chart data: next-4-weeks predicted demand for the 5 products with
        # the most negative forecast gap (predicted demand > free stock).
        today = timezone.localdate()
        demand_rows = []
        for row in _forecast_product_rows()[:5]:
            points = list(
                DemandForecast.objects
                .filter(product=row['product'], model_version=MLP_VERSION,
                        forecast_date__gte=today)
                .order_by('forecast_date')
                .values_list('forecast_date', 'predicted_quantity')[:4])
            if not points:
                continue
            demand_rows.append({
                'product': row['product'].name,
                'stock_free': row['stock_free'],
                'horizon_dates': [str(d) for d, _ in points],
                'weekly': [round(q, 1) for _, q in points],
                'four_week_total': round(sum(q for _, q in points), 1),
                'shortfall': (row['stock_free'] - sum(q for _, q in points)
                              < 0),
            })
        # Impact counters — all-time honest numbers (definitions shared with Reports).
        completed_plans = AllocationPlan.objects.filter(status='COMPLETED')
        impact_units = (AllocationItem.objects.filter(plan__in=completed_plans)
                        .aggregate(total=Sum('quantity_allocated'))['total'] or 0)
        rescued_batches = FoodBatch.objects.filter(status='DEPLETED').count()
        expired_batches = FoodBatch.objects.filter(status='EXPIRED').count()
        if rescued_batches + expired_batches:
            rescue_pct = round(
                100 * rescued_batches / (rescued_batches + expired_batches))
        else:
            rescue_pct = None
        impact_partners = (AllocationItem.objects
                           .filter(plan__in=completed_plans)
                           .values('request_item__request__organization')
                           .distinct().count())

        context = {
            'total_batches': FoodBatch.objects.filter(status='AVAILABLE').count(),
            'expiring_soon': FoodBatch.objects.filter(
                status='AVAILABLE',
                distribution_cutoff__lte=today + timedelta(days=7),
                distribution_cutoff__gte=today,
            ).count(),
            'pending_requests': Request.objects.filter(status='PENDING').count(),
            'draft_plans': AllocationPlan.objects.filter(status='DRAFT').count(),
            'impact_units': impact_units,
            'impact_rescued': rescued_batches,
            'impact_partners': impact_partners,
            'impact_rescue_pct': rescue_pct,
            'forecast_top_gaps': _forecast_product_rows()[:5],
            'forecast_generated': DemandForecast.objects.filter(
                model_version=MLP_VERSION
            ).order_by('-forecast_date').first(),
            'demand_rows': demand_rows,
            'sourcing_top': sourcing_engine.compute_sourcing_rows(
                MLP_VERSION)[:3],
        }
        return render(request, 'core/dashboard_coordinator.html', context)
    else:
        org = profile.organization
        context = {
            'my_requests': Request.objects.filter(organization=org).order_by('-created_at')[:5] if org else [],
            'organization': org,
        }
        return render(request, 'core/dashboard_partner.html', context)


# ---------------------------------------------------------------------------
# Inventory (coordinator)
# ---------------------------------------------------------------------------

@coordinator_required
def inventory_list(request):
    batches = FoodBatch.objects.select_related(
        'product', 'donor').order_by('distribution_cutoff', 'batch_code')

    product_id = request.GET.get('product') or ''
    status = request.GET.get('status') or ''
    expiring = request.GET.get('expiring') or ''
    query = request.GET.get('q') or ''

    if product_id:
        batches = batches.filter(product_id=product_id)
    if status:
        batches = batches.filter(status=status)
    if expiring == '1':
        today = timezone.now().date()
        batches = batches.filter(
            status='AVAILABLE',
            distribution_cutoff__lte=today + timedelta(days=7),
            distribution_cutoff__gte=today,
        )
    if query:
        batches = batches.filter(
            Q(batch_code__icontains=query)
            | Q(product__name__icontains=query)
            | Q(donor__name__icontains=query)
        )

    paginator = Paginator(batches, 25)
    page = paginator.get_page(request.GET.get('page'))

    context = {
        'page_obj': page,
        'total': paginator.count,
        'products': Product.objects.all(),
        'status_choices': FoodBatch.Status.choices,
    }
    if request.headers.get('HX-Request'):
        return render(request, 'core/_batch_table.html', context)
    return render(request, 'core/inventory_list.html', context)


@coordinator_required
def batch_detail(request, pk):
    batch = get_object_or_404(
        FoodBatch.objects.select_related('product', 'donor'), pk=pk)
    allocations = (
        AllocationItem.objects.filter(food_batch=batch)
        .select_related('request_item__product', 'plan')
        .order_by('-plan__created_at')
    )
    return render(request, 'core/batch_detail.html', {
        'batch': batch,
        'allocations': allocations,
    })


@coordinator_required
def batch_toggle_block(request, pk):
    if request.method != 'POST':
        return redirect('inventory')
    batch = get_object_or_404(FoodBatch, pk=pk)

    if batch.status == 'AVAILABLE':
        batch.status = 'BLOCKED'
        batch.save(update_fields=['status', 'updated_at'])
        log_activity(request.user, 'BATCH_BLOCKED',
                     f'Batch {batch.batch_code} blocked.', batch)
        messages.success(request, f'Batch {batch.batch_code} blocked.')
    elif batch.status == 'BLOCKED':
        batch.status = 'AVAILABLE'
        batch.save(update_fields=['status', 'updated_at'])
        log_activity(request.user, 'BATCH_UNBLOCKED',
                     f'Batch {batch.batch_code} unblocked.', batch)
        messages.success(request, f'Batch {batch.batch_code} unblocked.')
    else:
        messages.error(request,
                       'Only AVAILABLE or BLOCKED batches can be toggled.')

    if request.headers.get('HX-Request'):
        batch.refresh_from_db()
        return render(request, 'core/_batch_row.html', {'batch': batch})
    return redirect('batch_detail', pk=pk)


@coordinator_required
def donation_create(request):
    if request.method == 'POST':
        form = BatchForm(request.POST)
        if form.is_valid():
            with transaction.atomic():
                batch = form.save(commit=False)
                batch.quantity_available = batch.quantity_received
                batch.quantity_reserved = 0
                batch.status = 'AVAILABLE'
                batch.save()
                log_activity(
                    request.user, 'DONATION_RECORDED',
                    f'Donation batch {batch.batch_code} recorded: '
                    f'{batch.quantity_received} x {batch.product}.', batch)
            messages.success(request,
                             f'Donation batch {batch.batch_code} recorded.')
            return redirect('inventory')
    else:
        initial = {'received_date': timezone.now().date()}
        try:
            initial['product'] = Product.objects.get(
                pk=int(request.GET.get('product', '')))
        except (ValueError, Product.DoesNotExist):
            pass
        form = BatchForm(initial=initial)
    return render(request, 'core/donation_form.html', {
        'form': form,
        'product_form': ProductForm(),
    })


@coordinator_required
def product_create(request):
    next_url = request.POST.get('next') or request.GET.get('next') or ''
    if request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            product = form.save()
            log_activity(request.user, 'PRODUCT_CREATED',
                         f'Product "{product.name}" created.', product)
            messages.success(request, f'Product "{product.name}" created.')
            return redirect(next_url or 'inventory')
    else:
        form = ProductForm()
    return render(request, 'core/product_form.html', {
        'form': form,
        'next': next_url,
    })


# ---------------------------------------------------------------------------
# Scan-to-import (coordinator): photo of a donation list -> AI draft
# -> coordinator review -> approved products + batches. The AI output is
# a draft only: nothing touches stock until the explicit approval POST.
# ---------------------------------------------------------------------------

SCAN_UPLOAD_DIR = BASE_DIR / 'data' / 'uploads'
SCAN_MAX_BYTES = 5 * 1024 * 1024
SCAN_ALLOWED_MIMES = {
    'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp',
}
SCAN_PROMPT = (
    'This photo shows a food donation list (may be printed or handwritten; '
    'may include product names, pack sizes and quantities). Read every '
    'product line. Reply ONLY with JSON, no markdown fences, in exactly '
    'this shape: {"items": [{"name": <product name>, "quantity": <int '
    'units>, "unit": <pack size or "">}]}. If nothing is readable, reply '
    '{"items": []}.'
)


def _extract_json_object(text):
    """Parse the first JSON object out of an LLM answer (fence-tolerant)."""
    text = (text or '').strip()
    if text.startswith('```'):
        text = text.strip('`')
        text = text[text.find('{'):] if '{' in text else text
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('no JSON object in vision answer')
    return json.loads(text[start:end + 1])


def _save_scan_image(request):
    """Validate + store the uploaded photo. Returns stored filename."""
    upload = request.FILES.get('photo')
    if upload is None:
        raise ValueError('No photo was uploaded.')
    if upload.size > SCAN_MAX_BYTES:
        raise ValueError('Photo is larger than 5 MB.')
    mime = (upload.content_type or '').lower().split(';')[0]
    if mime not in SCAN_ALLOWED_MIMES:
        raise ValueError('Please upload a PNG, JPG or WebP photo.')
    SCAN_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stamp = timezone.now().strftime('%Y%m%d-%H%M%S')
    stored = f'scan-{stamp}{SCAN_ALLOWED_MIMES[mime]}'
    with open(SCAN_UPLOAD_DIR / stored, 'wb') as handle:
        for chunk in upload.chunks():
            handle.write(chunk)
    return stored


@coordinator_required
def scan_import(request):
    """Step 1 (GET): upload/snap form. Step 2 (POST): AI parse -> review
    table. Parsed rows live in the session until approved or discarded.
    """
    if request.method == 'POST':
        if request.POST.get('action') == 'discard':
            request.session.pop('scan_draft', None)
            messages.info(request, 'Scan discarded — nothing imported.')
            return redirect('scan_import')
        try:
            stored = _save_scan_image(request)
        except (ValueError, OSError) as exc:
            messages.error(request, str(exc))
            return redirect('scan_import')

        image_bytes = (SCAN_UPLOAD_DIR / stored).read_bytes()
        mime = request.FILES['photo'].content_type.split(';')[0]
        try:
            response = llm_module.chat_vision(
                SCAN_PROMPT, image_bytes, mime=mime, max_tokens=600)
            items = _extract_json_object(
                response['content']).get('items') or []
        except (llm_module.LLMUnavailable, ValueError, KeyError,
                TypeError, json.JSONDecodeError) as exc:
            items = []
            messages.info(request,
                          'Could not read the photo automatically '
                          f'({str(exc)[:120]}). Enter the rows manually '
                          'below and approve — nothing is lost.')
        rows = []
        for item in items:
            try:
                name = str(item.get('name', '')).strip()
                qty = max(1, int(item.get('quantity') or 1))
            except (TypeError, ValueError):
                continue
            if name:
                rows.append({'name': name[:200], 'qty': qty,
                             'unit': str(item.get('unit') or '')[:50],
                             'category': 'OTHER'})
        request.session['scan_draft'] = {
            'image': stored, 'rows': rows,
            'donor_id': request.POST.get('donor') or '',
            'created': timezone.now().isoformat(),
        }
        return redirect('scan_import')

    draft = request.session.get('scan_draft')
    context = {
        'draft': draft,
        'draft_rows': draft['rows'] if draft else [],
        'image_url': (f'/inventory/scan/img/{draft["image"]}/'
                      if draft else None),
        'donors': Donor.objects.order_by('name'),
        'categories': Product.Category.choices,
        'cutoff_default': timezone.now().date() + timedelta(days=7),
    }
    return render(request, 'core/scan_import.html', context)


@coordinator_required
def scan_approve(request):
    """Approval step: create products + batches for the ticked rows only."""
    if request.method != 'POST':
        return redirect('scan_import')
    draft = request.session.get('scan_draft')
    if not draft:
        messages.info(request, 'No scan in progress — upload a photo first.')
        return redirect('scan_import')

    cutoff_default = timezone.now().date() + timedelta(days=7)
    donor = None
    try:
        donor = Donor.objects.get(pk=int(draft.get('donor_id') or 0))
    except (ValueError, Donor.DoesNotExist):
        donor = None

    created_products = []
    created_batches = []
    day_stamp = timezone.now().strftime('%Y%m%d')
    seq = 1
    include_names = request.POST.getlist('include')
    if not include_names:
        messages.warning(request,
                         'No rows were ticked — nothing imported. '
                         'The scan is kept for another try.')
        return redirect('scan_import')

    with transaction.atomic():
        for idx in include_names:
            try:
                i = int(idx)
                name = request.POST.get(f'name-{i}', '').strip()
                qty = max(1, int(request.POST.get(f'qty-{i}', '1')))
                unit = request.POST.get(f'unit-{i}', '').strip() or '1 pack'
                category = request.POST.get(f'category-{i}', 'OTHER')
                cutoff_raw = request.POST.get(f'cutoff-{i}', '')
            except (TypeError, ValueError):
                continue
            if not name:
                continue
            if category not in [c[0] for c in Product.Category.choices]:
                category = Product.Category.OTHER
            product = (Product.objects
                       .filter(name__iexact=name).first())
            if product is None:
                product = Product.objects.create(
                    name=name, unit=unit, category=category)
                created_products.append(name)
                log_activity(request.user, 'PRODUCT_CREATED',
                             f'Product "{product.name}" created from scan.',
                             product)
            else:
                if unit and product.unit != unit:
                    product.unit = unit
                    product.save(update_fields=['unit'])

            cutoff = timezone.now().date()
            for candidate in (cutoff_raw, str(cutoff_default)):
                try:
                    parsed = datetime.strptime(
                        candidate, '%Y-%m-%d').date()
                    if parsed >= cutoff:
                        cutoff = parsed
                        break
                except ValueError:
                    continue

            batch_code = f'SCAN-{day_stamp}-{seq}'
            while FoodBatch.objects.filter(batch_code=batch_code).exists():
                seq += 1
                batch_code = f'SCAN-{day_stamp}-{seq}'
            seq += 1
            batch = FoodBatch.objects.create(
                product=product, donor=donor, batch_code=batch_code,
                quantity_received=qty, quantity_available=qty,
                quantity_reserved=0, received_date=timezone.now().date(),
                distribution_cutoff=cutoff, status='AVAILABLE',
                notes=f'Imported from scanned photo {draft["image"]}')
            created_batches.append(batch)
            log_activity(request.user, 'DONATION_RECORDED',
                         f'Batch {batch.batch_code} scanned in: '
                         f'{qty} x {product.name}.', batch)

    request.session.pop('scan_draft', None)
    log_activity(request.user, 'SCAN_IMPORT',
                 f'Scan approved: {len(created_batches)} batches '
                 f'({len(created_products)} new products) from '
                 f'{draft["image"]}.')
    messages.success(request,
                     f'Import finished: {len(created_batches)} donation '
                     f'batches ({len(created_products)} new products).')
    return redirect('inventory')


@coordinator_required
def scan_image(request, name):
    """Serve one uploaded scan photo (dev server has no media serving)."""
    import re
    if not re.fullmatch(r'scan-[0-9\-]+\.(png|jpg|webp)', name):
        raise Http404('Unknown scan file')
    path = SCAN_UPLOAD_DIR / name
    if not path.exists():
        raise Http404('Scan photo no longer stored')
    mime = ('image/png' if name.endswith('.png')
            else 'image/jpg' if name.endswith('.jpg')
            else 'image/webp')
    return HttpResponse(path.read_bytes(), content_type=mime)


# ---------------------------------------------------------------------------
# Requests (both roles)
# ---------------------------------------------------------------------------

@coordinator_required
def request_list(request):
    requests_qs = (
        Request.objects.select_related('organization')
        .prefetch_related('items')
        .annotate(item_count=Count('items'),
                  total_qty=Sum('items__quantity_requested'))
        .order_by('required_by_date', 'created_at')
    )
    status = request.GET.get('status') or ''
    org_id = request.GET.get('org') or ''
    if status:
        requests_qs = requests_qs.filter(status=status)
    if org_id:
        requests_qs = requests_qs.filter(organization_id=org_id)

    paginator = Paginator(requests_qs, 20)
    page = paginator.get_page(request.GET.get('page'))

    return render(request, 'core/request_list.html', {
        'page_obj': page,
        'total': paginator.count,
        'status_choices': Request.Status.choices,
        'organizations': Organization.objects.all(),
    })


@partner_required
def my_requests(request):
    org = request.user.profile.organization
    requests_qs = (
        Request.objects.filter(organization=org)
        .prefetch_related('items')
        .order_by('-created_at')
    )
    paginator = Paginator(requests_qs, 20)
    page = paginator.get_page(request.GET.get('page'))
    return render(request, 'core/my_requests.html', {
        'page_obj': page,
        'total': paginator.count,
        'organization': org,
    })


@partner_required
def request_create(request):
    org = request.user.profile.organization
    if org is None:
        messages.error(request,
                       'Your account has no organization linked. Contact the administrator.')
        return redirect('dashboard')

    if request.method == 'POST':
        form = RequestForm(request.POST)
        formset = RequestItemFormSet(request.POST)
        if form.is_valid() and formset.is_valid():
            with transaction.atomic():
                req = form.save(commit=False)
                req.organization = org
                req.requested_by = request.user
                req.status = 'PENDING'
                req.save()
                for item_form in formset:
                    RequestItem.objects.create(
                        request=req,
                        product=item_form.cleaned_data['product'],
                        quantity_requested=item_form.cleaned_data['quantity_requested'],
                    )
                n_items = len(formset)
                log_activity(
                    request.user, 'REQUEST_SUBMITTED',
                    f'Request #{req.pk} submitted by {org.name} '
                    f'({n_items} items, required by '
                    f'{req.required_by_date.isoformat()}).', req)
            messages.success(request, f'Request #{req.pk} submitted.')
            return redirect('request_detail', pk=req.pk)
    else:
        form = RequestForm()
        formset = RequestItemFormSet()
    return render(request, 'core/request_form.html', {
        'form': form,
        'formset': formset,
    })


@login_required
def request_detail(request, pk):
    req = get_object_or_404(
        Request.objects.select_related('organization', 'requested_by'), pk=pk)
    profile = request.user.profile

    if profile.is_partner and req.organization_id != profile.organization_id:
        messages.error(request,
                       "You can only view your organization's requests.")
        return redirect('my_requests')

    allocations = None
    if profile.is_partner:
        allocations = (
            AllocationItem.objects.filter(request_item__request=req)
            .select_related('food_batch__product', 'plan')
            .order_by('plan__created_at')
        )

    return render(request, 'core/request_detail.html', {
        'req': req,
        'items': req.items.select_related('product'),
        'allocations': allocations,
        'can_update': profile.is_coordinator,
        'can_cancel': (
            profile.is_partner
            and req.organization_id == profile.organization_id
            and req.status == 'PENDING'
        ),
    })


@coordinator_required
def request_status_update(request, pk):
    if request.method != 'POST':
        return redirect('request_detail', pk=pk)
    req = get_object_or_404(Request, pk=pk)

    allowed = {
        'approve': ('PENDING', 'APPROVED'),
        'cancel': ('PENDING', 'APPROVED'),
        'reopen': ('CANCELLED',),
    }
    target = {'approve': 'APPROVED', 'cancel': 'CANCELLED',
              'reopen': 'PENDING'}
    action = request.POST.get('action') or ''

    if action not in allowed or req.status not in allowed[action]:
        messages.error(request, 'Invalid status change for this request.')
    else:
        old = req.status
        req.status = target[action]
        req.save(update_fields=['status', 'updated_at'])
        log_activity(request.user, 'REQUEST_STATUS_CHANGED',
                     f'Request #{req.pk} status: {old} -> {req.status}.', req)
        messages.success(request,
                         f'Request #{req.pk} is now {req.get_status_display()}.')

    if request.headers.get('HX-Request'):
        return render(request, 'core/_request_status.html',
                      {'req': req, 'can_update': True})
    return redirect('request_detail', pk=pk)


@partner_required
def request_cancel(request, pk):
    if request.method != 'POST':
        return redirect('request_detail', pk=pk)
    req = get_object_or_404(Request, pk=pk)
    if req.organization_id != request.user.profile.organization_id:
        messages.error(request,
                       "You can only cancel your organization's requests.")
        return redirect('my_requests')
    if req.status != 'PENDING':
        messages.error(request, 'Only pending requests can be cancelled.')
        return redirect('request_detail', pk=pk)

    req.status = 'CANCELLED'
    req.save(update_fields=['status', 'updated_at'])
    log_activity(request.user, 'REQUEST_CANCELLED',
                 f'Request #{req.pk} cancelled by partner.', req)
    messages.success(request, f'Request #{req.pk} cancelled.')
    return redirect('my_requests')

# ---------------------------------------------------------------------------
# Allocation plans (coordinator)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sourcing recommendations (coordinator) — "you will run out, source this"
# ---------------------------------------------------------------------------

@coordinator_required
def sourcing(request):
    """Visible sourcing recommendations: products whose forecast demand
    outruns free stock, with cover-by dates, exposed partners, an action
    brief (cached LLM narrative or deterministic), and next-step links.
    """
    rows = sourcing_engine.compute_sourcing_rows(MLP_VERSION)
    cached = sourcing_engine.load_cached_brief(rows)
    context = {
        'rows': rows,
        'brief': (cached or {}).get('brief') or sourcing_engine.deterministic_brief(rows),
        'brief_status': (cached or {}).get('status', 'FALLBACK'),
        'brief_model': (cached or {}).get('model', ''),
        'brief_is_llm': bool(cached and cached.get('status') == 'OK'),
        'horizon_days': sourcing_engine.HORIZON_DAYS,
    }
    return render(request, 'core/sourcing.html', context)


@coordinator_required
def sourcing_explain(request):
    """Re-run the analyst narrative pass over the current rows."""
    if request.method != 'POST':
        return redirect('sourcing')
    rows = sourcing_engine.compute_sourcing_rows(MLP_VERSION)
    _text, _model, status = sourcing_engine.phase_brief_with_llm(
        rows, request.user)
    if status == 'OK':
        messages.success(request, 'AI brief generated for the current '
                                  'recommendations.')
    else:
        messages.info(request, 'LLM unavailable — showing the deterministic '
                               'brief instead.')
    return redirect('sourcing')


@coordinator_required
def plan_list(request):
    plans = (
        AllocationPlan.objects
        .annotate(item_count=Count('items'),
                  total_qty=Sum('items__quantity_allocated'))
        .order_by('-created_at')
    )
    status = request.GET.get('status') or ''
    if status:
        plans = plans.filter(status=status)

    paginator = Paginator(plans, 20)
    page = paginator.get_page(request.GET.get('page'))
    return render(request, 'core/plan_list.html', {
        'page_obj': page,
        'total': paginator.count,
        'status_choices': AllocationPlan.Status.choices,
        'heading': 'Allocation Plans',
        'subtitle': 'FEFO proposals, approvals, and history.',
    })


@coordinator_required
def pickups(request):
    plans = (
        AllocationPlan.objects.filter(status__in=['APPROVED', 'COMPLETED'])
        .annotate(item_count=Count('items'),
                  total_qty=Sum('items__quantity_allocated'))
        .order_by('-created_at')
    )
    status = request.GET.get('status') or ''
    if status:
        plans = plans.filter(status=status)

    paginator = Paginator(plans, 20)
    page = paginator.get_page(request.GET.get('page'))
    return render(request, 'core/plan_list.html', {
        'page_obj': page,
        'total': paginator.count,
        'status_choices': [('APPROVED', 'Approved'), ('COMPLETED', 'Completed')],
        'heading': 'Pickups',
        'subtitle': 'Approved plans awaiting pickup and completed pickups.',
    })


@coordinator_required
def plan_detail(request, pk):
    plan = get_object_or_404(
        AllocationPlan.objects.select_related('created_by', 'approved_by'),
        pk=pk)
    items = plan.items.select_related(
        'request_item__product',
        'request_item__request__organization',
        'food_batch__product',
    )
    validation = allocation.validate_plan(plan) if plan.status == 'DRAFT' else None
    return render(request, 'core/plan_detail.html', {
        'plan': plan,
        'items': items,
        'validation': validation,
    })


@coordinator_required
def plan_generate(request):
    if request.method != 'POST':
        return redirect('plan_list')
    plan = allocation.generate_plan(request.user)
    if not plan.items.exists():
        plan.delete()
        messages.info(request,
                      'No unmet demand to allocate — every pending request is '
                      'already fully covered.')
        return redirect('plan_list')
    messages.success(request, f'Plan "{plan.name}" generated (draft).')
    return redirect('plan_detail', pk=plan.pk)


@coordinator_required
def plan_approve(request, pk):
    if request.method != 'POST':
        return redirect('plan_detail', pk=pk)
    plan = get_object_or_404(AllocationPlan, pk=pk)
    ok, errors = allocation.approve_plan(plan, request.user)
    if ok:
        messages.success(request, f'Plan "{plan.name}" approved — stock reserved.')
    else:
        for error in errors:
            messages.error(request, error)
    return redirect('plan_detail', pk=pk)


@coordinator_required
def plan_cancel(request, pk):
    if request.method != 'POST':
        return redirect('plan_detail', pk=pk)
    plan = get_object_or_404(AllocationPlan, pk=pk)
    was_approved = plan.status == 'APPROVED'
    ok, errors = allocation.cancel_plan(plan, request.user)
    if ok:
        messages.success(request, f'Plan "{plan.name}" cancelled'
                         + (' — reservations released.' if was_approved else '.'))
    else:
        for error in errors:
            messages.error(request, error)
    return redirect('plan_detail', pk=pk)


@coordinator_required
def plan_complete(request, pk):
    if request.method != 'POST':
        return redirect('plan_detail', pk=pk)
    plan = get_object_or_404(AllocationPlan, pk=pk)
    ok, errors = allocation.complete_plan(plan, request.user)
    if ok:
        messages.success(request,
                         f'Pickup for "{plan.name}" recorded — stock reduced.')
    else:
        for error in errors:
            messages.error(request, error)
    return redirect('plan_detail', pk=pk)


# ---------------------------------------------------------------------------
# Dispatch board (coordinator) — map, truck manifests, execution record
# ---------------------------------------------------------------------------

@coordinator_required
def plan_dispatch(request, pk):
    """Fulfilment-control-plane style board for one allocation plan:
    map with truck routes, per-truck manifests and the plan's execution
    timeline (activity log + agent runs). Route groups are persisted on
    the plan (computed once per truck count, then a stable record).
    """
    from django.contrib.contenttypes.models import ContentType

    try:
        n_trucks = max(1, min(int(request.GET.get('trucks', 2)), 4))
    except (TypeError, ValueError):
        n_trucks = 2

    plan = get_object_or_404(
        AllocationPlan.objects
        .prefetch_related('items__request_item__product',
                          'items__request_item__request__organization',
                          'items__food_batch'),
        pk=pk)

    # Group allocated units per destination organisation.
    deliveries = {}
    for row in plan.items.all():
        org = row.request_item.request.organization
        entry = deliveries.setdefault(
            org.pk, {'org_id': org.pk, 'org_name': org.name,
                     'lat': org.latitude, 'lng': org.longitude,
                     'units': 0, 'items': []})
        entry['units'] += row.quantity_allocated
        entry['items'].append({'product': row.request_item.product.name,
                               'qty': row.quantity_allocated,
                               'batch_code': row.food_batch.batch_code})

    # Persisted route record: compute per truck count once, then reuse.
    # ?refresh=1 forces recomputation (e.g. after a data correction).
    route_cache = plan.routes if isinstance(plan.routes, dict) else {}
    cache_key = str(n_trucks)
    if request.GET.get('refresh') == '1' or cache_key not in route_cache:
        route_data = routing.group_plan_into_trucks(
            plan, list(deliveries.values()), n_trucks=n_trucks)
        route_cache[cache_key] = route_data
        plan.routes = route_cache
        plan.save(update_fields=['routes'])
    else:
        route_data = route_cache[cache_key]

    total_units = sum(d['units'] for d in deliveries.values())
    total_cases = plan.items.count()
    completed = plan.status == 'COMPLETED'
    cancelled = plan.status == 'CANCELLED'
    delivered_units = total_units if completed else 0

    # KPI chips as in the reference board.
    kpis = [
        {'label': 'Trucks', 'value': n_trucks,
         'hint': 'route groups computed by nearest-neighbour heuristic'},
        {'label': 'Stops', 'value': len(deliveries),
         'hint': 'distinct partner destinations in this plan'},
        {'label': 'Cases', 'value': total_cases,
         'hint': 'allocation line items (batch → charity)'},
        {'label': 'Delivered', 'value': delivered_units,
         'hint': 'units picked up (completed)'},
        {'label': 'Remaining', 'value': max(0, total_units - delivered_units),
         'hint': 'units still to be picked up'},
    ]

    # Execution record: plan lifecycle events + AI agent runs.
    events = []
    for entry in (ActivityLog.objects
                  .filter(content_type=ContentType.objects.get_for_model(plan),
                          object_id=plan.pk)
                  .select_related('user')):
        events.append({'when': entry.created_at,
                       'actor': (entry.user.username if entry.user
                                 else 'system'),
                       'text': entry.description,
                       'kind': entry.action_type})
    for run in AgentRun.objects.filter(plan=plan):
        if run.role == AgentRun.Role.PLANNER:
            text = ('AI planner drafted the dispatch plan'
                    + (' (offline mode)' if run.status != 'OK' else ''))
            kind = 'AGENT_PLANNER'
        else:
            text = ('AI analyst reviewed stock and forecasts'
                    + (' (offline mode)' if run.status != 'OK' else ''))
            kind = 'AGENT_ANALYST'
        events.append({'when': run.created_at,
                       'actor': 'AI assistant',
                       'text': text,
                       'kind': kind})
    events.sort(key=lambda e: e['when'])

    for truck in route_data['trucks']:
        # Feed route geometry to the map as flat JSON-friendly lists.
        truck['point'] = {
            'start': list(route_data['depot']),
            'order': [list(s['pos']) for s in truck['stops']],
        }

    context = {
        'plan': plan,
        'route_data': route_data,
        'kpis': kpis,
        'events': events,
        'n_trucks': n_trucks,
        'cancelled': cancelled,
    }
    return render(request, 'core/plan_dispatch.html', context)


# ---------------------------------------------------------------------------
# Forecasts (coordinator)
# ---------------------------------------------------------------------------

REPORT_PATH = BASE_DIR / 'data' / 'forecast' / 'report.json'


def _forecast_product_rows():
    """Per-product stock vs 7-day NN forecast (latest stored rows)."""
    forecasts = {}
    mlp_rows = DemandForecast.objects.filter(
        model_version=MLP_VERSION).select_related('product')
    base_rows = DemandForecast.objects.filter(
        model_version=BASELINE_VERSION).select_related('product')
    for row in mlp_rows:
        forecasts.setdefault(row.product_id, {})['mlp'] = row
    for row in base_rows:
        forecasts.setdefault(row.product_id, {})['baseline'] = row

    today = timezone.localdate()
    product_rows = []
    for product in Product.objects.all():
        agg = FoodBatch.objects.filter(
            product=product, status='AVAILABLE',
            distribution_cutoff__gte=today).aggregate(
            available=Sum('quantity_available'),
            reserved=Sum('quantity_reserved'))
        available = agg['available'] or 0
        reserved = agg['reserved'] or 0
        entry = forecasts.get(product.pk, {})
        mlp_row = entry.get('mlp')
        base_row = entry.get('baseline')
        predicted = mlp_row.predicted_quantity if mlp_row else None
        baseline = base_row.predicted_quantity if base_row else None
        product_rows.append({
            'product': product,
            'stock_all': available,
            'stock_free': available - reserved,
            'forecast': predicted,
            'baseline': baseline,
            'gap': None if predicted is None else (available - reserved
                                                   - predicted),
            'forecast_date': mlp_row.forecast_date if mlp_row else None,
        })
    product_rows.sort(key=lambda r: (r['gap'] is None, r['gap'] if r['gap']
                                     is not None else 0))
    return product_rows


@coordinator_required
def forecast_list(request):
    report = forecasting.load_report(REPORT_PATH)
    product_rows = _forecast_product_rows()

    ov = report['overall'] if report else None
    gap_rows = [{'name': r['product'].name, 'gap': r['gap']}
                for r in product_rows if r['gap'] is not None]
    return render(request, 'core/forecast_list.html', {
        'report': report,
        'overall': ov,
        'per_product': (report['per_product'].items()
                        if report else []),
        'rows': product_rows,
        'gap_rows': gap_rows,
        'model_version': MLP_VERSION,
        'baseline_name': BASELINE_VERSION,
    })


# ---------------------------------------------------------------------------
# Reports (coordinator)
# ---------------------------------------------------------------------------

@coordinator_required
def reports(request):
    """Operational summary: distributions, waste, backlog, activity."""
    today = timezone.localdate()

    # Distribution totals — completed plans only (approved = distributed).
    completed_plans = AllocationPlan.objects.filter(status='COMPLETED')
    distributed_units = (AllocationItem.objects
                         .filter(plan__in=completed_plans)
                         .aggregate(total=Sum('quantity_allocated'))
                         ['total'] or 0)
    distributed_by_product = sorted(
        AllocationItem.objects
        .filter(plan__in=completed_plans)
        .values('request_item__product__name')
        .annotate(units=Sum('quantity_allocated'),
                  n=Count('pk'))
        .order_by('-units'),
        key=lambda r: r['request_item__product__name'])

    # Recorded waste — past distribution cutoff with undistributed units.
    wasted = (FoodBatch.objects
              .filter(distribution_cutoff__lt=today,
                      quantity_available__gt=0)
              .exclude(status__in=['BLOCKED'])
              .select_related('product', 'donor')
              .order_by('distribution_cutoff'))
    wasted_units = sum(b.quantity_available for b in wasted)

    # Requested vs distributed per product (top products by units).
    requested_rows = (RequestItem.objects
                      .exclude(request__status='CANCELLED')
                      .values('product__name')
                      .annotate(requested=Sum('quantity_requested'),
                                n=Count('pk')))
    distributed_map = {r['request_item__product__name']:
                       (r['units'], r['n'])
                       for r in distributed_by_product}
    demand_rows = []
    for row in requested_rows:
        name = row['product__name']
        distributed, _ = distributed_map.get(name, (0, 0))
        demand_rows.append({
            'product': name,
            'requested': row['requested'] or 0,
            'distributed': distributed,
        })
    demand_rows.sort(key=lambda r: -(r['requested'] + r['distributed']))
    demand_rows = demand_rows[:8]

    # At-risk stock — cut-off within the next 7 days, units still free.
    at_risk = (FoodBatch.objects
               .filter(status='AVAILABLE',
                       distribution_cutoff__gte=today,
                       distribution_cutoff__lte=today + timedelta(days=7)))
    at_risk_units = sum(b.quantity_available - b.quantity_reserved
                        for b in at_risk)
    at_risk_rows = (at_risk.select_related('product', 'donor')
                    .order_by('distribution_cutoff')[:10])

    # Open backlog — pending/pre-approved requests and their unmet units.
    backlog_items = RequestItem.objects.filter(
        request__status__in=['PENDING', 'APPROVED'])
    backlog_requested = sum(i.quantity_requested for i in backlog_items)
    backlog_allocated = sum(i.quantity_allocated for i in backlog_items)
    backlog_unmet = max(0, backlog_requested - backlog_allocated)

    activity = (ActivityLog.objects.select_related('user')
                .order_by('-created_at'))
    paginator = Paginator(activity, 20)
    page = paginator.get_page(request.GET.get('page'))

    context = {
        'distributed_units': distributed_units,
        'distributed_plans': completed_plans.count(),
        'distributed_by_product': distributed_by_product,
        'wasted_count': wasted.count(),
        'wasted_units': wasted_units,
        'wasted_rows': wasted[:10],
        'at_risk_count': at_risk.count(),
        'at_risk_units': at_risk_units,
        'at_risk_rows': at_risk_rows,
        'open_backlog_requests': backlog_items.values('request')
                                                .distinct().count(),
        'backlog_unmet': backlog_unmet,
        'page_obj': page,
        'forecast_highlights': _forecast_product_rows()[:5],
        'demand_rows': demand_rows,
    }
    return render(request, 'core/reports.html', context)


# ---------------------------------------------------------------------------
# Plain-English evaluation summary (A/B/C + forecast quality)
# ---------------------------------------------------------------------------

EVALUATION_PATH = BASE_DIR / 'data' / 'evaluation' / 'results.json'


@coordinator_required
def evaluation_report(request):
    """Render the A/B/C + forecast results as a thesis-ready page.

    Reads the artefacts produced by `manage.py evaluate` and
    `manage.py forecast`; friendsly empty state when not yet generated.
    """
    import json

    results = None
    if EVALUATION_PATH.exists():
        try:
            results = json.loads(
                EVALUATION_PATH.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            results = None

    config_keys = (
        ('config_a', 'A — FEFO engine only', 'Rule-based baseline.'),
        ('config_b', 'B — FEFO + forecast ordering',
         'Requests re-prioritised by predicted shortage.'),
        ('config_c', 'C — AI agents + FEFO',
         'The LLM/fallback agents draft the plan through the same engine.'),
    )
    configs = [
        {'key': key, 'label': label, 'note': note,
         'data': (results.get(key) if results else None) or {}}
        for key, label, note in config_keys
    ]

    forecast = (results or {}).get('forecast') or {}
    scenario = (results or {}).get('scenario') or {}
    agent_cost = ((results.get('config_c') or {}) if results
                  else {}).get('agent_cost') or {}
    agent_audit = ((results.get('config_c') or {}) if results
                   else {}).get('agent_audit') or {}

    forecast_report = forecasting.load_report(REPORT_PATH)

    # Derived, presentable artefacts (avoids template-side arithmetic).
    from datetime import datetime as _dt
    generated_dt = None
    if (results or {}).get('generated_at'):
        try:
            generated_dt = _dt.fromisoformat(results['generated_at'])
        except ValueError:
            generated_dt = None
    improvement_pct = None
    if forecast:
        try:
            improvement_pct = round(
                (1 - forecast['model_mae'] / forecast['baseline_mae']) * 100,
                1)
        except (KeyError, TypeError, ZeroDivisionError):
            improvement_pct = None

    return render(request, 'core/evaluation_report.html', {
        'results': results,
        'configs': configs,
        'forecast': forecast,
        'forecast_report': forecast_report,
        'scenario': scenario,
        'agent_cost': agent_cost,
        'agent_audit': agent_audit,
        'runtime_seconds': (results or {}).get('runtime_seconds'),
        'generated_at': generated_dt,
        'improvement_pct': improvement_pct,
    })


# ---------------------------------------------------------------------------
# AI agents (coordinator) — Config C
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Weekly ledger (coordinator) — donations received vs stock distributed
# ---------------------------------------------------------------------------

def _ledger_data(monday, sunday):
    """Aggregations for one ISO week (monday..sunday, dates).

    Definitions (kept honest on the page itself):
    - Received: batches RECEIVED in the week, grouped by donor.
    - Distributed: allocation items from plans APPROVED in the week that
      are still APPROVED/COMPLETED (later cancellations drop out), which
      mirrors what actually left the warehouse, grouped by partner org.
    - Waste: batches that expired (cutoff passed) in the week with the
      units still on the shelf.
    """
    received = {}
    for batch in (FoodBatch.objects
                  .filter(received_date__range=(monday, sunday))
                  .select_related('donor', 'product')):
        donor = batch.donor.name if batch.donor else 'Unattributed'
        row = received.setdefault(donor, {'name': donor, 'batches': 0,
                                          'units': 0, 'products': {}})
        row['batches'] += 1
        row['units'] += batch.quantity_received
        row['products'][batch.product.name] = (
            row['products'].get(batch.product.name, 0)
            + batch.quantity_received)

    distributed = {}
    for allocation_row in (AllocationItem.objects
                           .filter(plan__approved_at__date__range=(monday, sunday),
                                   plan__status__in=['APPROVED', 'COMPLETED'])
                           .select_related('request_item__product',
                                           'request_item__request__organization')):
        org = allocation_row.request_item.request.organization
        row = distributed.setdefault(org.name, {'name': org.name, 'requests': set(),
                                                'batches': 0, 'units': 0,
                                                'products': {}})
        row['requests'].add(allocation_row.request_item.request_id)
        row['batches'] += 1
        row['units'] += allocation_row.quantity_allocated
        product_name = allocation_row.request_item.product.name
        row['products'][product_name] = (
            row['products'].get(product_name, 0)
            + allocation_row.quantity_allocated)

    waste_batches = (FoodBatch.objects
                     .filter(status='EXPIRED',
                             distribution_cutoff__range=(monday, sunday)))

    def _product_label(products):
        if not products:
            return '—'
        top = sorted(products.items(), key=lambda kv: -kv[1])[:3]
        return ', '.join(f'{name} ×{qty}' for name, qty in top)

    received_rows = sorted(received.values(), key=lambda r: -r['units'])
    for row in received_rows:
        row['top_products'] = _product_label(row['products'])
    distributed_rows = sorted(distributed.values(), key=lambda r: -r['units'])
    for row in distributed_rows:
        row['top_products'] = _product_label(row['products'])

    return {
        'received': received_rows,
        'received_units': sum(r['units'] for r in received_rows),
        'received_batches': sum(r['batches'] for r in received_rows),
        'distributed': [dict(row, requests=len(row['requests']))
                        for row in distributed_rows],
        'distributed_units': sum(r['units'] for r in distributed_rows),
        'served_requests': len({rq for row in distributed.values()
                                for rq in row['requests']}),
        'waste_units': sum(b.quantity_available for b in waste_batches),
        'waste_batches': waste_batches.count(),
    }


def _ledger_week(request):
    """Parse ?w=<week offset> into (monday, sunday, offset)."""
    try:
        week_offset = max(-520, min(int(request.GET.get('w', '0')), 520))
    except (TypeError, ValueError):
        week_offset = 0
    today = timezone.now().date()
    monday = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)
    return monday, monday + timedelta(days=6), week_offset


@coordinator_required
def weekly_ledger(request):
    """One-week accountability ledger: who provided, who received."""
    monday, sunday, week_offset = _ledger_week(request)
    data = _ledger_data(monday, sunday)
    context = {
        'data': data,
        'monday': monday,
        'sunday': sunday,
        'week_offset': week_offset,
        'week_prev': week_offset - 1,
        'week_next': week_offset + 1,
        'week_is_current': week_offset == 0,
        'received_chart_data': {
            'labels': [row['name'] for row in data['received']],
            'units': [row['units'] for row in data['received']],
            'color': '#eb5310',
        },
        'distributed_chart_data': {
            'labels': [row['name'] for row in data['distributed']],
            'units': [row['units'] for row in data['distributed']],
            'color': '#082454',
        },
    }
    return render(request, 'core/weekly_ledger.html', context)


@coordinator_required
def weekly_ledger_csv(request):
    """CSV export of the weekly ledger (one section per flow)."""
    monday, sunday, _week = _ledger_week(request)
    data = _ledger_data(monday, sunday)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = (
        f'attachment; filename=nourishlink-ledger-{monday}.csv')
    writer = csv.writer(response)
    writer.writerow([f'NourishLink weekly ledger {monday} to {sunday}'])
    writer.writerow([])
    writer.writerow(['Received from donors'])
    writer.writerow(['Donor', 'Batches', 'Units received', 'Top products'])
    for row in data['received']:
        writer.writerow([row['name'], row['batches'], row['units'],
                         row['top_products']])
    writer.writerow([])
    writer.writerow(['Distributed to partners'])
    writer.writerow(['Organisation', 'Requests served', 'Allocation lines',
                     'Units distributed', 'Top products'])
    for row in data['distributed']:
        writer.writerow([row['name'], row['requests'], row['batches'],
                         row['units'], row['top_products']])
    writer.writerow([])
    writer.writerow(['Week totals', data['received_units'],
                     data['distributed_units'], data['waste_units']])
    return response


@coordinator_required
def ai_list(request):
    """Recent agent runs plus the run-proposal action.

    The button runs the Inventory Analyst, then the Distribution Planner.
    With an OPENROUTER_API_KEY set this reaches the free OpenRouter tier;
    without one it stays fully functional via the deterministic fallback
    (AgentRun.status FALLBACK).
    """
    import os
    from . import agents as agent_runner
    from .models import AgentRun

    if request.method == 'POST':
        analyst_run, planner_run = agent_runner.run_proposal_cycle(request.user)
        if planner_run.plan:
            messages.success(
                request,
                'AI proposal ready — review the draft plan below, then '
                'approve it to reserve stock.')
            return redirect('plan_detail', pk=planner_run.plan.pk)
        messages.success(
            request,
            'AI proposal ready — nothing to allocate right now, all '
            'pending requests are already covered.')
        return redirect('ai_list')

    has_key = bool(os.environ.get('OPENROUTER_API_KEY', '').strip())
    runs = (AgentRun.objects
            .select_related('user', 'plan__created_by')
            .order_by('-created_at'))
    paginator = Paginator(runs, 15)
    page = paginator.get_page(request.GET.get('page'))
    return render(request, 'core/ai_run_list.html', {
        'page_obj': page,
        'total': paginator.count,
        'has_key': has_key,
        'model_name': os.environ.get('OPENROUTER_MODEL', 'openrouter/free'),
    })


@coordinator_required
def ai_detail(request, pk):
    from .models import AgentRun
    run = get_object_or_404(
        AgentRun.objects.select_related('user', 'plan'), pk=pk)
    return render(request, 'core/ai_run_detail.html', {
        'run': run,
        'validation': (allocation.validate_plan(run.plan)
                       if run.plan and run.plan.status == 'DRAFT' else None),
    })
