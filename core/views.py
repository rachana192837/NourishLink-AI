from datetime import timedelta
from pathlib import Path

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Count, Q, Sum
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from . import allocation, forecast as forecasting
from .activity import log_activity
from .decorators import coordinator_required, partner_required
from .forms import BatchForm, ProductForm, RequestForm, RequestItemFormSet
from .models import (
    ActivityLog,
    AllocationItem,
    AllocationPlan,
    DemandForecast,
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
        context = {
            'total_batches': FoodBatch.objects.filter(status='AVAILABLE').count(),
            'expiring_soon': FoodBatch.objects.filter(
                status='AVAILABLE',
                distribution_cutoff__lte=today + timedelta(days=7),
                distribution_cutoff__gte=today,
            ).count(),
            'pending_requests': Request.objects.filter(status='PENDING').count(),
            'draft_plans': AllocationPlan.objects.filter(status='DRAFT').count(),
            'forecast_top_gaps': _forecast_product_rows()[:5],
            'forecast_generated': DemandForecast.objects.filter(
                model_version=MLP_VERSION
            ).order_by('-forecast_date').first(),
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
        form = BatchForm(initial={'received_date': timezone.now().date()})
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
    return render(request, 'core/forecast_list.html', {
        'report': report,
        'overall': ov,
        'per_product': (report['per_product'].items()
                        if report else []),
        'rows': product_rows,
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
        'open_backlog_requests': backlog_items.values('request')
                                                .distinct().count(),
        'backlog_unmet': backlog_unmet,
        'page_obj': page,
        'forecast_highlights': _forecast_product_rows()[:5],
    }
    return render(request, 'core/reports.html', context)


# ---------------------------------------------------------------------------
# AI agents (coordinator) — Config C
# ---------------------------------------------------------------------------

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
        messages.success(
            request,
            f'Proposal cycle complete: analyst run #{analyst_run.pk} '
            f'({analyst_run.status}), planner run #{planner_run.pk} '
            f'({planner_run.status}).')
        return redirect('ai_detail', pk=planner_run.pk)

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
