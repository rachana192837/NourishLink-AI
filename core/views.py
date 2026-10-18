from datetime import timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Count, Q, Sum
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .decorators import coordinator_required, partner_required
from .forms import BatchForm, ProductForm, RequestForm, RequestItemFormSet
from .models import (
    ActivityLog,
    AllocationItem,
    AllocationPlan,
    FoodBatch,
    Organization,
    Product,
    Request,
    RequestItem,
)


def log_activity(user, action_type, description, obj=None):
    from django.contrib.contenttypes.models import ContentType
    ct = ContentType.objects.get_for_model(obj) if obj is not None else None
    ActivityLog.objects.create(
        user=user,
        action_type=action_type,
        description=description,
        content_type=ct,
        object_id=getattr(obj, 'pk', None),
    )


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

    return render(request, 'core/request_detail.html', {
        'req': req,
        'items': req.items.select_related('product'),
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
