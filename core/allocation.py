"""FEFO allocation engine for NourishLink AI (Config A baseline).

Rules (documented in the project plan):
  1. Select pending demand in a fixed order: earliest required-by date,
     then submission time.
  2. Match eligible batches: status AVAILABLE, distribution cutoff not
     passed at generation time, unreserved stock > 0.
  3. Allocate from the batch with the EARLIEST cutoff first (FEFO).
  4. Record any unmet quantity — never invent supply.
  5. Draft plans reserve nothing. Approval rechecks everything inside a
     database transaction, then reserves stock.

This module is a transparent heuristic, not an optimiser and not a claim
of fair or optimal distribution.
"""

from django.db import transaction
from django.utils import timezone

from .activity import log_activity
from .models import AllocationItem, AllocationPlan, FoodBatch, Request, RequestItem


def _eligible_batches(product, as_of):
    """Eligible batches for a product, FEFO-ordered."""
    return FoodBatch.objects.filter(
        product=product,
        status='AVAILABLE',
        distribution_cutoff__gte=as_of,
    ).order_by('distribution_cutoff', 'received_date', 'pk')


def _pending_demand(request_ids=None):
    """Requests needing allocation, in documented priority order."""
    qs = (
        Request.objects
        .filter(status__in=['PENDING', 'APPROVED'])
        .prefetch_related('items')
    )
    if request_ids:
        qs = qs.filter(pk__in=request_ids)
    return sorted(qs, key=lambda r: (r.required_by_date, r.created_at))


def generate_plan(user, name=None, request_ids=None):
    """Create a DRAFT allocation plan from unmet demand. Reserves nothing."""
    as_of = timezone.now().date()
    requests = _pending_demand(request_ids)

    plan = AllocationPlan.objects.create(
        name=name or f'Plan {as_of.isoformat()}',
        created_by=user,
        status='DRAFT',
        notes='FEFO draft proposal — awaiting coordinator approval.',
    )

    # Unreserved quantities, cached per batch for this generation run.
    remaining = {}
    n_items = 0
    total_units = 0
    n_unmet = 0

    for req in requests:
        for item in sorted(req.items.all(), key=lambda i: (i.product_id, i.pk)):
            need = item.quantity_requested - item.quantity_allocated
            if need <= 0:
                continue
            for batch in _eligible_batches(item.product, as_of):
                if need <= 0:
                    break
                unres = remaining.get(batch.pk)
                if unres is None:
                    unres = batch.quantity_available - batch.quantity_reserved
                if unres <= 0:
                    continue
                take = min(need, unres)
                AllocationItem.objects.create(
                    plan=plan,
                    request_item=item,
                    food_batch=batch,
                    quantity_allocated=take,
                )
                remaining[batch.pk] = unres - take
                need -= take
                n_items += 1
                total_units += take
            if need > 0:
                n_unmet += 1

    log_activity(
        user, 'PLAN_GENERATED',
        f'Plan "{plan.name}" generated: {n_items} allocation items, '
        f'{total_units} units, {n_unmet} request items with unmet quantity.',
        plan,
    )
    return plan


def validate_plan(plan, as_of=None):
    """Static validation of a plan. Returns {'valid': bool, 'errors': [...]}."""
    as_of = as_of or timezone.now().date()
    errors = []
    items = plan.items.select_related(
        'request_item__request', 'request_item__product', 'food_batch')

    per_batch = {}
    for item in items:
        batch = item.food_batch
        req = item.request_item.request
        if req.status == 'CANCELLED':
            errors.append(
                f'Allocation item {item.pk}: request #{req.pk} is cancelled.')
        if batch.status != 'AVAILABLE':
            errors.append(
                f'Allocation item {item.pk}: batch {batch.batch_code} is '
                f'{batch.status}, not AVAILABLE.')
        if batch.distribution_cutoff < as_of:
            errors.append(
                f'Allocation item {item.pk}: batch {batch.batch_code} cutoff '
                f'({batch.distribution_cutoff}) has passed.')
        per_batch[batch.pk] = per_batch.get(batch.pk, 0) + item.quantity_allocated

    for batch_pk, qty in per_batch.items():
        batch = FoodBatch.objects.get(pk=batch_pk)
        unres = batch.quantity_available - batch.quantity_reserved
        if qty > unres:
            errors.append(
                f'Batch {batch.batch_code}: plan needs {qty} units but only '
                f'{unres} are unreserved.')

    return {'valid': not errors, 'errors': errors}


def recompute_request_status(req, on_complete=False):
    """Derive request status from its items' allocated quantities.

    Pre-pickup:  fully allocated -> APPROVED, partial -> PARTIALLY_FULFILLED,
                 nothing -> PENDING. Post-pickup (on_complete): FULFILLED.
    """
    if req.status == 'CANCELLED':
        return
    items = list(RequestItem.objects.filter(request=req))
    if not items:
        return
    fully = all(i.quantity_allocated >= i.quantity_requested for i in items)
    any_alloc = any(i.quantity_allocated > 0 for i in items)
    if fully and any_alloc:
        new = 'FULFILLED' if on_complete else 'APPROVED'
    elif any_alloc:
        new = 'PARTIALLY_FULFILLED'
    else:
        new = 'PENDING'
    if new != req.status:
        req.status = new
        req.save(update_fields=['status', 'updated_at'])


def approve_plan(plan, user):
    """Validate and approve a DRAFT plan inside a transaction, reserving stock.

    Returns (ok: bool, errors: list[str]). On failure nothing is changed.
    """
    as_of = timezone.now().date()
    with transaction.atomic():
        plan = AllocationPlan.objects.select_for_update().get(pk=plan.pk)
        if plan.status != 'DRAFT':
            return False, [f'Plan is {plan.status}; only DRAFT plans can be approved.']

        items = list(plan.items.select_related(
            'request_item__request', 'food_batch'))
        if not items:
            return False, ['Plan has no allocation items.']

        batch_pks = {it.food_batch_id for it in items}
        batches = {b.pk: b for b in FoodBatch.objects.select_for_update().filter(
            pk__in=batch_pks)}

        per_batch = {}
        for it in items:
            batch = batches[it.food_batch_id]
            req = it.request_item.request
            if req.status == 'CANCELLED':
                return False, [f'Request #{req.pk} was cancelled.']
            if batch.status != 'AVAILABLE':
                return False, [
                    f'Batch {batch.batch_code} is {batch.status}, not AVAILABLE.']
            if batch.distribution_cutoff < as_of:
                return False, [
                    f'Batch {batch.batch_code} cutoff '
                    f'({batch.distribution_cutoff}) has passed.']
            per_batch[batch.pk] = per_batch.get(batch.pk, 0) + it.quantity_allocated

        for batch_pk, qty in per_batch.items():
            batch = batches[batch_pk]
            unres = batch.quantity_available - batch.quantity_reserved
            if qty > unres:
                return False, [
                    f'Batch {batch.batch_code}: plan needs {qty} units but '
                    f'only {unres} are unreserved.']

        # Commit: reserve stock, mark request items allocated.
        for batch_pk, qty in per_batch.items():
            batch = batches[batch_pk]
            batch.quantity_reserved += qty
            batch.save(update_fields=['quantity_reserved', 'updated_at'])
        for it in items:
            ri = it.request_item
            ri.quantity_allocated += it.quantity_allocated
            ri.save(update_fields=['quantity_allocated'])
        plan.status = 'APPROVED'
        plan.approved_by = user
        plan.approved_at = timezone.now()
        plan.save(update_fields=['status', 'approved_by', 'approved_at'])

        affected = {it.request_item.request_id: it.request_item.request
                    for it in items}
        for req in affected.values():
            recompute_request_status(req)

        log_activity(
            user, 'PLAN_APPROVED',
            f'Plan "{plan.name}" approved: {len(items)} items, '
            f'{sum(i.quantity_allocated for i in items)} units reserved.',
            plan,
        )
    return True, []


def cancel_plan(plan, user):
    """Cancel a DRAFT or APPROVED plan. Approved plans release reservations."""
    with transaction.atomic():
        plan = AllocationPlan.objects.select_for_update().get(pk=plan.pk)
        if plan.status not in ('DRAFT', 'APPROVED'):
            return False, [f'Plan is {plan.status}; only DRAFT or APPROVED '
                           f'plans can be cancelled.']

        was_approved = plan.status == 'APPROVED'
        if was_approved:
            items = list(plan.items.select_related('request_item'))
            batch_pks = {it.food_batch_id for it in items}
            batches = {b.pk: b for b in FoodBatch.objects.select_for_update().filter(
                pk__in=batch_pks)}
            for it in items:
                batch = batches[it.food_batch_id]
                batch.quantity_reserved = max(
                    0, batch.quantity_reserved - it.quantity_allocated)
                batch.save(update_fields=['quantity_reserved', 'updated_at'])
                ri = it.request_item
                ri.quantity_allocated = max(
                    0, ri.quantity_allocated - it.quantity_allocated)
                ri.save(update_fields=['quantity_allocated'])

        old = plan.status
        plan.status = 'CANCELLED'
        plan.save(update_fields=['status'])
        log_activity(user, 'PLAN_CANCELLED',
                     f'Plan "{plan.name}" cancelled (was {old}).', plan)

        if was_approved:
            affected = {it.request_item.request_id: it.request_item.request
                        for it in plan.items.select_related('request_item__request')}
            for req in affected.values():
                recompute_request_status(req)
    return True, []


def complete_plan(plan, user):
    """Record pickup completion: reduce physical stock, clear reservations."""
    with transaction.atomic():
        plan = AllocationPlan.objects.select_for_update().get(pk=plan.pk)
        if plan.status != 'APPROVED':
            return False, [f'Plan is {plan.status}; only APPROVED plans can '
                           f'be completed (pickup recorded).']

        items = list(plan.items.select_related('request_item'))
        batch_pks = {it.food_batch_id for it in items}
        batches = {b.pk: b for b in FoodBatch.objects.select_for_update().filter(
            pk__in=batch_pks)}

        for it in items:
            batch = batches[it.food_batch_id]
            if batch.status == 'BLOCKED':
                return False, [
                    f'Batch {batch.batch_code} is BLOCKED — pickup prevented; '
                    f'review affected allocations.']

        for it in items:
            batch = batches[it.food_batch_id]
            batch.quantity_available = max(
                0, batch.quantity_available - it.quantity_allocated)
            batch.quantity_reserved = max(
                0, batch.quantity_reserved - it.quantity_allocated)
            batch.save(update_fields=['quantity_available', 'quantity_reserved',
                                      'updated_at'])
            if batch.quantity_available == 0 and batch.status == 'AVAILABLE':
                batch.status = 'DEPLETED'
                batch.save(update_fields=['status'])

        plan.status = 'COMPLETED'
        plan.save(update_fields=['status'])
        log_activity(
            user, 'PICKUP_COMPLETED',
            f'Pickup completed for plan "{plan.name}": '
            f'{sum(i.quantity_allocated for i in items)} units distributed.',
            plan,
        )

        affected = {it.request_item.request_id: it.request_item.request
                    for it in items}
        for req in affected.values():
            recompute_request_status(req, on_complete=True)
    return True, []
