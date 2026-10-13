from datetime import timedelta

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils import timezone

from .models import AllocationPlan, FoodBatch, Profile, Request


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
