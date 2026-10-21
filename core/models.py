import uuid
from django.db import models
from django.contrib.auth.models import User
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.utils import timezone


class Organization(models.Model):
    class OrgType(models.TextChoices):
        SHELTER = 'SHELTER', 'Shelter'
        COMMUNITY_KITCHEN = 'COMMUNITY_KITCHEN', 'Community Kitchen'
        NGO = 'NGO', 'NGO'
        OTHER = 'OTHER', 'Other'

    name = models.CharField(max_length=200)
    org_type = models.CharField(max_length=20, choices=OrgType.choices, default=OrgType.OTHER)
    contact_email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Profile(models.Model):
    class Role(models.TextChoices):
        COORDINATOR = 'COORDINATOR', 'Coordinator'
        PARTNER = 'PARTNER', 'Partner'

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=15, choices=Role.choices)
    organization = models.ForeignKey(Organization, on_delete=models.SET_NULL, null=True, blank=True, related_name='members')

    def __str__(self):
        return f"{self.user.username} ({self.get_role_display()})"

    @property
    def is_coordinator(self):
        return self.role == self.Role.COORDINATOR

    @property
    def is_partner(self):
        return self.role == self.Role.PARTNER


class Donor(models.Model):
    name = models.CharField(max_length=200)
    contact_email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class Product(models.Model):
    class Category(models.TextChoices):
        GRAIN = 'GRAIN', 'Grain & Cereal'
        CANNED = 'CANNED', 'Canned Goods'
        PASTA = 'PASTA', 'Pasta & Noodles'
        LEGUME = 'LEGUME', 'Legumes & Pulses'
        OIL = 'OIL', 'Oil & Condiments'
        SNACK = 'SNACK', 'Snacks & Biscuits'
        BEVERAGE = 'BEVERAGE', 'Beverages'
        OTHER = 'OTHER', 'Other'

    name = models.CharField(max_length=200)
    unit = models.CharField(max_length=50, help_text="e.g., '1kg pack', '400g tin'")
    category = models.CharField(max_length=15, choices=Category.choices, default=Category.OTHER)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.unit})"


class FoodBatch(models.Model):
    class Status(models.TextChoices):
        AVAILABLE = 'AVAILABLE', 'Available'
        BLOCKED = 'BLOCKED', 'Blocked'
        DEPLETED = 'DEPLETED', 'Depleted'
        EXPIRED = 'EXPIRED', 'Expired'

    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='batches')
    donor = models.ForeignKey(Donor, on_delete=models.SET_NULL, null=True, blank=True, related_name='donations')
    batch_code = models.CharField(max_length=50, unique=True)
    quantity_received = models.PositiveIntegerField()
    quantity_available = models.PositiveIntegerField()
    quantity_reserved = models.PositiveIntegerField(default=0)
    received_date = models.DateField()
    distribution_cutoff = models.DateField(help_text="Last date this batch should be distributed")
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.AVAILABLE)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = 'Food batches'
        ordering = ['distribution_cutoff', 'received_date']

    def __str__(self):
        return f"{self.batch_code} - {self.product.name}"

    @property
    def quantity_unreserved(self):
        return self.quantity_available - self.quantity_reserved

    @property
    def is_expired(self):
        return self.distribution_cutoff < timezone.now().date()

    @property
    def days_until_cutoff(self):
        return (self.distribution_cutoff - timezone.now().date()).days


class Request(models.Model):
    class Status(models.TextChoices):
        PENDING = 'PENDING', 'Pending'
        APPROVED = 'APPROVED', 'Approved'
        PARTIALLY_FULFILLED = 'PARTIALLY_FULFILLED', 'Partially Fulfilled'
        FULFILLED = 'FULFILLED', 'Fulfilled'
        CANCELLED = 'CANCELLED', 'Cancelled'

    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='requests')
    requested_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='requests_made')
    required_by_date = models.DateField()
    status = models.CharField(max_length=25, choices=Status.choices, default=Status.PENDING)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['required_by_date', 'created_at']

    def __str__(self):
        return f"Request #{self.pk} by {self.organization.name}"


class RequestItem(models.Model):
    request = models.ForeignKey(Request, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity_requested = models.PositiveIntegerField()
    quantity_allocated = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"{self.product.name} x{self.quantity_requested}"

    @property
    def quantity_unmet(self):
        return self.quantity_requested - self.quantity_allocated


class AllocationPlan(models.Model):
    class Status(models.TextChoices):
        DRAFT = 'DRAFT', 'Draft'
        APPROVED = 'APPROVED', 'Approved'
        COMPLETED = 'COMPLETED', 'Completed'
        CANCELLED = 'CANCELLED', 'Cancelled'

    name = models.CharField(max_length=200)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='plans_created')
    status = models.CharField(max_length=15, choices=Status.choices, default=Status.DRAFT)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    approved_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='plans_approved')

    def __str__(self):
        return f"{self.name} ({self.get_status_display()})"


class AllocationItem(models.Model):
    plan = models.ForeignKey(AllocationPlan, on_delete=models.CASCADE, related_name='items')
    request_item = models.ForeignKey(RequestItem, on_delete=models.CASCADE, related_name='allocations')
    food_batch = models.ForeignKey(FoodBatch, on_delete=models.CASCADE, related_name='allocations')
    quantity_allocated = models.PositiveIntegerField()

    def __str__(self):
        return f"{self.food_batch.batch_code} → {self.request_item.product.name} x{self.quantity_allocated}"


class DemandForecast(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='forecasts')
    forecast_date = models.DateField()
    period_days = models.PositiveIntegerField(default=7)
    predicted_quantity = models.FloatField()
    model_version = models.CharField(max_length=50, default='v1')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-forecast_date']

    def __str__(self):
        return f"{self.product.name} forecast for {self.forecast_date}"


class AgentRun(models.Model):
    """Record of one AI agent invocation (LLM-backed or deterministic fallback).

    Persisted for the factual-consistency audit and cost metrics in the
    evaluation phase; never used to authorise anything itself.
    """

    class Role(models.TextChoices):
        ANALYST = 'ANALYST', 'Inventory Analyst'
        PLANNER = 'PLANNER', 'Distribution Planner'

    class Status(models.TextChoices):
        OK = 'OK', 'LLM responded'
        FALLBACK = 'FALLBACK', 'Deterministic fallback'

    role = models.CharField(max_length=10, choices=Role.choices)
    status = models.CharField(max_length=10, choices=Status.choices)
    model_name = models.CharField(max_length=120, blank=True,
                                  help_text='Model id reported by the provider, blank for fallback')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True,
                             blank=True, related_name='agent_runs')
    plan = models.ForeignKey(AllocationPlan, on_delete=models.SET_NULL,
                             null=True, blank=True, related_name='agent_runs')
    tool_calls = models.JSONField(default=list, blank=True)
    output = models.JSONField(default=dict, blank=True)
    prompt_tokens = models.PositiveIntegerField(default=0)
    completion_tokens = models.PositiveIntegerField(default=0)
    latency_ms = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.get_role_display()} run #{self.pk} ({self.status})"


class ActivityLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    action_type = models.CharField(max_length=50)
    description = models.TextField()
    content_type = models.ForeignKey(ContentType, on_delete=models.SET_NULL, null=True, blank=True)
    object_id = models.PositiveIntegerField(null=True, blank=True)
    content_object = GenericForeignKey('content_type', 'object_id')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.action_type}: {self.description[:50]}"
