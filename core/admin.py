from django.contrib import admin
from .models import (
    Organization, Profile, Donor, Product, FoodBatch,
    Request, RequestItem, AllocationPlan, AllocationItem,
    DemandForecast, ActivityLog,
)


@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ('name', 'org_type', 'contact_email', 'is_active')
    list_filter = ('org_type', 'is_active')
    search_fields = ('name', 'contact_email')


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'organization')
    list_filter = ('role',)
    search_fields = ('user__username', 'user__email')


@admin.register(Donor)
class DonorAdmin(admin.ModelAdmin):
    list_display = ('name', 'contact_email', 'phone', 'created_at')
    search_fields = ('name', 'contact_email')


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'unit', 'category')
    list_filter = ('category',)
    search_fields = ('name',)


@admin.register(FoodBatch)
class FoodBatchAdmin(admin.ModelAdmin):
    list_display = ('batch_code', 'product', 'quantity_available', 'quantity_reserved', 'distribution_cutoff', 'status')
    list_filter = ('status', 'product')
    search_fields = ('batch_code', 'product__name')
    date_hierarchy = 'distribution_cutoff'


class RequestItemInline(admin.TabularInline):
    model = RequestItem
    extra = 1


@admin.register(Request)
class RequestAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'organization', 'required_by_date', 'status', 'created_at')
    list_filter = ('status', 'organization')
    inlines = [RequestItemInline]
    date_hierarchy = 'required_by_date'


class AllocationItemInline(admin.TabularInline):
    model = AllocationItem
    extra = 1


@admin.register(AllocationPlan)
class AllocationPlanAdmin(admin.ModelAdmin):
    list_display = ('name', 'status', 'created_by', 'created_at', 'approved_at')
    list_filter = ('status',)
    inlines = [AllocationItemInline]


@admin.register(DemandForecast)
class DemandForecastAdmin(admin.ModelAdmin):
    list_display = ('product', 'forecast_date', 'period_days', 'predicted_quantity', 'model_version')
    list_filter = ('product', 'model_version')
    date_hierarchy = 'forecast_date'


@admin.register(ActivityLog)
class ActivityLogAdmin(admin.ModelAdmin):
    list_display = ('action_type', 'user', 'description', 'created_at')
    list_filter = ('action_type',)
    search_fields = ('description',)
    date_hierarchy = 'created_at'
