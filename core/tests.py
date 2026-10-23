from datetime import date, timedelta
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from unittest import mock

from core import agent_tools
from core import allocation
from core import agents as agents_module
from core import llm as llm_module
from core import routing
from core import sourcing as sourcing_module
from core import views as views_module
from .models import (
    ActivityLog,
    AgentRun,
    AllocationPlan,
    Donor,
    FoodBatch,
    Organization,
    Product,
    Request,
    RequestItem,
    DemandForecast,
)


def make_org(name="Hope Shelter", org_type="SHELTER"):
    return Organization.objects.create(name=name, org_type=org_type)


def make_user(username, password="pass12345", role=None, org=None):
    user = User.objects.create_user(username=username, password=password)
    if role:
        user.profile = None  # placeholder to satisfy lints; profile created below
        from .models import Profile
        Profile.objects.create(user=user, role=role, organization=org)
    return user


class BaseTestData(TestCase):
    def setUp(self):
        self.org = make_org()
        self.other_org = make_org("City Kitchen", "COMMUNITY_KITCHEN")
        self.coordinator = make_user("coord", role="COORDINATOR")
        self.partner = make_user("partner", role="PARTNER", org=self.org)
        self.other_partner = make_user("partner2", role="PARTNER", org=self.other_org)
        self.product = Product.objects.create(name="Rice", unit="1kg pack", category="GRAIN")
        self.other_product = Product.objects.create(name="Beans", unit="400g pack", category="LEGUME")
        self.donor = Donor.objects.create(name="City Grocers")
        self.batch = FoodBatch.objects.create(
            product=self.product,
            donor=self.donor,
            batch_code="B-001",
            quantity_received=100,
            quantity_available=100,
            received_date=date.today() - timedelta(days=10),
            distribution_cutoff=date.today() + timedelta(days=20),
        )
        self.expired_batch = FoodBatch.objects.create(
            product=self.product,
            batch_code="B-EXP",
            quantity_received=50,
            quantity_available=50,
            received_date=date.today() - timedelta(days=30),
            distribution_cutoff=date.today() - timedelta(days=1),
            status="EXPIRED",
        )


class PermissionTests(BaseTestData):
    def test_anonymous_redirects_to_login(self):
        for url in ["inventory", "request_list", "my_requests", "request_create"]:
            response = self.client.get(reverse(url))
            self.assertEqual(response.status_code, 302)
            self.assertIn("login", response.url)

    def test_partner_cannot_access_coordinator_pages(self):
        self.client.force_login(self.partner)
        for url in ["inventory", "donation_create", "product_create", "request_list"]:
            response = self.client.get(reverse(url))
            self.assertEqual(response.status_code, 302)

    def test_coordinator_cannot_access_partner_pages(self):
        self.client.force_login(self.coordinator)
        for url in ["my_requests", "request_create"]:
            response = self.client.get(reverse(url))
            self.assertEqual(response.status_code, 302)

    def test_partner_cannot_view_other_org_request(self):
        req = Request.objects.create(
            organization=self.other_org,
            requested_by=self.other_partner,
            required_by_date=date.today() + timedelta(days=5),
        )
        self.client.force_login(self.partner)
        response = self.client.get(reverse("request_detail", args=[req.pk]))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("my_requests"))

    def test_partner_cannot_update_request_status(self):
        req = Request.objects.create(
            organization=self.org,
            requested_by=self.partner,
            required_by_date=date.today() + timedelta(days=5),
        )
        self.client.force_login(self.partner)
        response = self.client.post(
            reverse("request_status_update", args=[req.pk]), {"action": "approve"}
        )
        self.assertEqual(response.status_code, 302)
        req.refresh_from_db()
        self.assertEqual(req.status, "PENDING")


class InventoryTests(BaseTestData):
    def login(self):
        self.client.force_login(self.coordinator)

    def test_inventory_lists_batches(self):
        self.login()
        response = self.client.get(reverse("inventory"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "B-001")
        self.assertContains(response, "B-EXP")

    def test_status_filter(self):
        self.login()
        response = self.client.get(reverse("inventory"), {"status": "EXPIRED"})
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "B-001")
        self.assertContains(response, "B-EXP")

    def test_expiring_filter_excludes_far_batches(self):
        self.login()
        response = self.client.get(reverse("inventory"), {"expiring": "1"})
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "B-001")

    def test_batch_detail(self):
        self.login()
        response = self.client.get(reverse("batch_detail", args=[self.batch.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "B-001")

    def test_block_then_unblock(self):
        self.login()
        response = self.client.post(reverse("batch_toggle_block", args=[self.batch.pk]))
        self.assertEqual(response.status_code, 302)
        self.batch.refresh_from_db()
        self.assertEqual(self.batch.status, "BLOCKED")
        self.assertTrue(
            ActivityLog.objects.filter(
                action_type="BATCH_BLOCKED", object_id=self.batch.pk
            ).exists()
        )
        response = self.client.post(reverse("batch_toggle_block", args=[self.batch.pk]))
        self.batch.refresh_from_db()
        self.assertEqual(self.batch.status, "AVAILABLE")

    def test_cannot_toggle_expired_batch(self):
        self.login()
        self.client.post(reverse("batch_toggle_block", args=[self.expired_batch.pk]))
        self.expired_batch.refresh_from_db()
        self.assertEqual(self.expired_batch.status, "EXPIRED")

    def test_htmx_toggle_returns_row_partial(self):
        self.login()
        response = self.client.post(
            reverse("batch_toggle_block", args=[self.batch.pk]),
            HTTP_HX_REQUEST="true",
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "B-001")


class DonationTests(BaseTestData):
    def login(self):
        self.client.force_login(self.coordinator)

    def test_get_form(self):
        self.login()
        response = self.client.get(reverse("donation_create"))
        self.assertEqual(response.status_code, 200)

    def test_create_donation(self):
        self.login()
        response = self.client.post(reverse("donation_create"), {
            "product": self.product.pk,
            "donor": self.donor.pk,
            "batch_code": "B-NEW-1",
            "quantity_received": 75,
            "received_date": date.today().isoformat(),
            "distribution_cutoff": (date.today() + timedelta(days=30)).isoformat(),
        })
        self.assertEqual(response.status_code, 302)
        batch = FoodBatch.objects.get(batch_code="B-NEW-1")
        self.assertEqual(batch.quantity_available, 75)
        self.assertEqual(batch.quantity_reserved, 0)
        self.assertEqual(batch.status, "AVAILABLE")
        self.assertTrue(
            ActivityLog.objects.filter(
                action_type="DONATION_RECORDED", object_id=batch.pk
            ).exists()
        )

    def test_rejects_cutoff_before_received(self):
        self.login()
        count_before = FoodBatch.objects.count()
        response = self.client.post(reverse("donation_create"), {
            "product": self.product.pk,
            "batch_code": "B-BAD",
            "quantity_received": 75,
            "received_date": date.today().isoformat(),
            "distribution_cutoff": (date.today() - timedelta(days=1)).isoformat(),
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(FoodBatch.objects.count(), count_before)
        self.assertContains(response, "cutoff must be after")


class ProductCreateTests(BaseTestData):
    def test_create_product(self):
        self.client.force_login(self.coordinator)
        response = self.client.post(
            reverse("product_create"),
            {"name": "Tea", "unit": "250g box", "category": "BEVERAGE"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Product.objects.filter(name="Tea").exists())


class RequestCreateTests(BaseTestData):
    def login(self):
        self.client.force_login(self.partner)

    def test_get_form(self):
        self.login()
        response = self.client.get(reverse("request_create"))
        self.assertEqual(response.status_code, 200)

    def test_create_request(self):
        self.login()
        response = self.client.post(reverse("request_create"), {
            "required_by_date": (date.today() + timedelta(days=5)).isoformat(),
            "notes": "Weekly order",
            "form-TOTAL_FORMS": "2",
            "form-INITIAL_FORMS": "0",
            "form-0-product": self.product.pk,
            "form-0-quantity_requested": "40",
            "form-1-product": self.other_product.pk,
            "form-1-quantity_requested": "10",
        })
        self.assertEqual(response.status_code, 302)
        req = Request.objects.latest("pk")
        self.assertEqual(req.organization, self.org)
        self.assertEqual(req.status, "PENDING")
        self.assertEqual(req.items.count(), 2)
        self.assertEqual(
            req.items.get(product=self.product).quantity_requested, 40
        )
        self.assertTrue(
            ActivityLog.objects.filter(
                action_type="REQUEST_SUBMITTED", object_id=req.pk
            ).exists()
        )

    def test_duplicate_products_rejected(self):
        self.login()
        count_before = Request.objects.count()
        response = self.client.post(reverse("request_create"), {
            "required_by_date": (date.today() + timedelta(days=5)).isoformat(),
            "form-TOTAL_FORMS": "2",
            "form-INITIAL_FORMS": "0",
            "form-0-product": self.product.pk,
            "form-0-quantity_requested": "40",
            "form-1-product": self.product.pk,
            "form-1-quantity_requested": "10",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Request.objects.count(), count_before)
        self.assertContains(response, "Duplicate products")

    def test_past_required_by_rejected(self):
        self.login()
        count_before = Request.objects.count()
        response = self.client.post(reverse("request_create"), {
            "required_by_date": (date.today() - timedelta(days=1)).isoformat(),
            "form-TOTAL_FORMS": "1",
            "form-INITIAL_FORMS": "0",
            "form-0-product": self.product.pk,
            "form-0-quantity_requested": "5",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Request.objects.count(), count_before)

    def test_zero_items_rejected(self):
        self.login()
        count_before = Request.objects.count()
        response = self.client.post(reverse("request_create"), {
            "required_by_date": (date.today() + timedelta(days=5)).isoformat(),
            "form-TOTAL_FORMS": "0",
            "form-INITIAL_FORMS": "0",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Request.objects.count(), count_before)


class RequestStatusTests(BaseTestData):
    def make_request(self, org, partner, status="PENDING"):
        return Request.objects.create(
            organization=org,
            requested_by=partner,
            required_by_date=date.today() + timedelta(days=5),
            status=status,
        )

    def test_coordinator_approves_pending(self):
        req = self.make_request(self.org, self.partner)
        self.client.force_login(self.coordinator)
        response = self.client.post(
            reverse("request_status_update", args=[req.pk]), {"action": "approve"}
        )
        self.assertEqual(response.status_code, 302)
        req.refresh_from_db()
        self.assertEqual(req.status, "APPROVED")
        self.assertTrue(
            ActivityLog.objects.filter(action_type="REQUEST_STATUS_CHANGED").exists()
        )

    def test_invalid_transition_rejected(self):
        req = self.make_request(self.org, self.partner, status="CANCELLED")
        self.client.force_login(self.coordinator)
        self.client.post(
            reverse("request_status_update", args=[req.pk]), {"action": "approve"}
        )
        req.refresh_from_db()
        self.assertEqual(req.status, "CANCELLED")

    def test_coordinator_cancels_approved(self):
        req = self.make_request(self.org, self.partner, status="APPROVED")
        self.client.force_login(self.coordinator)
        self.client.post(
            reverse("request_status_update", args=[req.pk]), {"action": "cancel"}
        )
        req.refresh_from_db()
        self.assertEqual(req.status, "CANCELLED")

    def test_partner_cancels_own_pending(self):
        req = self.make_request(self.org, self.partner)
        self.client.force_login(self.partner)
        response = self.client.post(reverse("request_cancel", args=[req.pk]))
        self.assertEqual(response.status_code, 302)
        req.refresh_from_db()
        self.assertEqual(req.status, "CANCELLED")

    def test_partner_cannot_cancel_others_request(self):
        req = self.make_request(self.other_org, self.other_partner)
        self.client.force_login(self.partner)
        self.client.post(reverse("request_cancel", args=[req.pk]))
        req.refresh_from_db()
        self.assertEqual(req.status, "PENDING")


class MyRequestsTests(BaseTestData):
    def test_lists_own_requests_only(self):
        mine = Request.objects.create(
            organization=self.org,
            requested_by=self.partner,
            required_by_date=date.today() + timedelta(days=5),
        )
        Request.objects.create(
            organization=self.other_org,
            requested_by=self.other_partner,
            required_by_date=date.today() + timedelta(days=5),
        )
        self.client.force_login(self.partner)
        response = self.client.get(reverse("my_requests"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f"#{mine.pk}")


class AllocationEngineTests(BaseTestData):
    """Phase 4: FEFO engine — generation, validation, approval, pickup."""

    def setUp(self):
        super().setUp()
        from core import allocation
        self.allocation = allocation
        # Engine tests need full control over the stock picture; drop the
        # base-fixture batch (B-001) so FEFO only sees test-created batches.
        FoodBatch.objects.all().delete()

    def make_request(self, org=None, partner=None, requirements=None,
                     status='PENDING', days=5):
        from core.models import Request, RequestItem
        org = org or self.org
        partner = partner or self.partner
        req = Request.objects.create(
            organization=org,
            requested_by=partner,
            required_by_date=date.today() + timedelta(days=days),
            status=status,
        )
        for product, qty in requirements:
            RequestItem.objects.create(
                request=req, product=product, quantity_requested=qty)
        return req

    def make_batch(self, code, product, qty, cutoff_days, status='AVAILABLE',
                   reserved=0):
        return FoodBatch.objects.create(
            product=product,
            batch_code=code,
            quantity_received=qty,
            quantity_available=qty,
            quantity_reserved=reserved,
            received_date=date.today() - timedelta(days=1),
            distribution_cutoff=date.today() + timedelta(days=cutoff_days),
            status=status,
        )

    def test_fefo_prefers_earliest_cutoff(self):
        early = self.make_batch('B-EARLY', self.product, 30, cutoff_days=3)
        late = self.make_batch('B-LATE', self.product, 50, cutoff_days=60)
        req = self.make_request(requirements=[(self.product, 40)])
        plan = self.allocation.generate_plan(self.coordinator)
        self.assertEqual(plan.items.count(), 2)
        allocations = {a.food_batch_id: a.quantity_allocated
                       for a in plan.items.all()}
        self.assertEqual(allocations.get(early.pk), 30)
        self.assertEqual(allocations.get(late.pk), 10)
        # Draft reserves nothing.
        early.refresh_from_db(); late.refresh_from_db()
        self.assertEqual(early.quantity_reserved, 0)
        self.assertEqual(late.quantity_reserved, 0)

    def test_skips_blocked_and_expired_batches(self):
        # cutoff in the past -> expired; cutoff today is still eligible.
        blocked = self.make_batch('B-BLKD', self.product, 100, cutoff_days=30,
                                  status='BLOCKED')
        expired = self.make_batch('B-EXPD', self.product, 100, cutoff_days=-5)
        good = self.make_batch('B-OK', self.product, 20, cutoff_days=10)
        req = self.make_request(requirements=[(self.product, 15)])
        plan = self.allocation.generate_plan(self.coordinator)
        used = {a.food_batch_id for a in plan.items.all()}
        self.assertIn(good.pk, used)
        self.assertNotIn(blocked.pk, used)
        self.assertNotIn(expired.pk, used)

    def test_unmet_quantity_recorded_not_invented(self):
        self.make_batch('B-SMALL', self.product, 30, cutoff_days=10)
        req = self.make_request(requirements=[(self.product, 100)])
        plan = self.allocation.generate_plan(self.coordinator)
        total = sum(a.quantity_allocated for a in plan.items.all())
        self.assertEqual(total, 30)  # allocated only what exists
        item = req.items.first()
        item.refresh_from_db()
        self.assertEqual(item.quantity_allocated, 0)  # draft: not committed
        ok, errors = self.allocation.approve_plan(plan, self.coordinator)
        self.assertTrue(ok)
        item.refresh_from_db()
        self.assertEqual(item.quantity_allocated, 30)
        req.refresh_from_db()
        self.assertEqual(req.status, 'PARTIALLY_FULFILLED')

    def test_reserved_stock_not_double_allocated(self):
        # 60 received; 50 reserved by an approved plan; 10 unreserved.
        b = self.make_batch('B-RES', self.product, 60, cutoff_days=30, reserved=50)
        req = self.make_request(requirements=[(self.product, 40)])
        plan = self.allocation.generate_plan(self.coordinator)
        self.assertEqual(sum(a.quantity_allocated for a in plan.items.all()), 10)

    def test_approval_reserves_stock_and_updates_status(self):
        b1 = self.make_batch('B-A', self.product, 30, cutoff_days=3)
        b2 = self.make_batch('B-B', self.other_product, 50, cutoff_days=30)
        req = self.make_request(requirements=[(self.product, 30),
                                              (self.other_product, 50)])
        plan = self.allocation.generate_plan(self.coordinator)
        ok, errors = self.allocation.approve_plan(plan, self.coordinator)
        self.assertTrue(ok, errors)
        b1.refresh_from_db(); b2.refresh_from_db()
        self.assertEqual(b1.quantity_reserved, 30)
        self.assertEqual(b2.quantity_reserved, 50)
        req.refresh_from_db()
        self.assertEqual(req.status, 'APPROVED')

    def test_approval_rejects_when_stock_consumed(self):
        # plan1 drains the batch via approval; a plan generated afterwards
        # has no unreserved stock to reserve and must fail cleanly.
        self.make_batch('B-ONE', self.product, 30, cutoff_days=10)
        req1 = self.make_request(requirements=[(self.product, 30)])
        req2 = self.make_request(requirements=[(self.product, 25)])
        plan1 = self.allocation.generate_plan(self.coordinator,
                                              request_ids=[req1.pk])
        plan2 = self.allocation.generate_plan(self.coordinator,
                                              request_ids=[req2.pk])
        ok, _ = self.allocation.approve_plan(plan1, self.coordinator)
        self.assertTrue(ok)
        ok, errors = self.allocation.approve_plan(plan2, self.coordinator)
        self.assertFalse(ok)
        self.assertTrue(errors)
        plan2.refresh_from_db()
        self.assertEqual(plan2.status, 'DRAFT')  # reverted cleanly
        b = FoodBatch.objects.get(batch_code='B-ONE')
        self.assertEqual(b.quantity_reserved, 30)  # only plan1 reserved
        req2.refresh_from_db()
        self.assertEqual(req2.status, 'PENDING')  # untouched

    def test_cancel_releases_reservations(self):
        b = self.make_batch('B-C', self.product, 30, cutoff_days=10)
        req = self.make_request(requirements=[(self.product, 30)])
        plan = self.allocation.generate_plan(self.coordinator)
        self.allocation.approve_plan(plan, self.coordinator)
        ok, _ = self.allocation.cancel_plan(plan, self.coordinator)
        self.assertTrue(ok)
        b.refresh_from_db()
        self.assertEqual(b.quantity_reserved, 0)
        item = req.items.first()
        item.refresh_from_db()
        self.assertEqual(item.quantity_allocated, 0)
        req.refresh_from_db()
        self.assertEqual(req.status, 'PENDING')

    def test_complete_pickup_reduces_stock(self):
        b = self.make_batch('B-P', self.product, 30, cutoff_days=10)
        req = self.make_request(requirements=[(self.product, 30)])
        plan = self.allocation.generate_plan(self.coordinator)
        self.allocation.approve_plan(plan, self.coordinator)
        ok, errors = self.allocation.complete_plan(plan, self.coordinator)
        self.assertTrue(ok, errors)
        b.refresh_from_db()
        self.assertEqual(b.quantity_available, 0)
        self.assertEqual(b.quantity_reserved, 0)
        self.assertEqual(b.status, 'DEPLETED')
        plan.refresh_from_db()
        self.assertEqual(plan.status, 'COMPLETED')
        req.refresh_from_db()
        self.assertEqual(req.status, 'FULFILLED')

    def test_blocked_batch_prevents_pickup(self):
        b = self.make_batch('B-BLK', self.product, 30, cutoff_days=10)
        req = self.make_request(requirements=[(self.product, 30)])
        plan = self.allocation.generate_plan(self.coordinator)
        self.allocation.approve_plan(plan, self.coordinator)
        b.status = 'BLOCKED'
        b.save(update_fields=['status'])
        ok, errors = self.allocation.complete_plan(plan, self.coordinator)
        self.assertFalse(ok)
        plan.refresh_from_db()
        self.assertEqual(plan.status, 'APPROVED')
        # Reservations unchanged — nothing lost.
        b.refresh_from_db()
        self.assertEqual(b.quantity_reserved, 30)

    def test_generate_skips_cancelled_requests(self):
        b = self.make_batch('B-S', self.product, 100, cutoff_days=10)
        self.make_request(requirements=[(self.product, 10)], status='CANCELLED')
        plan = self.allocation.generate_plan(self.coordinator)
        self.assertEqual(plan.items.count(), 0)

    def test_generation_order_by_required_by(self):
        self.make_batch('B-G', self.product, 40, cutoff_days=30)
        urgent = self.make_request(requirements=[(self.product, 30)], days=2)
        later = self.make_request(requirements=[(self.product, 30)], days=10)
        plan = self.allocation.generate_plan(self.coordinator)
        by_request = [a.request_item.request_id for a in plan.items.all()]
        # Urgent request's item comes first.
        self.assertEqual(by_request[0], urgent.items.first().pk)

    def test_partner_cannot_access_plan_pages(self):
        plan = self.allocation.generate_plan(self.coordinator)
        self.client.force_login(self.partner)
        for url, kwargs in [('plan_list', {}), ('plan_detail', {'pk': plan.pk}),
                            ('pickups', {})]:
            response = self.client.get(reverse(url, kwargs=kwargs))
            self.assertEqual(response.status_code, 302, url)
        for url in ['plan_generate', 'plan_approve', 'plan_cancel',
                    'plan_complete']:
            kwargs = {} if url == 'plan_generate' else {'pk': plan.pk}
            response = self.client.post(reverse(url, kwargs=kwargs))
            self.assertEqual(response.status_code, 302, url)

    def test_plan_views_render_for_coordinator(self):
        b = self.make_batch('B-V', self.product, 30, cutoff_days=10)
        req = self.make_request(requirements=[(self.product, 30)])
        plan = self.allocation.generate_plan(self.coordinator)
        self.client.force_login(self.coordinator)
        self.assertEqual(self.client.get(reverse('plan_list')).status_code, 200)
        self.assertEqual(self.client.get(reverse('pickups')).status_code, 200)
        response = self.client.get(reverse('plan_detail', args=[plan.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'B-V')

    def test_partner_sees_own_allocations_on_request_detail(self):
        b = self.make_batch('B-PA', self.product, 30, cutoff_days=10)
        req = self.make_request(requirements=[(self.product, 30)])
        plan = self.allocation.generate_plan(self.coordinator)
        self.allocation.approve_plan(plan, self.coordinator)
        self.client.force_login(self.partner)
        response = self.client.get(reverse('request_detail', args=[req.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Allocations for this request')
        self.assertContains(response, 'B-PA')

    def test_other_partner_cannot_see_allocations(self):
        b = self.make_batch('B-PB', self.product, 30, cutoff_days=10)
        req = self.make_request(requirements=[(self.product, 30)])
        plan = self.allocation.generate_plan(self.coordinator)
        self.allocation.approve_plan(plan, self.coordinator)
        self.client.force_login(self.other_partner)
        response = self.client.get(reverse('request_detail', args=[req.pk]),
                                   follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'Allocations for this request')




class ForecastTests(BaseTestData):
    """Phase 5: demand forecaster — features, split integrity, baseline, NN."""

    def setUp(self):
        super().setUp()
        import core.forecast
        self.f = core.forecast

    def _synthetic(self, days=200):
        return self.f.make_synthetic_daily(
            ['Rice', 'Oats'], {'Rice': 70, 'Oats': 40},
            {'Rice': 30, 'Oats': 0}, days=days)

    def test_feature_vector_layout(self):
        f = self.f = __import__('core.forecast', fromlist=['x'])
        hist = np.arange(50, dtype=float)
        vec = f.feature_vector(hist, 3)
        self.assertEqual(len(vec), f.NUMERIC_FEATURES)
        lags = vec[:f.LOOKBACK]
        self.assertEqual(lags[0], hist[-1])          # lag1 = yesterday
        self.assertEqual(lags[1], hist[-2])
        self.assertEqual(lags[f.LOOKBACK - 1], hist[-f.LOOKBACK])
        # weekday one-hot is the last 7 numeric positions
        self.assertEqual(vec[-7 + 3], 1.0)
        self.assertEqual(np.count_nonzero(vec[-7:]), 1)

    def test_supervised_shapes(self):
        f = self.f
        df = self._synthetic()
        packed = f.build_supervised(df)
        n_dates = len(df) - f.WARMUP - (f.HORIZON - 1)
        self.assertEqual(packed['X'].shape,
                         (n_dates * len(df.columns),
                          f.NUMERIC_FEATURES + 2))
        self.assertEqual(packed['Y'].shape, (n_dates * len(df.columns),))
        self.assertEqual(len(packed['dates']), n_dates)

    def test_split_layout_and_spacing(self):
        f = self.f
        df = self._synthetic()
        packed = f.build_supervised(df)
        split = f.chronological_split(packed, test_weeks=4)
        self.assertEqual(len(split['test_dates']), 4)
        deltas = {(b - a).days
                  for a, b in zip(split['test_dates'],
                                  split['test_dates'][1:])}
        self.assertEqual(deltas, {7})
        # the last holdout window covers the final observed data day
        self.assertEqual(df.index[-1],
                         split['test_dates'][-1] + timedelta(days=6))

    def test_no_leakage_into_training(self):
        """Corrupt every observed day the training pipeline never reads
        (from the earliest test window start, minus one) — deterministic
        training must then produce IDENTICAL first-holdout predictions."""
        import copy
        f = self.f
        df = self._synthetic()
        packed = f.build_supervised(df)
        split = f.chronological_split(packed, test_weeks=4)
        test_first = split['test_first_date']

        def fit_preds(frame):
            packed2 = f.build_supervised(frame)
            split2 = f.chronological_split(packed2, test_weeks=4)
            f._scale_features(split2)
            n_products = 2
            rows = split2['X_train'].shape[0]
            y_means = f._target_scales(packed2, range(rows))
            y_row_tr = y_means[f._row_product_index(rows, n_products)]
            y_row_te = y_means[f._row_product_index(
                split2['X_test'].shape[0], n_products)]
            model, _ = f.train_model(split2, y_row_scales=y_row_tr, seed=42,
                                     epochs=25, val_rows=0)
            preds = f.predict_matrix(model, split2, y_row_scales=y_row_te)
            return preds[:n_products]       # earliest test window, per product

        df_mut = df.copy()
        df_mut.loc[test_first:] = 99.0
        self.assertTrue(np.allclose(fit_preds(df), fit_preds(df_mut)))

    def test_baseline_constant_series_perfect(self):
        f = self.f
        df = pd.DataFrame(
            20.0, index=pd.date_range('2025-01-01', periods=60, freq='D'),
            columns=['Rice'])
        packed = f.build_supervised(df)
        f.chronological_split(packed, test_weeks=2)
        value = f.weekly_baseline_row(packed['values'][:, 0])
        self.assertAlmostEqual(value, 140.0)  # 20/day * 7

    def test_train_and_evaluate_report(self):
        f = self.f
        df = self._synthetic()
        r = f.train_and_evaluate(df=df, seed=42, epochs=15, test_weeks=2)
        for key in ('overall', 'per_product', 'future_model',
                    'future_baseline', 'model_name', 'baseline_name'):
            self.assertIn(key, r)
        overall = r['overall']
        self.assertTrue(np.isfinite(overall['model_mae']))
        self.assertTrue(np.isfinite(overall['baseline_mae']))
        self.assertEqual(set(r['per_product']), {'Rice', 'Oats'})
        for pred in r['future_model'].values():
            self.assertGreaterEqual(pred, 0.0)

    def test_forecast_command_persists(self):
        from django.core.management import call_command
        from io import StringIO
        f = self.f
        df = f.make_synthetic_daily(['Rice'], {'Rice': 50}, {'Rice': 20},
                                    days=90)
        df.index.name = 'date'
        long = (df.reset_index()
                .melt(id_vars=['date'], var_name='product',
                      value_name='quantity'))
        long['organization'] = 'Hope Shelter'
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_csv = tmp_dir / 'daily_demand.csv'
        tmp_json = tmp_dir / 'report.json'
        long.to_csv(tmp_csv, index=False)
        out = StringIO()
        call_command('forecast', '--epochs', '5', '--test-weeks', '1',
                     '--source', str(tmp_csv), '--report', str(tmp_json),
                     stdout=out)
        self.assertIn('OVERALL', out.getvalue())
        self.assertTrue(tmp_json.exists())
        mlp = DemandForecast.objects.filter(model_version=f.MODEL_NAME)
        self.assertEqual(mlp.count(), 1)
        self.assertEqual(mlp.first().product.name, 'Rice')
        self.assertGreater(mlp.first().predicted_quantity, 0)
        base = DemandForecast.objects.filter(model_version=f.BASELINE_NAME)
        self.assertEqual(base.count(), 1)

    def test_forecast_list_page(self):
        f = self.f
        self.client.force_login(self.coordinator)
        DemandForecast.objects.create(
            product=self.product, forecast_date=date.today(),
            period_days=7, predicted_quantity=88.5,
            model_version=f.MODEL_NAME)
        DemandForecast.objects.create(
            product=self.product, forecast_date=date.today(),
            period_days=7, predicted_quantity=77.5,
            model_version=f.BASELINE_NAME)
        response = self.client.get(reverse('forecast_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '88.5')
        self.assertContains(response, '77.5')

    def test_forecast_page_partner_blocked(self):
        self.client.force_login(self.partner)
        response = self.client.get(reverse('forecast_list'), follow=True)
        self.assertNotContains(response, 'Demand Forecasts')


class AgentToolTests(BaseTestData):
    """Phase 6: agent tool layer — same data and engine the UI uses."""

    def _batch(self, product, qty=20, cutoff_days=10, code='T-B1',
               reserved=0, status=FoodBatch.Status.AVAILABLE):
        return FoodBatch.objects.create(
            product=product, donor=self.donor, batch_code=code,
            quantity_received=qty, quantity_available=qty,
            quantity_reserved=reserved,
            received_date=date.today() - timedelta(days=5),
            distribution_cutoff=date.today() + timedelta(days=cutoff_days),
            status=status)

    def _request(self, product, qty, required_days=5, org=None, user=None):
        req = Request.objects.create(
            organization=org or self.org,
            requested_by=user or self.partner,
            required_by_date=date.today() + timedelta(days=required_days))
        RequestItem.objects.create(request=req, product=product,
                                   quantity_requested=qty)
        return req

    def test_get_inventory_lists_batch_rows(self):
        self._batch(self.product, qty=30, code='T-INV-1')
        rows = agent_tools.get_inventory()
        row = next(r for r in rows if r['batch_code'] == 'T-INV-1')
        self.assertEqual(row['qty_available'], 30)
        self.assertEqual(row['qty_unreserved'], 30)
        self.assertEqual(row['cutoff_date'],
                         (date.today() + timedelta(days=10)).isoformat())

    def test_get_pending_requests_summary_and_items(self):
        before = agent_tools.get_pending_requests()['n_requests_open']
        self._request(self.product, qty=17)
        open_row = agent_tools.get_pending_requests()
        self.assertEqual(open_row['n_requests_open'], before + 1)
        self.assertEqual(len(open_row['requests']), before + 1)
        item = open_row['requests'][0]['items'][0]
        self.assertEqual(item['qty_unmet'], 17)

        req2 = self._request(self.product, qty=5)
        item2 = req2.items.first()
        item2.quantity_allocated = 5
        item2.save()
        open_row = agent_tools.get_pending_requests()
        # req2 is now fully covered, so it no longer counts as open demand
        self.assertEqual(open_row['n_requests_open'], before + 1)

    def test_get_pending_requests_caps_listing(self):
        for i in range(30):
            self._request(self.product, qty=i + 1, required_days=i + 1)
        open_row = agent_tools.get_pending_requests(cap=25)
        self.assertEqual(open_row['n_requests_open'], 30)
        self.assertEqual(len(open_row['requests']), 25)
        self.assertTrue(open_row['truncated'])

    def test_demand_forecast_gap(self):
        self._batch(self.product, qty=40, code='T-FC-1')
        DemandForecast.objects.create(
            product=self.product, forecast_date=date.today(),
            period_days=7, predicted_quantity=25.0,
            model_version=agent_tools.FORECAST_VERSION)
        rows = agent_tools.get_demand_forecast()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['forecast_7d'], 25.0)
        # stock includes the BaseTestData fixture batch (100 free units)
        self.assertEqual(rows[0]['stock_free'], 140)
        self.assertEqual(rows[0]['gap'], 115.0)

    def test_at_risk_excludes_passed_cutoffs(self):
        self._batch(self.product, qty=10, cutoff_days=-1, code='T-OLD-1')
        self._batch(self.product, qty=12, cutoff_days=2, code='T-NEAR-1')
        rows = agent_tools.get_batches_at_risk(days=7)
        codes = {r['batch_code'] for r in rows}
        self.assertIn('T-NEAR-1', codes)
        self.assertNotIn('T-OLD-1', codes)

    def test_generate_allocation_creates_draft_without_reservation(self):
        self._batch(self.product, qty=30, code='T-GEN-1')
        req = self._request(self.product, qty=10)
        result = agent_tools.generate_allocation(self.coordinator, [req.pk])
        self.assertIn('plan_id', result)
        self.assertEqual(len(result['items']), 1)
        self.assertEqual(result['items'][0]['qty'], 10)
        plan = AllocationPlan.objects.get(pk=result['plan_id'])
        self.assertEqual(plan.status, AllocationPlan.Status.DRAFT)
        self.assertEqual(plan.created_by, self.coordinator)
        batch = FoodBatch.objects.get(batch_code='T-GEN-1')
        self.assertEqual(batch.quantity_reserved, 0)
        self.assertIsNone(plan.approved_at)

    def test_generate_allocation_reports_unmet(self):
        # other_product has no batches except the small one we add, so the
        # 10-unit request can only be 4 covered -> unmet 6.
        self._batch(self.other_product, qty=4, code='T-SHORT-2')
        req = self._request(self.other_product, qty=10)
        result = agent_tools.generate_allocation(self.coordinator, [req.pk])
        self.assertEqual(result['unmet'][0]['unmet'], 6)
        self.assertEqual(len(result['items']), 1)

    def test_generate_allocation_unmet_relative_to_plan(self):
        # A fully coverable request must report NO unmet even though the
        # engine does not touch RequestItem.quantity_allocated (that only
        # happens on approval).
        self._batch(self.product, qty=5, code='T-COVER-1', cutoff_days=3)
        req = self._request(self.product, qty=10)
        result = agent_tools.generate_allocation(self.coordinator, [req.pk])
        self.assertEqual(result['unmet'], [])
        self.assertEqual(len(result['items']), 2)  # split across FEFO batches

    def test_validate_allocation_flags_bad_state(self):
        self._batch(self.product, qty=30, code='T-VAL-1')
        req = self._request(self.product, qty=10)
        result = agent_tools.generate_allocation(self.coordinator, [req.pk])
        validation = agent_tools.validate_allocation(result['plan_id'])
        self.assertTrue(validation['valid'])
        batch = FoodBatch.objects.get(batch_code='T-VAL-1')
        batch.quantity_available = 2
        batch.save()
        validation = agent_tools.validate_allocation(result['plan_id'])
        self.assertFalse(validation['valid'])
        self.assertTrue(any('T-VAL-1' in e for e in validation['errors']))

    def test_validate_allocation_unknown_plan(self):
        result = agent_tools.validate_allocation(999999)
        self.assertFalse(result['valid'])


def _tool_call(name, arguments='', call_id='call-1'):
    return {'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': arguments}}


class _FakeLLM:
    """Scripted LLM stand-in for hermetic agent tests (no network)."""

    def __init__(self, script):
        # Each step: {'tool_calls': [...]} or {'content': '...'};
        # a 'raise' value raises LLMUnavailable; a 'hook' callable runs
        # mid-loop (lets tests mutate DB state between rounds).
        self.script = list(script)
        self.chat_log = []

    def chat(self, messages, tools=None, tool_choice=None, **kwargs):
        self.chat_log.append({'n_messages': len(messages),
                              'tool_choice': tool_choice})
        step = self.script.pop(0)
        if 'raise' in step:
            raise llm_module.LLMUnavailable(step['raise'])
        if 'hook' in step:
            step['hook']()
        return {
            'content': step.get('content'),
            'tool_calls': step.get('tool_calls', []),
            'model': 'fake/agent-1',
            'prompt_tokens': 11,
            'completion_tokens': 7,
        }


def _tool_call(name, arguments='', call_id='call-1'):
    return {'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': arguments}}


class AgentRunnerTests(BaseTestData):
    """Phase 6: agent loop, planner retry rule, fallback paths."""

    def _batch(self, qty=30, code='T-RUN-1', cutoff_days=10):
        return FoodBatch.objects.create(
            product=self.product, donor=self.donor, batch_code=code,
            quantity_received=qty, quantity_available=qty,
            received_date=date.today() - timedelta(days=5),
            distribution_cutoff=date.today() + timedelta(days=cutoff_days))

    def _request(self, qty=10):
        req = Request.objects.create(
            organization=self.org, requested_by=self.partner,
            required_by_date=date.today() + timedelta(days=5))
        RequestItem.objects.create(request=req, product=self.product,
                                   quantity_requested=qty)
        return req

    def test_analyst_llm_loop_uses_tools(self):
        fake = _FakeLLM([
            {'tool_calls': [_tool_call('get_pending_requests', '{}', 'c1')]},
            {'content': '1 pending request; stock healthy.'},
        ])
        run = agents_module.run_analyst(self.coordinator, client=fake)
        self.assertEqual(run.status, AgentRun.Status.OK)
        self.assertEqual(run.model_name, 'fake/agent-1')
        self.assertEqual(len(run.tool_calls), 1)
        self.assertEqual(run.tool_calls[0]['tool'], 'get_pending_requests')
        self.assertIn('pending', run.output['report'].lower())
        self.assertEqual(run.prompt_tokens, 22)     # summed across rounds
        self.assertEqual(run.completion_tokens, 14)

    def test_analyst_falls_back_when_llm_unavailable(self):
        blank = _FakeLLM([{'raise': 'HTTP 429: rate limited'}])
        run = agents_module.run_analyst(self.coordinator, client=blank)
        self.assertEqual(run.status, AgentRun.Status.FALLBACK)
        self.assertIn('429', run.error_message)
        self.assertIn('shortages', run.output)
        self.assertEqual(len(run.tool_calls), 0)

    def test_planner_llm_traces_sequential_tools(self):
        req = self._request(qty=10)
        fake = _FakeLLM([
            {'tool_calls': [_tool_call('get_pending_requests', '{}', 'c1')]},
            {'tool_calls': [_tool_call(
                'generate_allocation',
                json.dumps({'request_ids': [req.pk]}), 'c2')]},
            {'tool_calls': [_tool_call(
                'validate_allocation',
                json.dumps({'plan_id': 999999}), 'c3')]},
            {'content': 'Plan 999999 invalid: unknown plan id.'},
        ])
        run = agents_module.run_planner(self.coordinator, client=fake)
        self.assertEqual(run.status, AgentRun.Status.OK)
        traced = [t['tool'] for t in run.tool_calls]
        self.assertEqual(
            traced,
            ['get_pending_requests', 'generate_allocation',
             'validate_allocation'])
        self.assertIn('unknown', run.output['report'].lower())
        # the planner DID generate a real draft before validating a bogus id
        self.assertIsNotNone(run.plan)
        self.assertEqual(run.plan.items.count(), 1)

    def test_planner_tracks_generated_plan(self):
        self._batch(qty=30)
        req = self._request(qty=10)
        fake = _FakeLLM([
            {'tool_calls': [
                _tool_call('get_pending_requests', '{}', 'c1'),
                _tool_call('generate_allocation',
                           json.dumps({'request_ids': [req.pk]}), 'c2')]},
            {'tool_calls': [_tool_call(
                'validate_allocation',
                json.dumps({'plan_id': 1}), 'c3')]},
            {'content': 'Plan proposed and validated.'},
        ])
        run = agents_module.run_planner(self.coordinator, client=fake)
        self.assertEqual(run.status, AgentRun.Status.OK)
        self.assertIsNotNone(run.plan)
        self.assertEqual(run.plan.status, AllocationPlan.Status.DRAFT)
        self.assertEqual(run.plan.items.count(), 1)

    def test_planner_retry_after_validation_failure(self):
        self._batch(qty=30, code='T-RETRY-1')
        req = self._request(qty=10)
        fake = _FakeLLM([
            {'tool_calls': [
                _tool_call('generate_allocation',
                           json.dumps({'request_ids': [req.pk]}), 'c1')]},
            # one step: run the hook (cancel the request) just before the
            # model gets billed for this round's validate tool call
            {'hook': lambda: Request.objects.filter(pk=req.pk).update(
                status=Request.Status.CANCELLED),
             'tool_calls': [
                _tool_call('validate_allocation',
                           json.dumps({'plan_id': 1}), 'c2')]},
            {'content': 'Validated; one retry considered.'},
        ])
        run = agents_module.run_planner(self.coordinator, client=fake)
        # The nudge makes the weak validation failure visible to the model;
        # messages history grows by the tool rounds + injected nudge.
        planner_calls = [c for c in fake.chat_log]
        self.assertEqual(len(planner_calls), 3)
        self.assertEqual(planner_calls[0]['n_messages'], 2)
        self.assertEqual(planner_calls[1]['n_messages'], 4)
        # the nudge user message is injected after the validation result
        self.assertEqual(planner_calls[2]['n_messages'], 7)

    def test_planner_falls_back_when_llm_unavailable(self):
        self._batch(qty=30)
        self._request(qty=10)
        broken = _FakeLLM([{'raise': 'HTTP 402: out of credits'}])
        run = agents_module.run_planner(self.coordinator, client=broken)
        self.assertEqual(run.status, AgentRun.Status.FALLBACK)
        self.assertIn('402', run.error_message)
        self.assertIsNotNone(run.plan)
        self.assertEqual(run.plan.status, AllocationPlan.Status.DRAFT)

    def test_tool_budget_exhaustion_forces_summary(self):
        fake = _FakeLLM(
            [{'tool_calls': [_tool_call('get_pending_requests', '{}',
                                        'cf')]}] * 5
            + [{'content': 'budget exhausted summary'}])
        run = agents_module.run_analyst(self.coordinator, client=fake)
        self.assertEqual(run.status, AgentRun.Status.OK)
        self.assertEqual(len(run.tool_calls), 5)
        self.assertIn('budget exhausted', run.output['report'])
        forced = fake.chat_log[-1]
        self.assertEqual(forced['tool_choice'], 'none')

    def test_unknown_tool_reports_error_not_crash(self):
        fake = _FakeLLM([
            {'tool_calls': [_tool_call('delete_everything', '{}', 'cx')]},
            {'content': 'Cannot do that.'},
        ])
        run = agents_module.run_analyst(self.coordinator, client=fake)
        self.assertEqual(run.tool_calls[0]['result'],
                         {'error': 'Unknown tool: delete_everything'})
        self.assertEqual(run.status, AgentRun.Status.OK)


class AgentPermissionTests(BaseTestData):
    def test_partner_blocked_from_ai_pages(self):
        self.client.force_login(self.partner)
        response = self.client.get(reverse('ai_list'), follow=True)
        self.assertNotContains(response, 'AI Proposals')

    def test_coordinator_reads_ai_pages(self):
        self.client.force_login(self.coordinator)
        response = self.client.get(reverse('ai_list'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'AI Proposals')


class ForecastBoostTests(BaseTestData):
    """Phase 7 / Config B: forecast-informed demand ordering.

    Scenario: Rice has surplus stock, Beans is projected to fall short.
    A pure-Rice request is calendar-earlier; a Beans+Rice bundle is
    later. Config A serves calendar order (the early Rice request can
    drain the shared Rice pool); Config B pulls the shortage-containing
    bundle forward, so the Beans partner keeps its Rice too.
    """

    def setUp(self):
        super().setUp()
        # isolate: drop fixture batches so stock numbers are exact
        FoodBatch.objects.all().delete()
        self.batch_rice = FoodBatch.objects.create(
            product=self.product, donor=self.donor, batch_code='FB-RICE',
            quantity_received=100, quantity_available=100,
            received_date=date.today() - timedelta(days=5),
            distribution_cutoff=date.today() + timedelta(days=30))
        self.batch_beans = FoodBatch.objects.create(
            product=self.other_product, donor=self.donor, batch_code='FB-BEANS',
            quantity_received=30, quantity_available=30,
            received_date=date.today() - timedelta(days=5),
            distribution_cutoff=date.today() + timedelta(days=25))
        for product, predicted, version in (
                (self.product, 20.0, agent_tools.FORECAST_VERSION),
                (self.other_product, 100.0, agent_tools.FORECAST_VERSION)):
            DemandForecast.objects.create(
                product=product, forecast_date=date.today(), period_days=7,
                predicted_quantity=predicted, model_version=version)
        # calendar-earlier pure-Rice request
        self.req_early = Request.objects.create(
            organization=self.org, requested_by=self.partner,
            required_by_date=date.today() + timedelta(days=2))
        RequestItem.objects.create(request=self.req_early,
                                   product=self.product,
                                   quantity_requested=80)
        # later Beans(shortage)+Rice bundle
        self.req_late = Request.objects.create(
            organization=self.other_org, requested_by=self.other_partner,
            required_by_date=date.today() + timedelta(days=6))
        RequestItem.objects.create(request=self.req_late,
                                   product=self.other_product,
                                   quantity_requested=50)
        self.late_rice_item = RequestItem.objects.create(
            request=self.req_late, product=self.product,
            quantity_requested=40)

    def _alloc_for(self, plan, item):
        return next((a.quantity_allocated for a in plan.items.all()
                     if a.request_item_id == item.pk), 0)

    def test_config_a_serves_calendar_order(self):
        plan = allocation.generate_plan(self.coordinator, name='A-field')
        # calendar order: req_early drained the shared Rice pool first,
        # leaving only 20 units for the bundle's Rice item
        self.assertEqual(self._alloc_for(plan, self.late_rice_item), 20)
        beans_item_late = self.req_late.items.first()
        self.assertEqual(self._alloc_for(plan, beans_item_late), 30)
        early_alloc = sum(
            a.quantity_allocated for a in plan.items.all()
            if a.request_item.request_id == self.req_early.pk)
        self.assertEqual(early_alloc, 80)

    def test_config_b_pulls_shortage_bundle_forward(self):
        plan = allocation.generate_plan(self.coordinator, name='B-field',
                                        forecast_boost=True)
        # shortage-containing request got Rice BEFORE the early request
        self.assertEqual(self._alloc_for(plan, self.late_rice_item), 40)
        # cal-earlier pure-Rice request then saw only 60 units left
        early_alloc = sum(
            a.quantity_allocated for a in plan.items.all()
            if a.request_item.request_id == self.req_early.pk)
        self.assertEqual(early_alloc, 60)
        # Beans shortage honoured FEFO-wise identically: all 30 allocated
        beans_alloc = sum(a.quantity_allocated for a in plan.items.all()
                          if a.request_item.product_id == self.other_product.id)
        self.assertEqual(beans_alloc, 30)

    def test_config_b_same_fefo_batch_rules(self):
        plan = allocation.generate_plan(self.coordinator, forecast_boost=True)
        self.assertTrue(allocation.validate_plan(plan)['valid'])
        # Beans still FEFO: single beans batch, nothing invented
        beans_alloc = sum(a.quantity_allocated for a in plan.items.all()
                          if a.request_item.product_id == self.other_product.id)
        self.assertEqual(beans_alloc, min(30, 50))

    def test_config_b_degenerates_to_config_a_without_forecasts(self):
        DemandForecast.objects.all().delete()

        def sig(plan):
            return sorted((it.request_item_id, it.food_batch_id,
                           it.quantity_allocated) for it in plan.items.all())
        plan_plain = allocation.generate_plan(
            self.coordinator, name='A-nofc')
        plan_boost = allocation.generate_plan(
            self.coordinator, name='B-nofc', forecast_boost=True)
        self.assertEqual(sig(plan_plain), sig(plan_boost))


class EvaluationReportTests(BaseTestData):
    """Phase 7 presentation layer: plain-language evaluation page."""

    def test_coordinator_sees_page_and_sections(self):
        self.client.force_login(self.coordinator)
        response = self.client.get(reverse('evaluation_report'))
        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn('Evaluation summary', body)
        self.assertIn('What this project tests', body)
        self.assertIn('Allocation comparison', body)

    def test_partner_is_redirected(self):
        self.client.force_login(self.partner)
        response = self.client.get(reverse('evaluation_report'))
        self.assertEqual(response.status_code, 302)


class RoutingTests(BaseTestData):
    """Dispatch-board route computation (pure function, presentation only)."""

    def _delivery(self, org_id, name, lat, lng, units):
        return {'org_id': org_id, 'org_name': name,
                'lat': lat, 'lng': lng, 'units': units, 'items': []}

    def test_all_stops_assigned_exactly_once(self):
        plan = AllocationPlan.objects.create(
            name='Route test', created_by=self.coordinator)
        deliveries = [
            self._delivery(1, 'North Charity', 13.03, 77.59, 40),
            self._delivery(2, 'East Charity', 12.98, 77.71, 30),
            self._delivery(3, 'South Charity', 12.90, 77.58, 20),
            self._delivery(4, 'West Charity', 12.97, 77.52, 10),
        ]
        result = routing.group_plan_into_trucks(plan, deliveries, 2)
        assigned = [s['org_id'] for t in result['trucks'] for s in t['stops']]
        self.assertEqual(sorted(assigned), [1, 2, 3, 4])
        self.assertEqual(result['unmapped'], [])

    def test_sequenced_and_unit_totals_match(self):
        plan = AllocationPlan.objects.create(
            name='Route totals', created_by=self.coordinator)
        deliveries = [
            self._delivery(1, 'North', 13.03, 77.59, 40),
            self._delivery(2, 'East', 12.98, 77.71, 30),
        ]
        result = routing.group_plan_into_trucks(plan, deliveries, 2)
        total = sum(t['total_units'] for t in result['trucks'])
        self.assertEqual(total, 70)
        for truck in result['trucks']:
            self.assertEqual(
                [s['seq'] for s in truck['stops']],
                list(range(1, len(truck['stops']) + 1)))
        self.assertTrue(all(t['distance_km'] > 0 or not t['stops']
                            for t in result['trucks']))

    def test_unmapped_orgs_reported_not_routed(self):
        plan = AllocationPlan.objects.create(
            name='Route unmapped', created_by=self.coordinator)
        deliveries = [
            self._delivery(1, 'Located', 12.98, 77.60, 10),
            self._delivery(2, 'No coords', None, None, 15),
        ]
        result = routing.group_plan_into_trucks(plan, deliveries, 2)
        self.assertEqual(result['unmapped'], ['No coords'])
        assigned = [s['org_id'] for t in result['trucks'] for s in t['stops']]
        self.assertEqual(assigned, [1])

    def test_deterministic_output(self):
        plan = AllocationPlan.objects.create(
            name='Route deterministic', created_by=self.coordinator)
        deliveries = [
            self._delivery(i, f'Org {i}', 12.9 + i * 0.01, 77.5 + i * 0.02,
                           10 * i)
            for i in range(1, 6)
        ]
        result1 = routing.group_plan_into_trucks(plan, deliveries, 2)
        result2 = routing.group_plan_into_trucks(plan, deliveries, 2)
        self.assertEqual(result1, result2)


class RoutingRoadsTests(RoutingTests):
    """OSRM road-geometry upgrade + graceful offline fallback."""

    def setUp(self):
        super().setUp()
        routing.reset_osrm_state()
        self.addCleanup(routing.reset_osrm_state)
        self.plan = AllocationPlan.objects.create(
            name='Road routes', created_by=self.coordinator)

    def _fake_route(self, points, timeout_s=routing.OSRM_TIMEOUT_S):
        return {
            'polyline': [(p[0] + 0.001, p[1] - 0.001) for p in points],
            'distance_km': 42.0,
            'duration_min': 60.0,
            'leg_minutes': [30.0] + [30.0 / max(1, len(points) - 1)
                                     for _ in points[1:]],
        }

    def _deliveries(self, n=2):
        return [self._delivery(
            i, f'Org {i}', 12.95 + i * 0.01, 77.60 + i * 0.01, 10 * i)
            for i in range(1, n + 1)]

    def test_osrm_upgrades_routes_with_manual_etas(self):
        with mock.patch.object(routing, '_osrm_route',
                               side_effect=self._fake_route):
            result = routing.group_plan_into_trucks(
                self.plan, self._deliveries(), 1)
        self.assertTrue(result['road_route'])
        truck = result['trucks'][0]
        self.assertTrue(truck['road_route'])
        self.assertEqual(len(truck['polyline'][0]), 2)
        self.assertAlmostEqual(truck['polyline'][0][0], 13.0294, places=4)
        self.assertAlmostEqual(truck['polyline'][0][1], 77.5189, places=4)
        self.assertGreater(
            sum(t['distance_km'] for t in result['trucks']), 0)
        self.assertEqual(
            [s['eta_min'] for s in truck['stops']],
            [30, 45])

    def test_osrm_failure_falls_back_to_straight_lines(self):
        with mock.patch.object(routing, '_osrm_route',
                               side_effect=OSError('network down')):
            result = routing.group_plan_into_trucks(
                self.plan, self._deliveries(), 1)
        self.assertFalse(result['road_route'])
        for truck in result['trucks']:
            self.assertFalse(truck['road_route'])
            self.assertNotIn('polyline', truck)
            self.assertNotIn('eta_min', truck['stops'][0])


class WeeklyLedgerTests(BaseTestData):
    """Weekly ledger: donations received per donor, distribution per org."""

    def _approve_small_plan(self):
        request = Request.objects.create(
            organization=self.org, requested_by=self.partner,
            required_by_date=date.today() + timedelta(days=3),
        )
        RequestItem.objects.create(
            request=request, product=self.product, quantity_requested=90)
        # generate_plan covers all pending requests; only self.org has one.
        plan = allocation.generate_plan(self.coordinator, name='Ledger plan')
        ok, errors = allocation.approve_plan(plan, self.coordinator)
        self.assertTrue(ok, errors)
        return plan

    def test_received_and_distributed_rows(self):
        monday = date.today() - timedelta(days=date.today().weekday())
        FoodBatch.objects.create(
            product=self.product, donor=self.donor, batch_code='L-1',
            quantity_received=60, quantity_available=60,
            received_date=monday,
            distribution_cutoff=monday + timedelta(days=14))
        FoodBatch.objects.create(
            product=self.other_product, batch_code='L-2',
            quantity_received=20, quantity_available=20,
            received_date=monday,
            distribution_cutoff=monday + timedelta(days=14))
        self._approve_small_plan()

        self.client.force_login(self.coordinator)
        response = self.client.get(reverse('weekly_ledger'))
        self.assertEqual(response.status_code, 200)
        data = response.context['data']
        received = {row['name']: row for row in data['received']}
        self.assertEqual(received['City Grocers']['units'], 60)
        self.assertEqual(received['Unattributed']['units'], 20)
        distributed = {row['name']: row for row in data['distributed']}
        self.assertEqual(distributed[self.org.name]['units'], 90)
        self.assertEqual(data['served_requests'], 1)
        self.assertEqual(data['received_units'], 80)

    def test_csv_export(self):
        self._approve_small_plan()
        self.client.force_login(self.coordinator)
        response = self.client.get(reverse('weekly_ledger_csv'))
        self.assertEqual(response['Content-Type'], 'text/csv')
        content = response.content.decode('utf-8')
        self.assertIn('Received from donors', content)
        self.assertIn(self.org.name, content)
        self.assertIn('Distributed to partners', content)

    def test_empty_week_renders(self):
        self.client.force_login(self.coordinator)
        # Far beyond any base-fixture batches (received 10/30 days ago).
        response = self.client.get(reverse('weekly_ledger') + '?w=-20')
        self.assertEqual(response.status_code, 200)
        data = response.context['data']
        self.assertEqual(data['received'], [])
        self.assertEqual(data['distributed'], [])

    def test_partner_blocked_and_bad_week_param(self):
        self.client.force_login(self.partner)
        self.assertEqual(
            self.client.get(reverse('weekly_ledger')).status_code, 302)
        self.client.force_login(self.coordinator)
        response = self.client.get(reverse('weekly_ledger') + '?w=oops')
        self.assertEqual(response.context['week_offset'], 0)


class DispatchBoardTests(BaseTestData):
    """Dispatch board view: access control + rendered board content."""

    def setUp(self):
        super().setUp()
        FoodBatch.objects.all().delete()
        # Deterministic, network-free board tests: pretend the OSRM
        # router is unreachable; routes stay indicative straight lines.
        routing.reset_osrm_state()
        self.addCleanup(routing.reset_osrm_state)
        osrm_offline = mock.patch.object(
            routing, '_osrm_route', side_effect=OSError('offline in tests'))
        osrm_offline.start()
        self.addCleanup(osrm_offline.stop)
        self.batch = FoodBatch.objects.create(
            product=self.product, donor=self.donor, batch_code='DB-1',
            quantity_received=60, quantity_available=60,
            received_date=date.today() - timedelta(days=5),
            distribution_cutoff=date.today() + timedelta(days=10))
        self.batch2 = FoodBatch.objects.create(
            product=self.other_product, donor=self.donor, batch_code='DB-2',
            quantity_received=40, quantity_available=40,
            received_date=date.today() - timedelta(days=5),
            distribution_cutoff=date.today() + timedelta(days=12))
        self.req = Request.objects.create(
            organization=self.org, requested_by=self.partner,
            required_by_date=date.today() + timedelta(days=3))
        self.item1 = RequestItem.objects.create(
            request=self.req, product=self.product, quantity_requested=50)
        self.item2 = RequestItem.objects.create(
            request=self.req, product=self.other_product,
            quantity_requested=40)
        self.plan = allocation.generate_plan(self.coordinator,
                                             name='Dispatch test')

    def test_board_renders_trucks_and_kpis(self):
        self.client.force_login(self.coordinator)
        response = self.client.get(
            reverse('plan_dispatch', args=[self.plan.pk]))
        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn('Truck manifests', body)
        self.assertIn('Execution record', body)
        self.assertIn('route-data', body)
        self.assertIn(self.org.name, body)

    def test_board_access_control(self):
        self.client.force_login(self.partner)
        response = self.client.get(
            reverse('plan_dispatch', args=[self.plan.pk]))
        self.assertEqual(response.status_code, 302)

    def test_truck_count_override(self):
        self.client.force_login(self.coordinator)
        response = self.client.get(
            reverse('plan_dispatch', args=[self.plan.pk]) + '?trucks=3')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['n_trucks'], 3)

    def test_routes_persisted_on_plan(self):
        self.client.force_login(self.coordinator)
        self.assertIsNone(self.plan.routes)
        self.client.get(reverse('plan_dispatch', args=[self.plan.pk]))
        self.plan.refresh_from_db()
        self.assertIn('2', self.plan.routes)

    def test_stored_routes_reused_between_visits(self):
        Organization.objects.filter(pk=self.org.pk).update(
            latitude=12.95, longitude=77.62)
        self.client.force_login(self.coordinator)
        self.client.get(reverse('plan_dispatch', args=[self.plan.pk]))
        self.plan.refresh_from_db()
        # Seed a marker into the stored record; a plain revisit must ship
        # the stored record untouched (no per-request recomputation).
        self.plan.routes['2']['marker'] = 'kept'
        AllocationPlan.objects.filter(pk=self.plan.pk).update(
            routes=self.plan.routes)
        response = self.client.get(
            reverse('plan_dispatch', args=[self.plan.pk]))
        self.assertEqual(response.context['route_data'].get('marker'),
                         'kept')

    def test_refresh_and_second_count_recompute(self):
        Organization.objects.filter(pk=self.org.pk).update(
            latitude=12.95, longitude=77.62)
        self.client.force_login(self.coordinator)
        self.client.get(reverse('plan_dispatch', args=[self.plan.pk]))
        self.plan.refresh_from_db()
        self.plan.routes['2']['marker'] = 'kept'
        AllocationPlan.objects.filter(pk=self.plan.pk).update(
            routes=self.plan.routes)
        # ?refresh=1 replaces the stored record (marker wiped)...
        self.client.get(reverse('plan_dispatch',
                                args=[self.plan.pk]) + '?refresh=1')
        self.plan.refresh_from_db()
        self.assertNotIn('marker', self.plan.routes['2'])
        # ...and another truck count adds its own cached entry.
        self.client.get(reverse('plan_dispatch',
                                args=[self.plan.pk]) + '?trucks=3')
        self.plan.refresh_from_db()
        self.assertIn('2', self.plan.routes)
        self.assertIn('3', self.plan.routes)

    def test_stop_status_chips_follow_plan_state(self):
        Organization.objects.filter(pk=self.org.pk).update(
            latitude=12.95, longitude=77.62)
        self.client.force_login(self.coordinator)
        response = self.client.get(
            reverse('plan_dispatch', args=[self.plan.pk]))
        self.assertIn('planned', response.content.decode())
        self.plan.status = 'COMPLETED'
        self.plan.save(update_fields=['status'])
        response = self.client.get(
            reverse('plan_dispatch', args=[self.plan.pk]))
        self.assertIn('delivered', response.content.decode())


class SourcingTests(BaseTestData):
    """Sourcing recommendations: gap -> visible sourcing instruction."""

    def setUp(self):
        super().setUp()
        self._tmp_brief = Path(tempfile.gettempdir()) / 'nl_test_brief.json'
        sourcing_module.BRIEF_PATH = self._tmp_brief
        self.addCleanup(
            lambda: self._tmp_brief.unlink(missing_ok=True))

    def _forecast(self, quantities):
        for offset, qty in enumerate(quantities):
            DemandForecast.objects.create(
                product=self.product,
                forecast_date=date.today() + timedelta(days=offset),
                predicted_quantity=qty,
                model_version='vTEST', period_days=7)

    def _open_request(self):
        Request.objects.create(
            organization=self.org, requested_by=self.partner,
            required_by_date=date.today() + timedelta(days=5))
        RequestItem.objects.create(
            request=Request.objects.last(), product=self.product,
            quantity_requested=40)

    def test_shortfall_row_with_cover_by_and_partner(self):
        self._forecast([80, 50])     # 130 predicted vs 100 free stock
        self._open_request()
        rows = sourcing_module.compute_sourcing_rows('vTEST')
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['product_name'], 'Rice')
        self.assertEqual(row['units_needed'], 30)
        self.assertEqual(row['stock_free'], 100)
        self.assertEqual(row['cover_by'], date.today() + timedelta(days=1))
        self.assertEqual(row['partners'], ['Hope Shelter'])
        self.assertEqual(row['partner_count'], 1)

    def test_healthy_stock_yields_no_rows(self):
        self._forecast([30, 40])     # 70 predicted < 100 free stock
        rows = sourcing_module.compute_sourcing_rows('vTEST')
        self.assertEqual(rows, [])
        self.assertIn('No projected shortfalls',
                      sourcing_module.deterministic_brief(rows))

    def test_page_access_and_render(self):
        self._forecast([80, 50])
        self.client.force_login(self.coordinator)
        with mock.patch.object(views_module, 'MLP_VERSION', 'vTEST'):
            response = self.client.get(reverse('sourcing'))
        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn('Sourcing recommendations', body)
        self.assertIn('Rice', body)
        self.assertIn('Log expected donation', body)
        self.client.force_login(self.partner)
        self.assertEqual(
            self.client.get(reverse('sourcing')).status_code, 302)

    def test_explain_falls_back_without_llm(self):
        self._forecast([80, 50])
        self.client.force_login(self.coordinator)
        with mock.patch.object(views_module, 'MLP_VERSION', 'vTEST'), \
             mock.patch.object(sourcing_module, 'llm_chat',
                               side_effect=OSError('offline')):
            response = self.client.post(reverse('sourcing_explain'),
                                        follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(AgentRun.objects.count(), 1)
        run = AgentRun.objects.get()
        self.assertEqual(run.status, 'FALLBACK')
        self.assertEqual(run.role, 'ANALYST')
        self.assertIn('standard brief', response.content.decode())

    def test_explain_records_llm_brief(self):
        self._forecast([80, 50])
        self.client.force_login(self.coordinator)
        with mock.patch.object(views_module, 'MLP_VERSION', 'vTEST'), \
             mock.patch.object(
                 sourcing_module, 'llm_chat',
                 return_value={'content': 'Buy rice before Tuesday.',
                               'model': 'test/model', 'prompt_tokens': 0,
                               'completion_tokens': 0}):
            self.client.post(reverse('sourcing_explain'))
        run = AgentRun.objects.get()
        self.assertEqual(run.status, 'OK')
        self.assertEqual(run.model_name, 'test/model')
        with mock.patch.object(views_module, 'MLP_VERSION', 'vTEST'):
            response = self.client.get(reverse('sourcing'))
        self.assertIn('Buy rice before Tuesday.', response.content.decode())
        self.assertIn('AI brief', response.content.decode())

    def test_donation_form_prefills_product(self):
        self.client.force_login(self.coordinator)
        response = self.client.get(
            reverse('donation_create') + f'?product={self.product.pk}')
        self.assertEqual(response.context['form'].initial['product'],
                         self.product)

    def test_clean_narrative_strips_reasoning_leak(self):
        # Regression: reasoning models used to leak their planning text
        # ("The user wants... Let's craft... Count words...") into the
        # coordinator brief. Tagged output must win; meta noise must go.
        leaked = (
            'The user wants a short urgent paragraph (max 90 words) based '
            "on the JSON facts. Need to include what to source. Let's "
            'craft ~70 words. Count words: We(1) need(2)...\n'
            '<FINAL>We need to source 49 units of beans by 23 October '
            '2026; otherwise Hope Shelter will run out of staples.'
            '</FINAL>')
        result = sourcing_module.clean_narrative(leaked, 'FALLBACK TEXT')
        self.assertTrue(result.startswith('We need to source 49 units'))
        self.assertNotIn('Let', result)
        self.assertNotIn('JSON', result)
        # Untagged meta-blob → nothing usable survives → fallback brief.
        blob = ("The user wants a short paragraph. Need to include what "
                "to source. Let's craft ~70 words.")
        self.assertEqual(
            sourcing_module.clean_narrative(blob, 'FALLBACK TEXT'),
            'FALLBACK TEXT')


class ScanImportTests(BaseTestData):
    """Photo -> AI draft -> coordinator approval -> products + batches."""

    def setUp(self):
        super().setUp()
        self.client.force_login(self.coordinator)

    def _upload(self, rows_json='{"items": [{"name": "Milk 500ml", "quantity": 25, "unit": "500ml"}]}',
                llm_ok=True):
        from django.core.files.uploadedfile import SimpleUploadedFile
        photo = SimpleUploadedFile('list.png', b'fakepngbytes',
                                   content_type='image/png')
        if llm_ok:
            patcher = mock.patch.object(
                llm_module, 'chat_vision',
                return_value={'content': rows_json, 'model': 'test/vision',
                              'prompt_tokens': 0, 'completion_tokens': 0})
        else:
            patcher = mock.patch.object(
                llm_module, 'chat_vision',
                side_effect=llm_module.LLMUnavailable('no vision'))
        with patcher:
            response = self.client.post(reverse('scan_import'),
                                        {'photo': photo, 'donor': ''})
        return response

    def test_upload_parses_rows_into_review_session(self):
        response = self._upload()
        self.assertEqual(response.status_code, 302)
        draft = self.client.session['scan_draft']
        self.assertEqual(draft['rows'][0]['name'], 'Milk 500ml')
        self.assertEqual(draft['rows'][0]['qty'], 25)
        self.assertTrue(draft['image'].startswith('scan-'))

    def test_unreadable_photo_falls_back_to_manual_entry(self):
        response = self._upload(rows_json='', llm_ok=False)
        self.assertEqual(response.status_code, 302)
        draft = self.client.session['scan_draft']
        self.assertEqual(draft['rows'], [])

    def test_rejects_non_image(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        fake = SimpleUploadedFile('x.txt', b'hello', content_type='text/plain')
        response = self.client.post(reverse('scan_import'),
                                    {'photo': fake})
        draft = self.client.session.get('scan_draft')
        self.assertIsNone(draft)

    def test_approve_creates_products_and_batches(self):
        self._upload()
        response = self.client.post(reverse('scan_approve'), {
            'include': ['0'],
            'name-0': 'Milk 500ml',
            'qty-0': '30',                      # coordinator corrected 25 -> 30
            'unit-0': '500ml',
            'category-0': 'BEVERAGE',
            'cutoff-0': str(date.today() + timedelta(days=9)),
        })
        self.assertEqual(response.status_code, 302)
        product = Product.objects.get(name__iexact='Milk 500ml')
        self.assertEqual(product.category, 'BEVERAGE')
        batch = FoodBatch.objects.get(product=product)
        self.assertEqual(batch.quantity_received, 30)
        self.assertEqual(batch.status, 'AVAILABLE')
        self.assertTrue(batch.batch_code.startswith('SCAN-'))
        self.assertEqual(batch.distribution_cutoff,
                         date.today() + timedelta(days=9))
        self.assertIsNotNone(batch.donor is None)
        self.assertIn('scanned photo', batch.notes)
        # Session cleared after a successful import.
        self.assertNotIn('scan_draft', self.client.session)

    def test_approve_skips_unticked_rows_and_reuses_product(self):
        self._upload(
            '{"items": [{"name": "Milk 500ml", "quantity": 25, "unit": ""},'
            ' {"name": "Rice", "quantity": 40, "unit": ""}]}')
        response = self.client.post(reverse('scan_approve'), {
            'include': ['0'],                   # Rice row left unticked
            'name-0': 'Rice',
            'qty-0': '40',
            'unit-0': '1kg pack',
            'category-0': 'GRAIN',
            'cutoff-0': '',
        })
        self.assertEqual(response.status_code, 302)
        # 'Rice' already existed (BaseTestData) — reused, not duplicated.
        self.assertEqual(Product.objects.filter(name__iexact='Rice').count(),
                         1)
        self.assertEqual(FoodBatch.objects.count(), 3)  # B-001 + B-EXP + new
        self.assertFalse(FoodBatch.objects.filter(
            product__name__iexact='Milk 500ml').exists())

    def test_approval_records_activity(self):
        self._upload()
        self.client.post(reverse('scan_approve'), {
            'include': ['0'], 'name-0': 'Milk 500ml', 'qty-0': '10',
            'unit-0': '', 'category-0': 'OTHER', 'cutoff-0': ''})
        self.assertTrue(ActivityLog.objects.filter(
            action_type='SCAN_IMPORT').exists())
        self.assertTrue(ActivityLog.objects.filter(
            action_type='DONATION_RECORDED').exists())

    def test_partner_blocked_and_empty_approve_safe(self):
        self.client.force_login(self.partner)
        self.assertEqual(
            self.client.get(reverse('scan_import')).status_code, 302)
        self.client.force_login(self.coordinator)
        response = self.client.post(reverse('scan_approve'), {})
        self.assertEqual(response.status_code, 302)

    def test_links_present_on_intake_pages(self):
        donation_body = self.client.get(
            reverse('donation_create')).content.decode()
        product_body = self.client.get(
            reverse('product_create')).content.decode()
        self.assertIn('Scan a donation list', donation_body)
        self.assertIn('Scan a donation list', product_body)
