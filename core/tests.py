from datetime import date, timedelta

from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from .models import (
    ActivityLog,
    Donor,
    FoodBatch,
    Organization,
    Product,
    Request,
    RequestItem,
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
