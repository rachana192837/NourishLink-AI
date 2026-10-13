import random
from datetime import timedelta

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import (
    AllocationItem,
    AllocationPlan,
    DemandForecast,
    Donor,
    FoodBatch,
    Organization,
    Product,
    Profile,
    Request,
    RequestItem,
)


class Command(BaseCommand):
    help = "Seed the database with sample development data."

    def handle(self, *args, **options):
        self.stdout.write("Clearing existing data (except superuser)...")
        self._clear()
        self.stdout.write("Seeding data...")

        superuser = self._create_superuser()
        coordinators = self._create_coordinators()
        organizations = self._create_organizations()
        partners = self._create_partners(organizations)
        donors = self._create_donors()
        products = self._create_products()
        batches = self._create_food_batches(products, donors)
        requests = self._create_requests(organizations, partners, products)

        self.stdout.write(self.style.SUCCESS("\n--- Seed Summary ---"))
        self.stdout.write(f"  Superuser:      1 ({'created' if superuser else 'already existed'})")
        self.stdout.write(f"  Coordinators:   {len(coordinators)}")
        self.stdout.write(f"  Organizations:  {len(organizations)}")
        self.stdout.write(f"  Partners:       {len(partners)}")
        self.stdout.write(f"  Donors:         {len(donors)}")
        self.stdout.write(f"  Products:       {len(products)}")
        self.stdout.write(f"  Food Batches:   {len(batches)}")
        self.stdout.write(f"  Requests:       {len(requests)}")
        self.stdout.write(f"  Request Items:  {RequestItem.objects.count()}")
        self.stdout.write(self.style.SUCCESS("Done!\n"))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clear(self):
        """Delete all seeded data. Superuser is preserved."""
        RequestItem.objects.all().delete()
        Request.objects.all().delete()
        AllocationItem.objects.all().delete()
        AllocationPlan.objects.all().delete()
        DemandForecast.objects.all().delete()
        FoodBatch.objects.all().delete()
        Product.objects.all().delete()
        Donor.objects.all().delete()
        Profile.objects.all().delete()
        Organization.objects.all().delete()
        User.objects.filter(is_superuser=False).delete()

    # -- Users -----------------------------------------------------------

    def _create_superuser(self):
        if User.objects.filter(username="admin").exists():
            self.stdout.write("  Superuser 'admin' already exists – skipping.")
            return None
        User.objects.create_superuser(
            username="admin",
            email="admin@nourishlink.local",
            password="admin123",
        )
        self.stdout.write("  Created superuser 'admin'.")
        return True

    def _create_coordinators(self):
        coordinators = []
        for i in range(1, 3):
            user = User.objects.create_user(
                username=f"coord{i}",
                email=f"coord{i}@nourishlink.local",
                password="coord123",
                first_name=f"Coordinator",
                last_name=f"{i}",
            )
            Profile.objects.create(user=user, role="COORDINATOR")
            coordinators.append(user)
            self.stdout.write(f"  Created coordinator 'coord{i}'.")
        return coordinators

    # -- Organizations & Partners ----------------------------------------

    def _create_organizations(self):
        org_data = [
            ("Hope Shelter", "SHELTER", "hope@example.com", "+1-555-0101", "123 Main St"),
            ("City Kitchen", "COMMUNITY_KITCHEN", "kitchen@example.com", "+1-555-0102", "456 Oak Ave"),
            ("Care NGO", "NGO", "care@example.com", "+1-555-0103", "789 Pine Rd"),
            ("Sunrise Shelter", "SHELTER", "sunrise@example.com", "+1-555-0104", "321 Elm St"),
            ("Community Kitchen", "COMMUNITY_KITCHEN", "community@example.com", "+1-555-0105", "654 Maple Dr"),
            ("Relief Foundation", "NGO", "relief@example.com", "+1-555-0106", "987 Cedar Ln"),
        ]
        orgs = []
        for name, org_type, email, phone, address in org_data:
            org = Organization.objects.create(
                name=name,
                org_type=org_type,
                contact_email=email,
                phone=phone,
                address=address,
                is_active=True,
            )
            orgs.append(org)
        self.stdout.write(f"  Created {len(orgs)} organizations.")
        return orgs

    def _create_partners(self, organizations):
        partners = []
        for idx, org in enumerate(organizations, start=1):
            user = User.objects.create_user(
                username=f"partner{idx}",
                email=f"partner{idx}@nourishlink.local",
                password="partner123",
                first_name=org.name.split()[0],
                last_name="Staff",
            )
            Profile.objects.create(user=user, role="PARTNER", organization=org)
            partners.append(user)
        self.stdout.write(f"  Created {len(partners)} partner users.")
        return partners

    # -- Donors ----------------------------------------------------------

    def _create_donors(self):
        donor_data = [
            ("FreshFarms Co.", "farms@example.com", "+1-555-0201", "100 Farm Rd"),
            ("GoodHarvest Ltd.", "harvest@example.com", "+1-555-0202", "200 Harvest Way"),
            ("Metro Grocers", "metro@example.com", "+1-555-0203", "300 Market St"),
            ("Sunshine Foods", "sunshine@example.com", "+1-555-0204", "400 Sunny Blvd"),
        ]
        donors = []
        for name, email, phone, address in donor_data:
            donor = Donor.objects.create(
                name=name,
                contact_email=email,
                phone=phone,
                address=address,
                notes=f"Regular donor – {name}",
            )
            donors.append(donor)
        self.stdout.write(f"  Created {len(donors)} donors.")
        return donors

    # -- Products --------------------------------------------------------

    def _create_products(self):
        product_data = [
            ("Rice", "1kg pack", "GRAIN", "Long grain white rice"),
            ("Beans", "400g pack", "LEGUME", "Dried kidney beans"),
            ("Pasta", "500g pack", "PASTA", "Penne pasta"),
            ("Canned Tomatoes", "400g can", "CANNED", "Whole peeled tomatoes"),
            ("Cooking Oil", "1L bottle", "OIL", "Vegetable cooking oil"),
            ("Sugar", "1kg pack", "OTHER", "White granulated sugar"),
            ("Tea", "250g box", "BEVERAGE", "Black tea bags"),
            ("Biscuits", "200g pack", "SNACK", "Digestive biscuits"),
            ("Lentils", "500g pack", "LEGUME", "Red lentils"),
            ("Powdered Milk", "400g tin", "OTHER", "Full cream milk powder"),
        ]
        products = []
        for name, unit, category, description in product_data:
            product = Product.objects.create(
                name=name,
                unit=unit,
                category=category,
                description=description,
            )
            products.append(product)
        self.stdout.write(f"  Created {len(products)} products.")
        return products

    # -- Food Batches ----------------------------------------------------

    def _create_food_batches(self, products, donors):
        now = timezone.now()
        batches = []
        counter = {}

        def next_code(product):
            tag = product.name.upper().replace(" ", "-")
            counter[tag] = counter.get(tag, 0) + 1
            return f"BATCH-{tag}-{counter[tag]:03d}"

        # Generate 25 batches with varying dates and statuses
        configs = [
            # (product_index, qty_received, days_until_cutoff, status)
            (0, 200, 3, "AVAILABLE"),
            (0, 150, 30, "AVAILABLE"),
            (0, 100, -2, "EXPIRED"),
            (1, 120, 5, "AVAILABLE"),
            (1, 80, 45, "AVAILABLE"),
            (2, 90, 7, "AVAILABLE"),
            (2, 60, 2, "AVAILABLE"),
            (3, 200, 60, "AVAILABLE"),
            (3, 100, 10, "BLOCKED"),
            (4, 50, 90, "AVAILABLE"),
            (4, 75, 4, "AVAILABLE"),
            (5, 300, 180, "AVAILABLE"),
            (5, 100, 1, "AVAILABLE"),
            (6, 80, 120, "AVAILABLE"),
            (6, 40, -5, "EXPIRED"),
            (7, 150, 15, "AVAILABLE"),
            (7, 60, 3, "BLOCKED"),
            (8, 100, 30, "AVAILABLE"),
            (8, 200, 60, "AVAILABLE"),
            (9, 120, 20, "AVAILABLE"),
            (9, 80, 90, "AVAILABLE"),
            (0, 50, 14, "BLOCKED"),
            (1, 60, 25, "AVAILABLE"),
            (3, 150, 45, "AVAILABLE"),
            (5, 80, 10, "AVAILABLE"),
        ]

        for prod_idx, qty, cutoff_days, status in configs:
            product = products[prod_idx]
            donor = random.choice(donors)
            qty_available = 0 if status in ("EXPIRED", "DEPLETED") else int(qty * 0.8)
            qty_reserved = int(qty * 0.1) if status == "AVAILABLE" else 0

            batch = FoodBatch.objects.create(
                product=product,
                donor=donor,
                batch_code=next_code(product),
                quantity_received=qty,
                quantity_available=qty_available,
                quantity_reserved=qty_reserved,
                received_date=(now - timedelta(days=max(30, 60 - cutoff_days))).date(),
                distribution_cutoff=(now + timedelta(days=cutoff_days)).date(),
                status=status,
            )
            batches.append(batch)

        self.stdout.write(f"  Created {len(batches)} food batches.")
        return batches

    # -- Requests & RequestItems -----------------------------------------

    def _create_requests(self, organizations, partners, products):
        now = timezone.now()
        statuses = ["PENDING", "APPROVED", "PARTIALLY_FULFILLED", "FULFILLED", "CANCELLED"]

        request_configs = [
            # (org_index, status, days_ahead, product_indices_and_quantities)
            (0, "PENDING", 5, [(0, 50), (1, 30), (4, 10)]),
            (1, "APPROVED", 7, [(2, 40), (3, 60)]),
            (2, "PENDING", 3, [(5, 25), (6, 15), (8, 20)]),
            (3, "FULFILLED", 10, [(0, 100), (9, 40)]),
            (4, "PARTIALLY_FULFILLED", 6, [(1, 50), (7, 30), (4, 20)]),
            (0, "CANCELLED", 4, [(3, 80)]),
            (5, "PENDING", 8, [(0, 60), (2, 35), (5, 40)]),
            (2, "APPROVED", 12, [(8, 45), (9, 25)]),
            (1, "PENDING", 2, [(6, 20), (7, 50), (0, 30)]),
            (3, "APPROVED", 9, [(1, 40), (4, 15)]),
        ]

        requests = []
        for org_idx, status, days, items_spec in request_configs:
            org = organizations[org_idx]
            partner = partners[org_idx]
            req = Request.objects.create(
                organization=org,
                requested_by=partner,
                required_by_date=(now + timedelta(days=days)).date(),
                status=status,
                notes=f"Sample request from {org.name}",
            )
            for prod_idx, qty in items_spec:
                allocated = qty if status == "FULFILLED" else (int(qty * 0.5) if status == "PARTIALLY_FULFILLED" else 0)
                RequestItem.objects.create(
                    request=req,
                    product=products[prod_idx],
                    quantity_requested=qty,
                    quantity_allocated=allocated,
                )
            requests.append(req)

        self.stdout.write(f"  Created {len(requests)} requests with items.")
        return requests
