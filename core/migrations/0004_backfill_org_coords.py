"""Backfill organization coordinates around the Bangalore depot (dispatch board).

Deterministic per-pk placement inside ~6 km of the warehouse so existing
databases get a usable map instantly; `seed_data` uses the same scheme for
fresh setups.
"""

import hashlib

from django.db import migrations

DEPOT = (12.9716, 77.5946)  # Bengaluru city centre


def _coords_for(pk: int):
    """Stable pseudo-random offset per organization pk (±~0.05°)."""
    digest = hashlib.sha256(f'org-coord-{pk}'.encode()).digest()
    lat_off = ((digest[0] << 8) + digest[1]) % 1000 / 1000 - 0.5
    lng_off = ((digest[2] << 8) + digest[3]) % 1000 / 1000 - 0.5
    return (round(DEPOT[0] + lat_off * 0.1, 6),
            round(DEPOT[1] + lng_off * 0.1, 6))


def backfill_coords(apps, schema_editor):
    Organization = apps.get_model('core', 'Organization')
    for org in Organization.objects.filter(
            latitude__isnull=True, longitude__isnull=True):
        lat, lng = _coords_for(org.pk)
        org.latitude = lat
        org.longitude = lng
        org.save(update_fields=['latitude', 'longitude'])


def unbackfill(apps, schema_editor):
    Organization = apps.get_model('core', 'Organization')
    Organization.objects.update(latitude=None, longitude=None)


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0003_organization_coords'),
    ]

    operations = [
        migrations.RunPython(backfill_coords, unbackfill),
    ]
