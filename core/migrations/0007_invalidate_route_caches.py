"""Invalidate cached truck routes computed before the geography fix.

Stored dispatch-board routes still carry the old hash-scattered pins and
city-centre depot; they are recomputed lazily on the next board visit.
"""

from django.db import migrations


def clear_caches(apps, schema_editor):
    AllocationPlan = apps.get_model('core', 'AllocationPlan')
    AllocationPlan.objects.exclude(routes=None).update(routes=None)


def clear_caches_reverse(apps, schema_editor):
    clear_caches(apps, schema_editor)


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0006_add_plan_routes'),
    ]

    operations = [
        migrations.RunPython(clear_caches, clear_caches_reverse),
    ]
