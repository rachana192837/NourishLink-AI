"""Set geographically truthful coordinates for the seeded Bengaluru orgs.

Replaces the hash-scattered pins with named real localities (pins match
the printed address) and moves the depot concept to the Peenya warehouse
belt; `seed_data` seeds the same values for fresh setups.
"""

from django.db import migrations

DEPOT = (13.0284, 77.5199)  # Peenya Industrial Area (see core/routing.py)

ORG_LOCALITIES = {
    'Hope Shelter': ('12, 3rd Cross, Yelahanka New Town, Bengaluru 560106',
                     13.1007, 77.5963),
    'City Kitchen': ('44, 4th Block, Jayanagar, Bengaluru 560011',
                     12.9250, 77.5938),
    'Care NGO': ('7, Industrial Layout, Whitefield, Bengaluru 560066',
                 12.9698, 77.7500),
    'Sunrise Shelter': ('21, Balina Agrahara Rd, Hebbal, Bengaluru 560024',
                        13.0358, 77.5970),
    'Community Kitchen': ('312, 100 Feet Rd, Indiranagar, Bengaluru 560038',
                          12.9719, 77.6412),
    'Relief Foundation': ('18, 7th Main, BTM 2nd Stage, Bengaluru 560076',
                          12.9166, 77.6101),
}


def set_localities(apps, schema_editor):
    Organization = apps.get_model('core', 'Organization')
    for name, (address, lat, lng) in ORG_LOCALITIES.items():
        Organization.objects.filter(name=name).update(
            address=address, latitude=lat, longitude=lng)


def unset_localities(apps, schema_editor):
    Organization = apps.get_model('core', 'Organization')
    for name in ORG_LOCALITIES:
        Organization.objects.filter(name=name).update(
            latitude=None, longitude=None)


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0004_backfill_org_coords'),
    ]

    operations = [
        migrations.RunPython(set_localities, unset_localities),
    ]
