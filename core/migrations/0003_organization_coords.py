"""Geographic coordinates for partner organizations (dispatch board)."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_agentrun'),
    ]

    operations = [
        migrations.AddField(
            model_name='organization',
            name='latitude',
            field=models.FloatField(blank=True, help_text='Map latitude (dispatch board)', null=True),
        ),
        migrations.AddField(
            model_name='organization',
            name='longitude',
            field=models.FloatField(blank=True, help_text='Map longitude (dispatch board)', null=True),
        ),
    ]
