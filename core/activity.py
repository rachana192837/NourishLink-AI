from .models import ActivityLog


def log_activity(user, action_type, description, obj=None):
    from django.contrib.contenttypes.models import ContentType
    ct = ContentType.objects.get_for_model(obj) if obj is not None else None
    ActivityLog.objects.create(
        user=user,
        action_type=action_type,
        description=description,
        content_type=ct,
        object_id=getattr(obj, 'pk', None),
    )
