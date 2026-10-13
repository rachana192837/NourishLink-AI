from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages


def role_required(role):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('login')
            if not hasattr(request.user, 'profile'):
                messages.error(request, 'No profile found. Contact administrator.')
                return redirect('login')
            if request.user.profile.role != role:
                messages.error(request, 'You do not have permission to access this page.')
                return redirect('dashboard')
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def coordinator_required(view_func):
    return role_required('COORDINATOR')(view_func)


def partner_required(view_func):
    return role_required('PARTNER')(view_func)
