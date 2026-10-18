from django.urls import path

from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),

    # Inventory (coordinator)
    path('inventory/', views.inventory_list, name='inventory'),
    path('inventory/<int:pk>/', views.batch_detail, name='batch_detail'),
    path('inventory/<int:pk>/toggle-block/', views.batch_toggle_block,
         name='batch_toggle_block'),
    path('inventory/donate/', views.donation_create, name='donation_create'),
    path('products/new/', views.product_create, name='product_create'),

    # Requests
    path('requests/', views.request_list, name='request_list'),
    path('requests/mine/', views.my_requests, name='my_requests'),
    path('requests/new/', views.request_create, name='request_create'),
    path('requests/<int:pk>/', views.request_detail, name='request_detail'),
    path('requests/<int:pk>/status/', views.request_status_update,
         name='request_status_update'),
    path('requests/<int:pk>/cancel/', views.request_cancel,
         name='request_cancel'),
]
