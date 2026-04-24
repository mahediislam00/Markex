from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('tools/', views.tools, name='tools'),
    path('markets/', views.markets, name='markets'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('portfolio/', views.portfolio, name='portfolio'),
    path('api/forecast/<str:symbol>/', views.api_forecast, name='api_forecast'),
    path('api/instruments/', views.api_instruments, name='api_instruments'),
    path('api/price/<str:symbol>/', views.api_price, name='api_price'),
]
