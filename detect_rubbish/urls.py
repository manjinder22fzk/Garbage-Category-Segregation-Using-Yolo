from django.urls import path
from django.conf.urls.static import static
from django.conf import settings


from . import views

urlpatterns = [
    path('', views.detect_objects, name='detect_objects'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)