from django.contrib import admin
from django.urls import path
from .views import generate_image  # Import from the same directory

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/generate-image/', generate_image, name='generate-image'),
]
