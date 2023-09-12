from django.urls import path, include

from .import views

app_name = 'menu'
urlpatterns = [
    path('', views.home, name='menu_home'),
    path('Project/', include('project.urls')),
    #path('Segmentation/', include('segmentation.urls')),
    #path('Polygon/', include('polygon.urls')),

    # No longer needed.
    #path('Image/', include('image.urls')),
    #path('Video/', include('video.urls')),
]
