from django.urls import path,include

from .import views


app_name = 'classification'
urlpatterns = [
    path('', views.index, name='index'),
    path('Video/', include('video.urls')),
    path('Image/', include('image.urls')),
]
