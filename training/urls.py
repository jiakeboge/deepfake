from django.urls import path,include

from .import views


app_name = 'training'
urlpatterns = [
    path('', views.index, name='index'),
    path('', views.ModelTraining, name='ModelTraining'),
]
