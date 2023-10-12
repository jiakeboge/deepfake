from django.urls import path

from websocket_app import consumers

websocket_urlpatterns = [
    path('ws',consumers.TrainingConsumer.as_asgi()),
]
