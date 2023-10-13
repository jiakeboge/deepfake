# training/consumers.py
import json
import os
from channels.db import database_sync_to_async
from asgiref.sync import async_to_sync
from channels.generic.websocket import AsyncWebsocketConsumer

from AiServer.AiServerHandler import train_model
from AiServer.AiServerHandler import inference_model
from AiServer.AiServerHandler import stop_training


train_json = {
        "batch_size": 1,
        "epochs": 10,
        "lr": 0.001,
        "momentum": 0.9,
        "optim": "rmsprop",
        "data_path": "/media/hkuit164/638FF62A1FE9E82D/wider_face/",
        "save_folder": 'checkpoint/test'
    }

inferenceJson = {
    "model_path": "checkpoint/pyramidbox_-1.pth",
    # "input_path": "./Pyramidbox/test_custom_videos/",
    "interval": 10,
    # "output_path": "tmp",
}


class TrainingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        print("dasjkhdashduasud")
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        self.trainingInfor = text_data_json['trainingInfor']

        # Start the training task
        if message == 'start_training':
            train_json["lr"] = text_data_json['trainingInfor']["LR"]
            train_json["optim"] = text_data_json['trainingInfor']["optim"]
            train_json["save_folder"] = os.path.join("checkpoint", text_data_json['trainingInfor']["save_folder"])
            await self.start_training()
        elif message == 'start_inference':
            path = text_data_json['path']
            path = path.replace('http://127.0.0.1:8000/', "")
            inferenceJson["input_path"] = path
            await self.startInference()
        elif message == "stop_training":
            await self.stopTraining()


    @database_sync_to_async
    def _train_model_async(self):
        #return train_model(self.send_training_update, self.trainingInfor)
        return train_model(self.send_training_update, train_json)

    @database_sync_to_async
    def _inference_model_async(self):
        return inference_model(self.send_training_update, inferenceJson)

    @database_sync_to_async
    def _stop_training_async(self):
        stop_training()
    
    def send_training_update(self, message):
        # This method will be called by the train_model function to send messages to the frontend
        # Since it's being called from synchronous code, we need to use the "async_to_sync" helper
        async_to_sync(self.send)(json.dumps({'message': message}))

    async def start_training(self):
        # Add your training code here
        # Make sure to use asynchronous libraries or run synchronous code using "database_sync_to_async"
        await self.send(json.dumps({'message': 'Start Training'}))

        await self._train_model_async()

        await self.send(json.dumps({'message': 'Training completed'}))

    async def startInference(self):
        await self.send(json.dumps({'message': 'StartInference'}))

        jsonResult = await self._inference_model_async()

        await self.send(json.dumps({'message': 'Inference completed',
                                    'result': jsonResult
                                    }))

    async def stopTraining(self):
        await self.send(json.dumps({'message': 'StopTraining'}))

        await self._stop_training_async()

        # await self.send(json.dumps({'message': 'Inference completed',
        #                             'result': jsonResult
        #                             }))
