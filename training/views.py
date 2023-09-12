from django.shortcuts import render
from django.forms import ModelForm
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect

import json, os, shutil
import socket

from video.models import Video
from project.models import Project
import cv2
import tempfile

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

# Create your views here.
def ModelTraining(request):
    return render(request, 'training/training.html')

class DataForm(ModelForm):
    class Meta:
        model = Video
        fields = ["filename"]
        #fields = '__all__'

class ProjectForm(ModelForm):
    class Meta:
        model = Project
        fields = '__all__'

def project_list_data(request,pk):
    project_object = Project.objects.all()
    form1 = ProjectForm()
    form = DataForm()
    data_objects = Video.objects.filter(project_id = pk)
    context = {
           "form1":form1,
           "form":form,
           "data_objects":data_objects,
           "project_object":project_object
        }
    if request.method == "GET":
        return render(request, 'training/training.html', context)

    form = DataForm(data = request.POST, files= request.FILES)

    # write the inmemorydata in to cv2
    with tempfile.NamedTemporaryFile() as temp:
        for chunk in request.FILES.get('filename').chunks():
            temp.write(chunk)
        cap = cv2.VideoCapture(temp.name)
    VideoInfo = {}
    VideoInfo["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    VideoInfo["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    VideoInfo["frame_num"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    form.instance.video_info = VideoInfo
    form.instance.project_id = pk
    if form.is_valid():
        form.save()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            form.instance.filename
            s.sendall((os.path.join(os.getcwd(),'media',str(form.instance.filename)) + "*" + "V" + str(form.instance.id) ).encode())

        return render(request, 'training/training.html',context)
    print(form.errors)
    
# Create your views here.
def index(request):
    project_object = Project.objects.all()
    form1 = ProjectForm()
    context = {
           "form1":form1,
            "project_object":project_object,
        }
    if request.method == "GET":
        #print(project_object[0].projecName)
        return render(request, 'training/training.html',context)

    # User Post method hand on data, and check the data format

    form = ProjectForm(data = request.POST)
    if form.is_valid():
        form.save()
        return render(request, 'training/training.html',context)

def ajax_getdata(request):
    data_objects = Video.objects.filter(id = request.GET.get('id'))
    data_dict = {"data":data_objects[0].data, "video_info":data_objects[0].video_info}
    return JsonResponse(data_dict)