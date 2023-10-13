from django.shortcuts import render
from django.forms import ModelForm
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect
from django.urls import reverse

import json, os, shutil
from lib.JsonCreator import JsonCreator
import socket

from .models import Video
from project.models import Project

import cv2
import tempfile

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

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
        return render(request, 'video/video_test.html', context)

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
        form.instance.filename

        return render(request, 'video/video_test.html',context)
    print(form.errors)

def ajax_getdata(request):
    data_objects = Video.objects.filter(id = request.GET.get('id'))
    data_dict = {"data":data_objects[0].data, "video_info":data_objects[0].video_info}
    return JsonResponse(data_dict)

def ajax_submitdata(request):
    data = json.loads(request.POST.get('data'))
    Video.objects.filter(id = request.POST.get('id')).update(data = data)
    #Video.objects.filter(id = request.POST.get('id')).update(data = {} )
    return HttpResponse("SUCCESS")

def ajax_delete_item(request,pk):
    object = Video.objects.filter(id = request.GET.get('data_id'))[0]
    os.remove(os.path.join(os.getcwd(),'media',str(object.filename)))
    object.delete()
    data_dict = {"status":True}
    return JsonResponse(data_dict)

def delete_project(request,pk):
    path = os.path.join(os.getcwd(),'media',"project_" + str(pk))
    if os.path.isdir(path):
        shutil.rmtree(path)
    Project.objects.filter(id = pk).delete()
    #return render(request, 'video/show_data_items.html', context)
    return HttpResponseRedirect('/Project/')

def export_project(request,pk):

    videoOjects = Video.objects.filter(project_id = pk)
    path = os.path.join(os.getcwd(),'export')

    for videoOject in videoOjects:
        videoName = str(videoOject.filename)
        cap = cv2.VideoCapture(os.path.join(os.getcwd(), 'media', videoName))
        os.makedirs(os.path.join(path, videoName), exist_ok=True)
        for frameName in videoOject.data:
            str_frameName = str(frameName)
            cap.set(1, int(str_frameName[6:]))
            ret, frame = cap.read()
            cv2.imwrite(os.path.join(path, videoName,str_frameName + ".png"), frame)
            f= open(os.path.join(path, videoName,str_frameName + ".txt"), "w+")
            for face_label in videoOject.data[frameName]:
                f.write(str(face_label)+"\n")
    return HttpResponseRedirect('/Project/Video/{}'.format(pk))

def data_export(request,pk):
    imageOjects = Video.objects.filter(project_id = pk)
    JsonObj = JsonCreator("coco", imageOjects)
    JsonData = JsonObj.OutputJson()
    JsonObject = json.dumps(JsonData, indent=4)
    path = os.path.join(os.getcwd(),'export')
    project_path = os.path.join(path, "project_{}".format(pk))

    os.makedirs(project_path, exist_ok=True)
    # Writing to sample.json
    with open(os.path.join(project_path,"result_project_{}.json".format(pk)), "w") as outfile:
        outfile.write(JsonObject)

    return HttpResponseRedirect('/Project/Video/{}'.format(pk))

def ModelTraining(request,pk):
    return render(request, 'training/training.html')
