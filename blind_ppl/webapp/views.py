from urllib import request
from django.shortcuts import render, redirect
from django.http import HttpResponse
import shutil


from django.views.decorators.csrf import csrf_exempt
# import source
import numpy as np
from PIL import Image
import base64
import re
from binascii import a2b_base64
import pickle
import os
# from model_predict import prediction
from eval import prediction
def index(request):
    return render(request, 'index.html')



@csrf_exempt
def get_image(request):
    if request.method == 'POST':

        image_b64 = request.POST.get('imgBase64', None)
    
        binary_data = a2b_base64(image_b64)

        fd = open('webapp/static/images/image.jpg', 'wb')
        fd.write(binary_data)
        fd.close()

        print('Genrating Caption......')
        
        caption = prediction()
        print('#'*10,end='\n')
        print(caption)
        print('#'*10,end='\n')
    return ''


@csrf_exempt
def get_regional_lang(request):
    if request.method == 'POST':
        value = request.POST.get('dropdown')
        print(value)
        
        return redirect('/webapp/')
    return render(request, 'lang.html')



