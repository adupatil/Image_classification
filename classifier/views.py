from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
# Create your views here.
from keras.models import load_model
from keras.preprocessing import image
import json
from tensorflow.compat.v1 import Graph, Session
import numpy as np

img_height, img_width = 224, 224
with open('./models/imagenet_classes.json', 'r') as f:
    label_info = f.read()

labelInfo = json.loads(label_info)

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model = load_model('./models/MobileNetModelImagenet.h5')


def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)


def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = x/255
    x = x.reshape(1, img_height, img_width, 3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi = model.predict(x)

    predicted_label = labelInfo[str(np.argmax(predi[0]))]
    context = {'filePathName': filePathName,
               'predictedLabel': predicted_label[1]}
    return render(request, 'index.html', context)
