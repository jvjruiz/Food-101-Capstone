import io
import os
import csv

import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torch import optim
import torch.nn as nn
import torch

from flask import request, Flask, jsonify, render_template

import config as cfg

model = models.resnet50(pretrained=False)

n_inputs = model.fc.in_features
n_classes = 101

classifier = nn.Sequential(
    nn.Linear(n_inputs,cfg.FINAL_SIZE),
    nn.Linear(cfg.FINAL_SIZE,cfg.FINAL_SIZE),
    nn.Linear(cfg.FINAL_SIZE,n_classes)
)

model.fc = classifier
model.load_state_dict(torch.load('../models/vgg16-transfer5.pt'))
model.eval()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # get file from request
        img_data = request.files['file']
        #convert to bytes
        img_bytes = img_data.read()
        label = get_prediction(image_bytes=img_bytes)
        return jsonify(label=label)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        correct = bool(request.form['correct'])
        user_label = request.form['user_label']
        img_data = request.files['file']
        target = os.path.join(APP_ROOT, 'responses/images')
        print(target)
        if not os.path.isdir(target):
            os.mkdir(target)

        filename = img_data.filename
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print(destination)
        img_data.save(destination)  

        csv_path = os.path.join(APP_ROOT, 'responses/response.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            line_writer = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            line_writer.writerow([img_data.filename, correct, user_label])

        return jsonify(message='Info successfully uploaded')



def transform_image(image_bytes):
    image_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_bytes))

    return image_transforms(image).unsqueeze(0)

def get_classes():
    classes = []
    with open('../data/meta/labels.txt','r') as f:
        for line in f:
            classes.append(line)
    return classes

def get_prediction(image_bytes):
    classes = get_classes()
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    print(y_hat)
    predicted_idx = y_hat.item()
    prediction = classes[predicted_idx]
    return prediction

# start flask app
app.run(host="0.0.0.0", port=5000,debug=True)
