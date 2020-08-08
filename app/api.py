import io
import os
import csv
import json

import requests 
from PIL import Image

# pytorch imports
import torchvision.transforms as transforms
from torchvision import models
from torch import optim
import torch.nn as nn
import torch

from flask import request, Flask, jsonify, render_template

model = models.resnet50(pretrained=False)

n_inputs = model.fc.in_features
n_classes = 101
img_size = 224

classifier = nn.Sequential(
    nn.Linear(n_inputs,img_size),
    nn.LeakyReLU(),
    nn.Linear(img_size,n_classes)
    )

model.fc = classifier
model

if torch.cuda.is_available():
    model.load_state_dict(torch.load(os.path.join('models','food-classifier.pt')))
else:
    model.load_state_dict(torch.load(os.path.join('models','food-classifier.pt'), map_location= torch.device('cpu')))

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

        # ensure file is an image
        img_extensions =['ras', 'xwd', 'bmp', 'jpe', 'jpg', 'jpeg', 'xpm', 'ief', 'pbm', 'tif', 'gif', 'ppm', 'xbm', 'tiff', 'rgb', 'pgm', 'png', 'pnm']
        if img_data.filename.split('.')[1] not in img_extensions:
            return jsonify(message='Please upload an appropriate image file', error=True)

        # convert to bytes
        img_bytes = img_data.read()
        # get prediction for image
        label, confidence = get_prediction(image_bytes=img_bytes)

        return jsonify(label=label, confidence=confidence)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':

        correct = bool(request.args.get('correct'))
        user_label = request.args.get('label')
        img_data = request.files['file']

        target = os.path.join(APP_ROOT, 'responses/images')
        
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

def get_classes():
    classes = []
    with open('static/extra/labels.txt','r') as f:
        for line in f:
            classes.append(line)
    return classes

def transform_image(image_bytes):
    image_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_bytes))
    return image_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    # get all possible classes in a list
    classes = get_classes()
    # transform image to given tensor size and normalize
    tensor = transform_image(image_bytes)
    # get class outputs
    outputs = model(tensor)
    # run through softmax layer to normalize results to add up to 1
    softmax = torch.nn.functional.softmax(outputs, dim=1)
    # get top results
    top_prob, top_label = torch.topk(softmax, 1)

    predicted_idx = top_label
    prediction = classes[predicted_idx]

    return prediction, top_prob.item()

# start flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
