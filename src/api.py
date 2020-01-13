from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import requests
from utils.dataset import get_labels
from PIL import Image

from utils.dataset import resize_image

# Initialize the Flask application
app = Flask(__name__)

SERVER_URL = 'http://localhost:8501/v1/models/'

# route http posts to this method
@app.route('/api/predict_food', methods=['POST'])
def predict_food():
    app.logger.debug(request)
    r = request

    # decode image
    img = r.files[0]

    # do some fancy processing here....
    resized_image = resize_image(img)
    _, resized_img_encode = cv2.imencode('.jpg', resized_image)

    predict_response = requests.post(SERVER_URL, data=resize_image_encode.tostring())
    predict_response.raise_for_status()
    predict_response = predict_response.json()

    y_prob = np.array(response['predictions'])
    label_index = np.argmax(y_prob)

    # get food labels
    labels = get_labels()
    label = labels[label_index]

    # build a response dict to send back to client
    response = {'message': 'Predicted guess for food is {}'.format(label)
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)