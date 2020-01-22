from __future__ import print_function
import requests
import json
from PIL import Image
import io

import pdb

addr = 'http://localhost:5000'
test_url = addr + '/api/predict_food'

# prepare headers for http request
content_type = 'image/jpeg'

img = Image.open('../data/images/apple_pie/134.jpg')
imgByteArr = io.BytesIO()
img.save(imgByteArr,format='jpeg')

files = {'file': ('134.jpg', imgByteArr, content_type)}

# send http request with image and receive response
# response = requests.post(test_url, 
# files=files
# )
# print(response)
# # decode response
# print(json.loads(response.text))

with requests.Session() as s:
    response = s.post(test_url,files=files)
    response = json.loads(response.content)
    pdb.set_trace()
    print(response)