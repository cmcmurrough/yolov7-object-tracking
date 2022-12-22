import requests
import json
import cv2

address = 'http://10.0.0.205:5000'
test_url = address + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# open test image and encode to jpeg
img = cv2.imread('test.jpg')
_, img_encoded = cv2.imencode('.jpg', img)

# send the HTTP post request with encoded image attached
response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers)

# decode response
print(json.loads(response.text))

# expected output: {u'message': u'image received. size=124x124'}