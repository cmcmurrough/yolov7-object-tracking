import requests
import json
import cv2
import numpy as np
import jsonpickle


# function for sending an HTTP POST request containing a single encoded image
# response will contain detection results in JSON
def request_detect_and_track_single(image, address, path="/api/detect_and_track_single"):
    # prepare URL and headers for http request
    url = address + path
    content_type = 'image/png'
    headers = {'content-type': content_type}

    # send the HTTP post request with encoded image attached
    image_encoded = cv2.imencode('.png', image)[1]
    try:
        response = requests.post(url, data=np.array(image_encoded).tobytes(), headers=headers)
    except Exception as e:
        print("WARNING: exception occured while sending request: " + str(e))
        return None

    # return the response JSON    
    try:
        status = response.status_code
        if status != 200:
            raise Exception("Received unexpected HTTP status code " + str(status))
        return json.loads(response.text)
    except Exception as e:
        print("WARNING: exception occured while parsing response: " + str(e))
        return None   

# function for sending an HTTP POST request containing multiple encoded images
# response will contain detection results in JSON
def request_detect_and_track_multiple(images, address, path="/api/detect_and_track_multiple"):
    # prepare URL and headers for http request
    url = address + path
    content_type = 'application/json'
    headers = {'content-type': content_type}
    
    # send the HTTP post request with encoded image
    images_encoded = []
    for image in images:
        image_encoded = cv2.imencode('.png', image)[1]
        images_encoded.append(image_encoded)
    try:
        response = requests.post(url, data=jsonpickle.encode(images_encoded), headers=headers)
    except Exception as e:
        print("WARNING: exception occured while sending request: " + str(e))
        return None

    # return the response JSON    
    try:
        status = response.status_code
        if status != 200:
            raise Exception("Received unexpected HTTP status code " + str(status))
        return json.loads(response.text)
    except Exception as e:
        print("WARNING: exception occured while parsing response: " + str(e))
        return None   


# handler function for processing HTTP POST requests containing a single encoded image
# response will contain detection results in a single encoded image
def request_detect_track_annotate_single(image, address, path="/api/detect_track_annotate_single"):
    # prepare URL and headers for http request
    url = address + path
    content_type = 'image/png'
    headers = {'content-type': content_type}

    # send the HTTP post request with encoded image
    image_encoded = cv2.imencode('.png', image)[1]
    try:
        response = requests.post(url, data=np.array(image_encoded).tobytes(), headers=headers)
    except Exception as e:
        print("WARNING: exception occured while sending request: " + str(e))
        return None

    # return the response image    
    try:
        status = response.status_code
        if status != 200:
            raise Exception("Received unexpected HTTP status code " + str(status))
        image_response = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        return image_response
    except Exception as e:
        print("WARNING: exception occured while parsing response: " + str(e))
        return None  
