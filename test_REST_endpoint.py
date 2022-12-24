import requests
import json
import cv2
import numpy as np

# function for sending an HTTP POST request containing a single encoded image
# response will contain detection results in JSON
def test_single_image_post_endpoint(image, address, path):
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
def test_multiple_image_post_endpoint(images, address, path):
    # prepare URL and headers for http request
    url = address + path
    content_type = 'image/png'
    headers = {'content-type': content_type}
    
    # send the HTTP post request with encoded image
    images_encoded = []
    for image in images:
        image_encoded = cv2.imencode('.png', image)[1]
        images_encoded.append(image_encoded)
    try:
        response = requests.post(url, data=np.array(images_encoded, dtype=object).tobytes(), headers=headers)
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
def test_single_image_post_endpoint_with_response_image(image, address, path):
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


# program entry point
if __name__ == '__main__':
    address = "http://10.0.0.205:5000"
    
    # test the single image post request
    image = cv2.imread('test.jpg')
    endpoint = "/api/test_single_image_post"
    #json_response = test_single_image_post_endpoint(image, address, endpoint)
    #print(json.dumps(json_response, indent=2))
    
    # test the multiple image post request
    image = cv2.imread('test.jpg')
    endpoint = "/api/test_multiple_image_post"
    json_response = test_multiple_image_post_endpoint(image, address, endpoint)
    print(json.dumps(json_response, indent=2))
    
    # test the single image post request with return image
    image = cv2.imread('test.jpg')
    endpoint = "/api/detect_and_track_with_annotate"
    #response_image = test_single_image_post_endpoint_with_response_image(image, address, endpoint)
    #cv2.imshow("response image", response_image)
    cv2.waitKey()
