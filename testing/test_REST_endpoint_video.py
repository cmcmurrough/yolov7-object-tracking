import requests
import json
import cv2
import numpy as np
import jsonpickle
import sys
from flask import jsonify

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
if __name__ == "__main__":

    # parse the command line arguments
    print(sys.argv)
    try:
        video_source = sys.argv[1]
        address = "http://10.0.0.205:5000"
        display = False
    except:
        print("ERROR: unable to parse command line arguments")
        print("USAGE: " + os.path.basename(sys.argv[0]) + " <video_path> <address>")
        exit()

    # open the video capture
    try:
        capture = cv2.VideoCapture(video_source)
        capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except:
        print("ERROR: unable to open video source " + video_source)
        exit()

    # begin polling loop
    while(True):

        # capture the video frame
        success, frame = capture.read()

        # downsample the frame
        #scale = 0.5
        #width = int(frame.shape[1] * scale)
        #height = int(frame.shape[0] * scale)
        #dim = (width, height)
        #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        # check to see if we received a valid frame
        if success:

            t0 = cv2.getTickCount()
            #response = test_single_image_post_endpoint_with_response_image(frame, address, "/api/detect_and_track_with_annotate")
            response = test_single_image_post_endpoint(frame, address, "/api/test_single_image_post")
            #print(response)
            t1 = cv2.getTickCount()
            time_elapsed = (t1 - t0) / cv2.getTickFrequency()
            print("REST request completed in: " + str(time_elapsed) + " seconds")

            # display the frame
            if display:
                cv2.imshow("result", response)

            # check for 'q' key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("terminating program")
                break
        else:
            print("WARNING: unable to retrieve frame")