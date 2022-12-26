import requests
import json
import cv2
import numpy as np
import jsonpickle
from flask import jsonify
from restfulYOLOv7 import request_detect_and_track_single, request_detect_and_track_multiple, request_detect_track_annotate_single

# program entry point
if __name__ == '__main__':
    address = "http://10.0.0.205:5000"
    
    # test the single image post request
    print("TEST 1: /api/detect_and_track_single")
    image = cv2.imread('test_1.jpg')
    json_response = request_detect_and_track_single(image, address)
    print(json.dumps(json_response, indent=2))
    
    # test the multiple image post request
    print("TEST 2: /api/detect_and_track_multiple")
    images = [cv2.imread('test_1.jpg'), cv2.imread('test_2.jpg'), cv2.imread('test_3.jpg')]
    json_response = request_detect_and_track_multiple(images, address)
    print(json.dumps(json_response, indent=2))
    
    # test the single image post request with return image
    print("TEST 3: /api/detect_track_annotate_single")
    image = cv2.imread('test_3.jpg')
    response_image = request_detect_track_annotate_single(image, address)
    cv2.imshow("response image", response_image)
    cv2.waitKey()
