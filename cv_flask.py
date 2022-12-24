import cv2
from datetime import datetime
import numpy as np
from flask import Flask, request, Response, jsonify
import jsonpickle

api = Flask(__name__)

# handler function for processing HTTP POST requests containing a single encoded image
# response will contain processing results in JSON
@api.route('/api/test_single_image_post', methods=['POST'])
def test_single_image_post():
    # convert string request data to uint8 and decode to image
    image = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)

    # perform image processing
    print("Image processing started at: UTC " + str(datetime.utcnow()))
    start_time = datetime.now()
    result_image = image
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Image processing completed in: " + str(elapsed_time.total_seconds()) + " seconds")

    # create the response and send to client
    response = {'processing_time': str(elapsed_time.total_seconds())}
    return jsonify(response)

# handler function for processing HTTP POST requests containing a single encoded image
# response will contain processing results in a single encoded image
@api.route('/api/detect_and_track_with_annotate', methods=['POST'])
def detect_and_track_with_annotate():
    # convert string request data to uint8 and decode to image
    image = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)

    # perform image processing
    print("Image processing started at: UTC " + str(datetime.utcnow()))
    start_time = datetime.now()
    result_image = image
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Image processing completed in: " + str(elapsed_time.total_seconds()) + " seconds")

    # create the response and send to client
    image_encoded = cv2.imencode('.png', result_image)[1]
    content_type = 'image/png'
    content = np.array(image_encoded).tobytes()
    response = Response(status=200, mimetype=content_type)
    response.set_data(content)
    return response

# handler function for processing HTTP POST requests containing multiple encoded images
# response will contain detection results in JSON
@api.route('/api/test_multiple_image_post', methods=['POST'])
def test_multiple_image_post():
    # decode the serialized array of images
    images_encoded = jsonpickle.decode(request.data)
    images = []
    for image_encoded in images_encoded:
        images.append(cv2.imdecode(image_encoded, cv2.IMREAD_COLOR))

    # run detection algorithm
    print("Detection started at: UTC " + str(datetime.utcnow()))
    start_time = datetime.now()
    for i, image in enumerate(images):
        cv2.imshow(str(i), image)
    cv2.waitKey()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Detection completed in: " + str(elapsed_time.total_seconds()) + " seconds")

    # create the response and send to client
    response = {'processing_time': str(elapsed_time.total_seconds())}
    return jsonify(response)

# program entry point
if __name__ == '__main__':

    # start the server
    api.run(host="0.0.0.0", port=5000)
