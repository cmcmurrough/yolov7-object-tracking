import requests
import json
import cv2
import numpy as np
import jsonpickle
import sys
from flask import jsonify
from restfulYOLOv7 import request_detect_and_track_single

# program entry point
if __name__ == "__main__":

    # parse the command line arguments
    print(sys.argv)
    try:
        video_source = sys.argv[1]
        address = "http://10.0.0.204:5000"
        display = True
        save_video = True
    except:
        print("ERROR: unable to parse command line arguments")
        print("USAGE: " + os.path.basename(sys.argv[0]) + " <video_path> <address>")
        exit()

    # open the video capture
    try:
        capture = cv2.VideoCapture(video_source)
        capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
    except:
        print("ERROR: unable to open video source " + video_source)
        exit()

    # open the output video writer
    if save_video:
        video_writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (capture_width, capture_height))

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

        # convert to grayscale
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # apply adaptive equalization
        #clahe_model = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        #colorimage_b = clahe_model.apply(frame[:,:,0])
        #colorimage_g = clahe_model.apply(frame[:,:,1])
        #colorimage_r = clahe_model.apply(frame[:,:,2])
        #frame = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)
             
        # check to see if we received a valid frame
        if success:

            t0 = cv2.getTickCount()
            response = request_detect_and_track_single(frame, address)
            t1 = cv2.getTickCount()
            time_elapsed = (t1 - t0) / cv2.getTickFrequency()
            print("REST request completed in: " + str(time_elapsed) + " seconds")

            # annotate the video with JSON results
            display=True
            #print(response['inferences'])
            for item in response['inferences']:
                print(item)

                # extract the inference fields
                x1 = item['x1']
                y1 = item['y1']
                x2 = item['x2']
                y2 = item['y2']
                cx = item['cx']
                cy = item['cy']
                class_id = item['class_id']
                class_name = item['class_name']
                instance_id = item['instance_id']
                label = item['label']
                
                # annotate the image
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, class_name, (x1, y1), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, str(instance_id), (cx-20, cy), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # display the frame and write to video
            if display:
                cv2.imshow("frame", frame)
            if save_video:
                video_writer.write(frame)

            # check for 'q' key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("terminating program")
                break
        else:
            print("WARNING: unable to retrieve frame")
            capture.release()
            if save_video:
                video_writer.release()
            exit()