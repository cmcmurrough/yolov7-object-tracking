import os
import sys
import torch
import argparse
import cv2
import numpy as np
import traceback
sys.path.append("..")
from detect_and_track import YOLOv7
from detect_and_track import download
from detect_and_track import detect
from detect_and_track import strip_optimizer


# perform one iteration of YOLO inference, 
def process_frame(frame, yolo):

    # downsample the frame
    downsample_scale = 1.0
    if downsample_scale < 1.0:
        width = int(frame.shape[1] * downsample_scale)
        height = int(frame.shape[0] * downsample_scale)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            
    # run the YOLO iteration
    try:
        result_image, inferences = yolo.detect(arguments, frame, False)
        if not inferences:
            raise Exception("request returned invalid response")
    except Exception as e:
        print("WARNING: exception occured while running YOLO inference: " + str(e))
        traceback.print_exc()

    # annotate the video with JSON results
    image_out = frame.copy()
    for item in inferences:
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

        # filter any labels we don't care about
        if class_name != 'order_item_face':
            continue
        
        # annotate the image
        cv2.rectangle(image_out, (x1, y1), (x2, y2), (0,255,0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_out, class_name, (x1, y1), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image_out, str(instance_id), (cx-20, cy), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return image_out


# program entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')
    #parser.add_argument('--api', action='store_true', help='acquire images in REST API mode')
    #parser.add_argument('--lib', action='store_true', help='acquire images in library mode')
    parser.set_defaults(download=True)
    
    arguments = parser.parse_args()
    print(arguments)

    # download model weights if missing
    arguments.weights = arguments.weights[0]
    if arguments.download and not os.path.exists(str(arguments.weights)):
        print('INFO: Model weights not found, attempting download')
        download('./')

    # open the video source
    video_source = arguments.source
    try:
        capture = cv2.VideoCapture(video_source)
        capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture_fps = capture.get(cv2.CAP_PROP_FPS)
        display = False
        save_video = True
        save_video_path = os.path.basename(video_source).replace(".mp4", ".processed.mp4")
    except:
        print("ERROR: unable to open video source " + video_source)
        exit()

    # open the output video writer
    if save_video:
        video_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), capture_fps, (capture_width, capture_height))

    # launch the detection engine
    with torch.no_grad():

        # intialize the library
        print("INFO: LIB mode activated, initializing YOLOv7 engine")
        yolo = YOLOv7(arguments)

        # begin polling loop
        processing = True
        while(processing):

            # capture the video frame
            success, frame = capture.read()
            if success:

                # perform the inference
                t0 = cv2.getTickCount()
                annotated_frame = process_frame(frame, yolo)
                t1 = cv2.getTickCount()
                time_elapsed = (t1 - t0) / cv2.getTickFrequency()
                print("YOLO inference completed in: " + str(time_elapsed) + " seconds")

                # display the frame and write to video
                if display:
                    cv2.imshow("annotated_frame", annotated_frame)
                if save_video:
                    video_writer.write(annotated_frame)

                # check for 'q' key press
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("terminating program")
                    processing = False
                    break
            else:
                print("WARNING: unable to retrieve frame")
                processing = False
                break

        # close streams and save results
        capture.release()
        if save_video:
            video_writer.release()

