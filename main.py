from ultralytics import YOLO
import cv2
import argparse

import util
from sort.sort import *
from util import read_monitor, write_csv_monitor


# 讀輸入檔名
parser = argparse.ArgumentParser(description="Process an input video file.")
parser.add_argument('-input', type=str, required=True, help='Path to the input video file')
args = parser.parse_args()
input_path = args.input

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
cap = cv2.VideoCapture(input_path)

monitors = [62]

# read frames
frame_nmr = -1
ret = True

MAX_FRAMES = 100

for idx in range(MAX_FRAMES):
    if not ret:
        break
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect monitors
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in monitors:
                detections_.append([x1, y1, x2, y2, score])

        monitor_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
        monitor_crop_gray = cv2.cvtColor(monitor_crop, cv2.COLOR_BGR2GRAY)
        _, monitor_crop_thresh = cv2.threshold(monitor_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
        monitor_text, monitor_text_score = read_monitor(monitor_crop_thresh)

        if monitor_text is not None:
            results[frame_nmr] = {'bbox': [x1, y1, x2, y2], 'text': monitor_text, 'text_score': monitor_text_score}

"""
        # track monitors
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
"""
                                                                    

# write results
write_csv_monitor(results, './test.csv')