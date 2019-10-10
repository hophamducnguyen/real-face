# USAGE
# use to capture face, hit <space> to capture, hit <esc> to exit
# python face_capture.py -n <name> -a <number>
# if input number of auto, it will auto capture your face

import argparse
import os

import cv2
import face_recognition

# construct the argument parse and parse the arguments
from face_detector import capture_frame, image_dir

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str, required=True,
                help="name of recorded person.")
ap.add_argument("-a", "--auto", type=int, default=-1,
                help="number of images which will auto capture.")

args = vars(ap.parse_args())
user_name = args["name"]
max_img_counter = args["auto"]

# create images folder
record_dir = image_dir + user_name + '/real/'
os.makedirs(record_dir, exist_ok=True)

# open cam and capture images
cam = cv2.VideoCapture(0)
# cam.set(3, 640)  # set Width
# cam.set(4, 480)  # set Height

img_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(gray_frame)
    # print(faces)
    color_face = None
    for (top, right, bottom, left) in faces:
        color_face = frame[top + 1: bottom - 1, left + 1: right - 1]
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 0)

        # auto capture
        if 0 <= img_counter <= max_img_counter:
            # img_counter = capture_frame(color_face, record_dir, img_counter)
            img_counter = capture_frame(color_face)

    cv2.imshow('frame', frame)

    if not ret:
        break

    # wait for input key
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Esc hit, exit program and cleanup.")
        break
    elif k % 256 == 32 and color_face is not None:
        # SPACE pressed
        img_counter = capture_frame(color_face, record_dir, img_counter)

cam.release()
cv2.destroyAllWindows()
