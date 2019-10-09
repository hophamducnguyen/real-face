# USAGE
# use to capture face, hit <space> to capture, hit <esc> to exit
# python face_capture.py -n <name>
import os
import cv2
import argparse

import face_recognition

from face_config import image_dir, face_classifier, eye_classifier

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str, required=True,
                help="name of recorded person.")

args = vars(ap.parse_args())
name = args["name"]


# create images folder
record_dir = image_dir + name + '/real/'
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
        color = (255, 0, 0)  # BGR 0-255
        stroke = 0
        cv2.rectangle(frame, (left, top), (right, bottom), color, stroke)

    cv2.imshow('frame', frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    # elif k % 256 == 32:
    elif k % 256 == 32 and color_face is not None:
        # SPACE pressed
        img_name = record_dir + "{}.png".format(img_counter)
        cv2.imwrite(img_name, color_face)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
