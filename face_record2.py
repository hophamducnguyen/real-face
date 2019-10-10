import os
import cv2
import argparse

import face_recognition

import face_detector
from face_detector import image_dir, face_classifier, eye_classifier

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
cam.set(3, 640)  # set Width
cam.set(4, 480)  # set Height

img_counter = 0
# face_recognition.face_distance()

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))
    # faces = face_classifier.detectMultiScale(gray_frame)
    # faces = face_recognition.face_locations(gray_frame)
    # print(faces)
    color_face = None
    # for ((top, right, bottom, left), name) in zip(boxes, names):
    # for (top, right, bottom, left) in faces:
    # for (y, x, w, h) in faces:
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        gray_face = gray_frame[y:y + h, x:x + w]
        color_face = frame[y:y + h, x:x + w]
        rgb_face = frame[y:y + h, x:x + w]

        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        # cv2.rectangle(frame, (left, top), (right, bottom), color, stroke)
        # cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)

        eyes = eye_classifier.detectMultiScale(gray_frame)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

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
        cv2.imwrite(img_name, gray_frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
