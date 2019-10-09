import os
import sys

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "dataset/images/")
cascades_dir = os.path.join(BASE_DIR, "cascades/")

face_cascade = cascades_dir + 'haarcascade_frontalface_default.xml'
eye_cascade = cascades_dir + 'haarcascade_eye.xml'
smile_cascade = cascades_dir + 'haarcascade_smile.xml'

cascades = [face_cascade, eye_cascade, smile_cascade]
for cascade_path in cascades:
    if not os.path.exists(cascade_path):
        sys.exit("cascade_path not exists: " + cascade_path)

face_classifier = cv2.CascadeClassifier(face_cascade)
eye_classifier = cv2.CascadeClassifier(eye_cascade)
smile_classifier = cv2.CascadeClassifier(smile_cascade)