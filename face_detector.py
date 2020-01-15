import os
import sys

import cv2
import face_recognition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "dataset/images/")
cascades_dir = os.path.join(BASE_DIR, "cascades/")

face_cascade = cascades_dir + 'haarcascade_frontalface_default.xml'
eye_cascade = cascades_dir + 'haarcascade_eye.xml'
smile_cascade = cascades_dir + 'haarcascade_smile.xml'

detected_frame_color = (255, 90, 90)  # BGR 0-255
detected_frame_stroke = 2

cascades = [face_cascade, eye_cascade, smile_cascade]
for cascade_path in cascades:
    if not os.path.exists(cascade_path):
        sys.exit("cascade_path not exists: " + cascade_path)

face_classifier = cv2.CascadeClassifier(face_cascade)
eye_classifier = cv2.CascadeClassifier(eye_cascade)
smile_classifier = cv2.CascadeClassifier(smile_cascade)


def capture_frame(input_frame, img_dir='capture_images/', counter=0, prefix_name=''):
    img_name = img_dir + "{}{}.png".format(prefix_name, counter)
    cv2.imwrite(img_name, input_frame)
    print("{} written.".format(img_name))
    return counter + 1


def detect_by_cascade(frame):
    """
    Face detection using Haar Cascade Classifiers algorithm with OpenCV
    :param frame:
    :return: list faces with (x, y, end_x, end_y) / (left, top, right, bottom)
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), detected_frame_color, detected_frame_stroke)

        # eyes detect
        eyes = eye_classifier.detectMultiScale(gray_frame)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), detected_frame_color, detected_frame_stroke)
            # # smile detect
            # smiles = smile_classifier.detectMultiScale(gray_frame)
            # for (sx, sy, sw, sh) in smiles:
            #     cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), detected_frame_stroke)


def detect_by_hog(frame):
    """
    Face detection using Histogram of Oriented Gradients algorithm with Dlib
    :param frame:
    :return: list faces with (x, y, end_x, end_y) / (left, top, right, bottom)
    """
    return __face_recognition_detect(frame)


def detect_by_cnn(frame):
    """
    Face detection using Convolutional Neural Networks algorithm with Dlib
    :param frame:
    :return: list faces with (x, y, end_x, end_y) / (left, top, right, bottom)
    """
    return __face_recognition_detect(frame, model="cnn")


def __face_recognition_detect(frame, model="hog"):
    color_face = None
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_recognition.face_locations(gray_frame, model=model)
    for (top, right, bottom, left) in faces:
        color_face = frame[top + 1: bottom - 1, left + 1: right - 1]
        cv2.rectangle(frame, (left, top), (right, bottom), detected_frame_color, detected_frame_stroke)
    return color_face


def compare_faces(known_face_encodings, face_encoding_to_check):
    return face_recognition.compare_faces(known_face_encodings, face_encoding_to_check)
