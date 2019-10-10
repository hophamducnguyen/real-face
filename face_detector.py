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


def cascade_detect(frame):
    """
    Face detection using Haar Cascade Classifiers algorithm with OpenCV
    :param frame:
    :return: list faces with (x, y, end_x, end_y)
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_classifier.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))


def hog_detect(frame):
    """
    Face detection using Histogram of Oriented Gradients algorithm with Dlib
    :param frame:
    :return:
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_recognition.face_locations(gray_frame)


def cnn_detect(frame):
    """
    Face detection using Convolutional Neural Networks algorithm with Dlib
    :param frame:
    :return:
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_recognition.face_locations(gray_frame, model="cnn")


def compare_faces(known_face_encodings, face_encoding_to_check):
    return face_recognition.compare_faces(known_face_encodings, face_encoding_to_check)

