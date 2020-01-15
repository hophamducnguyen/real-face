import cv2

import face_detector

cam = cv2.VideoCapture(0)
# cam.set(3, 640)  # set Width
# cam.set(4, 480)

color = (255, 90, 90)
stroke = 2

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detect_by_cascade(frame)
    # faces = face_detector.detect_by_hog(frame)
    cv2.imshow('frame', frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()
cv2.destroyAllWindows()
