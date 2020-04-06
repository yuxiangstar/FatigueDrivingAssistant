# -*- coding:utf-8 -*-
import cv2
import os
import time
import base64
import numpy as np
import dlib
from imutils import face_utils
# faceCascade = cv2.CascadeClassifier(
#     'cascades/haarcascade_frontalface_alt.xml')
faceCascade2 = cv2.CascadeClassifier(
    'cascades/haarcascade_profileface.xml')

eyeCascade = cv2.CascadeClassifier(
    'cascades/haarcascade_eye_tree_eyeglasses.xml')
pwd = os.getcwd()
model_path = os.path.join(pwd, 'model')
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

LEFT_POINTS = 26
RIGHT_POINTS = 17
TOP_POINTS = 19
CENTRE_POINTS = 29
count = 0
i = 80
START_TIME = 85
END_TIME = 500
EYE_ADDRESS = "photo/newtest2.0/0.{0}.jpg"
FACE_ADDRESS = "photo/chuli2/0.{0}.jpg"
faceflag = 0
zheng = []
def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    #dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

while True:
    ret, frame = cap.read()
    # frame = Contrast_and_Brightness(0.5, -0.5, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 0)
    print(i)
    for rect in rects:
        print("zheng")
        faceflag = 1
        # cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 0, 255), 2)
        shape = predictor(gray, rect)
        points = face_utils.shape_to_np(shape)  # 特征点队列
        lpoint = points[LEFT_POINTS]
        rpoint = points[RIGHT_POINTS]
        cpoint = points[CENTRE_POINTS]
        tpoint = points[TOP_POINTS]
        # cv2.rectangle(frame, (lpoint[0], tpoint[1]), (rpoint[0], cpoint[1]), (0, 255, 0), 2)
        face = frame[rect.top():rect.bottom(), rect.left():rect.right()]
        # print(rect.top(), rect.bottom(), rect.left(), rect.right())
        eye = frame[tpoint[1]:cpoint[1], rpoint[0]:lpoint[0]]
        # print(tpoint[1], cpoint[1], lpoint[0], rpoint[0])
        listStr = [str(int(time.time())), str(count)]
        i += 1
        eye_filename = EYE_ADDRESS.format(i)
        face_filename = FACE_ADDRESS.format(i)

        if i >= START_TIME:
            cv2.imwrite(eye_filename, eye)
            cv2.imwrite(face_filename, face)
            count += 1
    cv2.imshow('opencvCut', frame)
    key = cv2.waitKey(60) & 0xff
    if key == 27 or key == ord('q') or i == END_TIME:
        break

cap.release()
cv2.destroyAllWindows()