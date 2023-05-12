#importing necessary packages
from scipy.spatial import distance
from imutils import face_utils
import time
import dlib
import cv2
import pygame

#font
font = cv2.FONT_HERSHEY_SIMPLEX

#music
pygame.mixer.init()
pygame.mixer.music.load('mixkit-appliance-ready-beep-1076.wav')

#minimum eye threshold
eye_ratio = 0.3

#minimum consecutive frames
eye_frames = 12

#initialize the count
count = 0

#load face cascade file
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#function to calculate and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    C = distance.euclidean(eye[0],eye[3])

    ear = (A+B)/(2*C)
    return ear

#load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#extract indexes of facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#start the camera
cam = cv2.VideoCapture(0)

while True:
    #reading the frames
    frame = cam.read()[1]
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame.shape[:2]

    #face detection through detection function
    faces = detector(gray,0)

    #detect faces through haar cascade xml
    face_rect = face_cas.detectMultiScale(gray,1.3,5)

    #draw rectangle around the face
    for (x,y,w,h) in face_rect:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

    #detect facial points
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #get coordinates of left and right eye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #calculating the eye aspect ratio
        left_EAR = eye_aspect_ratio(leftEye)
        right_EAR = eye_aspect_ratio(rightEye)

        eye_AR = (left_EAR + right_EAR) / 2

        #draw eye shape
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,0,255),1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,0,255),1)

        #if eye aspect ratio is less than threshold
        if (eye_AR < eye_ratio):
            count+=1

            #if no of frames is greater than threshold
            if count >= eye_frames:
                #cv2.putText(frame,"Count:"+str(count),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                pygame.mixer.music.play(-1)
                #cv2.putText(frame,"Drowsy",(10, height-20), font, 1, (255,255,255),1,cv2.LINE_AA)
        else:
            pygame.mixer.music.stop()
            count = 0

    #show live video feed
    cv2.imshow('Video',frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

#close the camera
cam.release()
cv2.destroyAllWindows()
