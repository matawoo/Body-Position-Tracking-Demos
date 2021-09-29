#Pip Libraries required
#!pip install mediapipe opencv-python


import cv2
import math
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose_user = mp.solutions.pose

# For webcam input:
cap_u = cv2.VideoCapture(1)

frame_width = int(cap_u.get(3))
frame_height = int(cap_u.get(4))
with mp_pose_user.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_u:
  while cap_u.isOpened():
    success_u, img_u = cap_u.read()
    if not success_u:
      print("Ignoring empty camera frame from USER.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    img_u = cv2.cvtColor(cv2.flip(img_u, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    img_u.flags.writeable = False
    results_u = pose_u.process(img_u)
    # Draw the pose annotation on the image.

    img_u.flags.writeable = True
    img_u = cv2.cvtColor(img_u, cv2.COLOR_RGB2BGR)

    width = img_u.shape[1]
    height = img_u.shape[0]
    width_reduced = int(width * 30 / 100)
    height_reduced = int(height * 30 / 100)  # Sets the image to be 20% of the user image


    mp_drawing.draw_landmarks(img_u, results_u.pose_landmarks, mp_pose_user.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

    cv2.imshow('BodyPose', img_u)
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
cap_u.release()