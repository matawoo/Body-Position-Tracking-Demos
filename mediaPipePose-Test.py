#Pip Libraries required
#!pip install mediapipe opencv-python


import cv2
import math
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose_goal = mp.solutions.pose
mp_pose_user = mp.solutions.pose

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def vector(p1, p2):
    return np.array(p1) - np.array(p2)

def get_angles(landmarks):
    poi = [[11, 12, 13],
           [12, 11, 14],
           [13, 15, 11],
           [14, 16, 12],
           [23, 11, 25],
           [24, 12, 26],
           [25, 23, 27],
           [26, 24, 28]]
    angles = np.zeros((33,3))
    for i in range(len(poi)):
        try:
            angles[poi[i][0]] = angle(vector([landmarks[poi[i][0]].z, landmarks[poi[i][0]].y, landmarks[poi[i][0]].x],
                                             [landmarks[poi[i][1]].z, landmarks[poi[i][1]].y, landmarks[poi[i][1]].x]),
                                      vector([landmarks[poi[i][0]].z, landmarks[poi[i][0]].y, landmarks[poi[i][0]].x],
                                             [landmarks[poi[i][2]].z, landmarks[poi[i][2]].y, landmarks[poi[i][2]].x]))
        except:
            pass

    return angles

def point_in_ranqge(a1, a2):
    if (a1[0] > (a2[0] * .9) and a1[0] > (a2[0] * 1.1)):
        return False
    elif (a1[1] > (a2[1] * .9) and a1[1] > (a2[1] * 1.1)):
        return False
    elif (a1[2] > (a2[2] * .9) and a1[2] > (a2[2] * 1.1)):
        return False
    return True

# For webcam input:
cap_g = cv2.VideoCapture(1)
cap_u = cv2.VideoCapture(1)

frame_width = int(cap_u.get(3))
frame_height = int(cap_u.get(4))

with mp_pose_goal.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_g:
    with mp_pose_user.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_u:
      while cap_g.isOpened() and cap_u.isOpened():
        success_g, img_g = cap_g.read()
        success_u, img_u = cap_u.read()
        if not success_g:
          print("Ignoring empty camera frame from GOAL.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        if not success_u:
          print("Ignoring empty camera frame from USER.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        img_g = cv2.cvtColor(cv2.flip(img_g, 1), cv2.COLOR_BGR2RGB)
        img_u = cv2.cvtColor(cv2.flip(img_u, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img_g.flags.writeable = False
        img_u.flags.writeable = False
        results_g = pose_g.process(img_g)
        results_u = pose_u.process(img_u)
        # Draw the pose annotation on the image.
        img_g.flags.writeable = True
        img_u.flags.writeable = True
        img_g = cv2.cvtColor(img_g, cv2.COLOR_RGB2BGR)
        img_u = cv2.cvtColor(img_u, cv2.COLOR_RGB2BGR)

        width = img_u.shape[1]
        height = img_u.shape[0]
        width_reduced = int(width * 30 / 100)
        height_reduced = int(height * 30 / 100)  # Sets the image to be 20% of the user image
        img_g_small = cv2.resize(img_g, (width_reduced, height_reduced), interpolation=cv2.INTER_AREA)

        try:
            angles_g = get_angles(results_g.pose_world_landmarks.landmark)
            angles_u = get_angles(results_u.pose_world_landmarks.landmark)
            landmarks = results_g.pose_landmarks.landmark
            poi = [11, 12, 13, 14, 23, 24, 25, 26]

            for i in range(len(poi)):
                if point_in_range(angles_u[poi[i]], angles_g[poi[i]]):
                    img_u = cv2.circle(img_u, (int(width * landmarks[poi[i]].x), int(height * landmarks[poi[i]].y)), 5,
                                       (0, 255, 0), -1)
                else:
                    img_u = cv2.circle(img_u, (int(width * landmarks[poi[i]].x), int(height * landmarks[poi[i]].y)), 5,
                                       (0, 0, 255), -1)
        except:
            pass


        mp_drawing.draw_landmarks(img_u, results_u.pose_landmarks, mp_pose_user.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        mp_drawing.draw_landmarks(img_g, results_g.pose_landmarks, mp_pose_user.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )


        img_u[height-height_reduced:height , width-width_reduced:width] = img_g_small

        cv2.imshow('MediaPipe Pose', img_u)
        if cv2.waitKey(10) & 0xFF == ord('q'):
          break
cap_g.release()
cap_u.release()