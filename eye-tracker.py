import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LinearRegression

# Documentation here: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python#video
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = FaceLandmarker.create_from_options(options)

model_x = LinearRegression()
model_y = LinearRegression()
trained = False

webcam = cv2.VideoCapture(0)
cv2.namedWindow("Eye tracker", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Eye tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
frame_timestamp = 0

x_coordinates = []
y_coordinates = []
point_1_done = False
point_2_done = False
point_3_done = False
point_4_done = False
point_5_done = False

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_image)

    result = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 33
    
    key = cv2.waitKey(10)

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            h, w, _ = image.shape

            # Left iris (LI): [474, 475, 476, 477]
            # Right iris (RI): [469, 470, 471, 472]
            LI_1_x = int(face_landmarks[474].x * w)
            LI_1_y = int(face_landmarks[474].y * h)
            LI_2_x = int(face_landmarks[475].x * w)
            LI_2_y = int(face_landmarks[475].y * h)
            LI_3_x = int(face_landmarks[476].x * w)
            LI_3_y = int(face_landmarks[476].y * h)
            LI_4_x = int(face_landmarks[477].x * w)
            LI_4_y = int(face_landmarks[477].y * h)
            LI_avg_x = int(np.mean([LI_1_x, LI_2_x, LI_3_x, LI_4_x]))
            LI_avg_y = int(np.mean([LI_1_y, LI_2_y, LI_3_y, LI_4_y]))

            cv2.circle(image, (LI_1_x, LI_1_y), 1, (0, 255, 0), 1) # green
            cv2.circle(image, (LI_2_x, LI_2_y), 1, (0, 255, 0), 1) # green
            cv2.circle(image, (LI_3_x, LI_3_y), 1, (0, 255, 0), 1) # green
            cv2.circle(image, (LI_4_x, LI_4_y), 1, (0, 255, 0), 1) # green
            cv2.circle(image, (LI_avg_x, LI_avg_y), 2, (255, 255, 255), 1) # white

            RI_1_x = int(face_landmarks[469].x * w)
            RI_1_y = int(face_landmarks[469].y * h)
            RI_2_x = int(face_landmarks[470].x * w)
            RI_2_y = int(face_landmarks[470].y * h)
            RI_3_x = int(face_landmarks[471].x * w)
            RI_3_y = int(face_landmarks[471].y * h)
            RI_4_x = int(face_landmarks[472].x * w)
            RI_4_y = int(face_landmarks[472].y * h)
            RI_avg_x = int(np.mean([RI_1_x, RI_2_x, RI_3_x, RI_4_x]))
            RI_avg_y = int(np.mean([RI_1_y, RI_2_y, RI_3_y, RI_4_y]))

            cv2.circle(image, (RI_1_x, RI_1_y), 1, (0, 255, 0), 1) # green
            cv2.circle(image, (RI_2_x, RI_2_y), 1, (0, 255, 0), 1) # green
            cv2.circle(image, (RI_3_x, RI_3_y), 1, (0, 255, 0), 1) # green
            cv2.circle(image, (RI_4_x, RI_4_y), 1, (0, 255, 0), 1) # green
            cv2.circle(image, (RI_avg_x, RI_avg_y), 2, (255, 255, 255), 1) # white

            # ---- Point 1
            if not point_1_done:
                cv2.circle(image, (10, 10), 5, (0, 0, 0), 6)
                cv2.putText(image, "Look at point 1/5 and press c to calibrate", (100, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                if key == ord('c'):
                    X.append([LI_1_x, LI_2_x, LI_3_x, LI_4_x, LI_avg_x])
                    point_1_done = True
            # ---- Point 2
            elif not point_2_done:
                cv2.circle(image, (10, h-10), 5, (0, 0, 0), 6)
                cv2.putText(image, "Look at point 2/5 and press c to calibrate", (100, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                if key == ord('c'):
                    X.append([LI_1_x, LI_2_x, LI_3_x, LI_4_x, LI_avg_x])
                    point_2_done = True
            # ---- Point 3
            elif not point_3_done:
                cv2.circle(image, (w // 2, h // 2), 5, (0, 0, 0), 6)
                cv2.putText(image, "Look at point 3/5 and press c to calibrate", (100, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                if key == ord('c'):
                    X.append([LI_1_x, LI_2_x, LI_3_x, LI_4_x, LI_avg_x])
                    point_3_done = True
            # ---- Point 4
            elif not point_4_done:
                cv2.circle(image, (w-10, 10), 5, (0, 0, 0), 6)
                cv2.putText(image, "Look at point 4/5 and press c to calibrate", (100, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                if key == ord('c'):
                    X.append([LI_1_x, LI_2_x, LI_3_x, LI_4_x, LI_avg_x])
                    point_4_done = True
            # ---- Point 5
            elif not point_5_done:
                cv2.circle(image, (w-10, h-10), 5, (0, 0, 0), 6)
                cv2.putText(image, "Look at point 5/5 and press c to calibrate", (100, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                if key == ord('c'):
                    X.append([LI_1_x, LI_2_x, LI_3_x, LI_4_x, LI_avg_x])
                    point_5_done = True
                    print(X)

    cv2.imshow("Eye tracker", image)

    if key == 27:
        break



webcam.release()
cv2.destroyAllWindows()