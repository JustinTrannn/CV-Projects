import cv2
import mediapipe as mp
import numpy as np

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

webcam = cv2.VideoCapture(0)
frame_timestamp = 0

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_image)

    result = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 33
    
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

    cv2.imshow("Eye tracker", image)

    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()