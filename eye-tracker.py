import cv2
import mediapipe as mp

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

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Eye tracker", image)

    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()