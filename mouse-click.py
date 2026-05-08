import cv2
import pyautogui
import mediapipe as mp

# ---------- Mediapipe Tasks Setup ----------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)
# ------------------------------------------

webcam = cv2.VideoCapture(0)

while True:
    
    key = cv2.waitKey(5)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()