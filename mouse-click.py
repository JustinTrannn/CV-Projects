import cv2
import pyautogui
import mediapipe as mp
import numpy as np

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

frame_timestamp = 0

while True:
    # Converting images
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # ---------- Get landmarker to do its thing ----------
    # Detect hands
    result = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 33

    # Put circles around your fingers
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = image.shape

            # Pointer finger (8) and thumb (4)
            x1 = int(hand_landmarks[8].x * w)
            y1 = int(hand_landmarks[8].y * h)

            x2 = int(hand_landmarks[4].x * w)
            y2 = int(hand_landmarks[4].y * h)

            cv2.circle(image, (x1, y1), 10, (0, 255, 0), 2)
            cv2.circle(image, (x2, y2), 10, (0, 255, 0), 2)

            distance = int(np.hypot(x2-x1, y2-y1))
            cv2.putText(image, f"Distance: {distance}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
            if distance < 40:
                pyautogui.click(interval=0.5)
                

            screen_w, screen_h = pyautogui.size()

            mouse_x = int(hand_landmarks[8].x * screen_w)
            mouse_y = int(hand_landmarks[8].y * screen_h)

            pyautogui.moveTo(mouse_x, mouse_y, duration=0.01)

    # ----------                                ----------


    cv2.imshow("Webcam", image)
    key = cv2.waitKey(1)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()