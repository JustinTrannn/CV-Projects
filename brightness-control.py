import cv2
import numpy as np
import mediapipe as mp
import screen_brightness_control as sbc

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

# ---------- OpenCV Setup ----------

webcam = cv2.VideoCapture(0)
# Brightness range
min_brightness = 0
max_brightness = 100
# Distance range
min_distance = 20
max_distance = 200

frame_timestamp = 0

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to mediapipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    # Detect hands
    result = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = image.shape
            # Thumb tip (4) and Index tip (8)
            x1 = int(hand_landmarks[4].x * w)
            y1 = int(hand_landmarks[4].y * h)

            x2 = int(hand_landmarks[8].x * w)
            y2 = int(hand_landmarks[8].y * h)

            cv2.circle(image, (x1, y1), 10, (0, 255, 0), 2)
            cv2.circle(image, (x2, y2), 10, (0, 255, 0), 2)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            distance = np.hypot(x2-x1, y2-y1)

            brightness = np.interp(distance, [min_distance, max_distance], [min_brightness, max_brightness])

            sbc.set_brightness(int(brightness))

            cv2.putText(image, f"Brightness: {int(brightness)}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Webcam", image)
    
    key = cv2.waitKey(10)
    if key == 27:
        break



webcam.release()
cv2.destroyAllWindows()