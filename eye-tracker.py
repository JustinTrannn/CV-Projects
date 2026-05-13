import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    rbg_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rbg_image)

    cv2.imshow("Eye tracker", image)

    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()