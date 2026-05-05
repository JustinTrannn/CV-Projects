import cv2
# This brings in a pretrained model from a link
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# This opens my webcam. The zero is the default camera 
webcam = cv2.VideoCapture(0)

while True:
    # Grabs a frame (image) from my webcam. The underscore is a T/F to see if it worked. We don't care about this
    _, img = webcam.read()
    # Displays the image using OpenCV
    cv2.imshow("Face detection", img)
    # This waits 10 milliseconds and checks if a key was pressed
    key = cv2.waitKey(10)
    # 27 is the escape key. If escape key is pressed, it will break loop
    if key == 27:
        break

# Frees the webcam from being used
webcam.release()

# Closes all windows created by OpenCV
cv2.destroyAllWindows()