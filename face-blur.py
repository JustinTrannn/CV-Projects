import cv2

# -------------
# I will try to attempt this with just the knowledge and code I currently have
# -------------

# This is the model I will use
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open camera
webcam = cv2.VideoCapture(0)

while True:
    # Getting the image from the webcam and then mirroring it
    _, image = webcam.read()
    image = cv2.flip(image, 1)

    # This uses the model to get the faces from the webcam's image
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.5, minNeighbors=2)

    for (x, y, w, h) in faces:
        # Only selects the dimensions of the face
        roi = image[x:x+w, y:y+h]
        # Takes the face and blurs it and makes that the image at the roi
        image[roi] = cv2.GaussianBlur(src=image[roi], ksize=(16, 16), sigmaX=0)
        


    # Display video
    cv2.imshow("Blurred face", image)

    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()