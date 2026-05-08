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

    # Grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # This uses the model to get the faces from the webcam's image
    faces = face_cascade.detectMultiScale(image=gray_image, scaleFactor=1.5, minNeighbors=3)

    for (x, y, w, h) in faces:
        # Put a bounding box just for reference
        # cv2.rectangle(img=image, pt1=(x, y), pt2=(x+w, y+h), color=(0,0,0))

        # Only selects the face
        roi = image[x:x+w, y:y+h]
        # Takes the face and blurs it and makes that the image at the roi
        image[x:x+w, y:y+h] = cv2.GaussianBlur(src=roi, ksize=(99, 99), sigmaX=0)
        # I had trouble with this line ^^^^^^ because I had my source wrong for the GaussianBlur (GB)
        # Before, I the GB blurring the image[roi] which doesn't make any sense because roi is the image itself, only a smaller part of it
        # Then I realized that and so I changed the source to roi. I also had my image wrong. Before I had the image[x:x+w, y:y+h], I had
        # image[roi] which also doesn't make any sense. Once I realized this, it was straight forward from there.
        # ksize is how blurry you want it to be. Still unsure about how sigmaX works



    # Display video
    cv2.imshow("Blurred face", image)

    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()