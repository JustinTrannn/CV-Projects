import cv2
# This brings in a pretrained model from a link
# Had to add the cv2.data.haarcascades because it wasn't working with only the .xml file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# This opens my webcam. The zero is the default camera 
webcam = cv2.VideoCapture(0)

while True:
    # Grabs a frame (image) from my webcam. The underscore is a T/F to see if it worked. We don't care about this
    _, img = webcam.read()
    
    # Turns the original image into gray scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # MultiScale checks the image at different sizes
    # First you pass in the gray scaled image
    # Parameter 2 is the scale factor; how much you want the image shrunk. Larger value = faster, less accurate
    # Parameter 3 controls how strict detection is. A face must be detected at least 4 times nearby to count. Higher value = fewer false positives but might miss faces.
    faces = face_cascade.detectMultiScale(gray, 1.5, 1)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

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