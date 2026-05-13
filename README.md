# Computer Vision Projects


## Facial Recognition
### Learning about facial recognition with OpenCV through this tutorial:
https://www.youtube.com/watch?v=AsHc3WiioOk&list=PLKikgSZ8GZYd7EXmKZSLwITvoCm0KFI5B

## Brightness Control
### Followed YouTube tutorial for brightness control: 
https://www.youtube.com/watch?v=8SoYONhlowY

## Face Blur
### Attempted to complete without following a tutorial
- I found it easy to setup the webcam without help because I've done it a few times now.
- I had trouble blurring the image because I did not have the correct source on line 31 with the cv2.GuassianBlur().
- Managed to fix the errors after thinking it through logically.
- Learned that ksize is how blurry you want it to be. It must be an odd number though. Don't understand why.
- Still unsure about what sigmaX does. I understand that it's STD but didn't see any changes in the code when I was messing around with it.

## Mouse Click
- Easy to set up the webcam.
- Now have a better understanding of how to setup a mediapipe model.
- The easiest part of working with pyautogui (controls the computer) was clicking the mouse because it's only one function: pyautogui.click()
- The hardest part of this project was getting the mouse to move along with my finger. At first, I had to mouse_x and mouse_y completely wrong. In the end, it's very simple because you just get the coordinates of the fingers on the webcam, and then correlate those to the coordinates of the laptop screen. Then you use the pyautogui.moveTo() function and it's very straight forward. Took a little bit of asking ChatGPT why my code wasn't working to understand.



## Ideas
### Posture detector
- shoulder alignment
- neck/head tilt
- slouch detection
- timer for bad posture
- alert after 10 seconds
- simple dashboard showing posture score

### Attendance system with face recognition
- a folder of known faces
- encode or compare faces
- mark attendance in a CSV file
- show name labels above faces
- prevent duplicate attendance entries
- add timestamp

### Eye-controlled mouse
- MediaPipe Face Mesh detects your face and eye landmarks
- the program finds approximate iris/pupil position
- you calibrate by looking at 5 or 9 screen points
- your cursor moves based on estimated gaze
- blinking or holding gaze triggers a click
