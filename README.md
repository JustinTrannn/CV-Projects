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

## Eye Tracker
- So far the hardest one I've built. Easy first step of setting up the webcam and model
- Then I was place dots on my eyes so I could track the coordinates of those. This was also easy as I just had to look at the code for the mouse hand.
- The hardest setup was trying to figure out how to calibrate the eyes. I asked ChatGPT and it told me to use a checkmark type solution. This was difficult because it wasn't super intuitive.
- Then I stored the data so a linear regression model could predict where the eyes were looking
- I simply ploted the predictions of where the linear regression model was looking
- I think that this could use a lot of work. For one, I could add way more calibration points to look at and I could functionize making points and storing data so it's faster and easier to do. Also, I could not have my transition from calibration to prediction so split up. Right now, you have to press "n" in order to move onto the prediction eye tracking. The code will destroy the cv2 window and create a new one. I just did this because it was simple to do and think about and I didn't want to complicate it too much since this was already hard enough.

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
