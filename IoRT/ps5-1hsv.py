from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import glob
import numpy as np

w = 640
h = 480
camera = PiCamera()
camera.resolution = (w, h)
#camera.framerate = 30
camera.framerate = 60
camera.hflip = True
#camera.vflip = True
camera.vflip = False
rawCapture = PiRGBArray(camera, size=(w, h))

#display_window = cv2.namedWindow("Images")

# face detection
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

time.sleep(1)

num = 1
img_list = glob.glob('./*.jpg')
num = len(img_list)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    output = image.copy()

    # face detection
    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);
    #faces = face_cascade.detectMultiScale(gray, 1.1, 5);
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # green (1/3: 60)
    lower_hue = np.array([40, 30, 30])
    upper_hue = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    input = image.copy()
    green = cv2.bitwise_and(input, input, mask=mask)

    # yellow (1/6: 30)
    lower_hue = np.array([10, 30, 30])
    upper_hue = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    input = image.copy()
    yellow = cv2.bitwise_and(input, input, mask=mask)

    # orange (1/12: 15)
    hue1 = np.array([0, 30, 30])
    hue2 = np.array([10, 255, 255])
    hue3 = np.array([175, 30, 30])
    hue4 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, hue1, hue2)
    mask2 = cv2.inRange(hsv, hue3, hue4)
    mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0)
    input = image.copy()
    orange = cv2.bitwise_and(input, input, mask=mask)
    
    # red (0: 0)
    lower_hue = np.array([140, 30, 30])
    upper_hue = np.array([175, 255, 255])
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    input = image.copy()
    red = cv2.bitwise_and(input, input, mask=mask)

    # purple (5/6: 150)
    lower_hue = np.array([120, 30, 30])
    upper_hue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    input = image.copy()
    purple = cv2.bitwise_and(input, input, mask=mask)
    
    # blue (2/3: 120)
    lower_hue = np.array([100, 30, 30])
    upper_hue = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    input = image.copy()
    blue = cv2.bitwise_and(input, input, mask=mask)

    cv2.imshow("Image", output)
    cv2.imshow("GYO", np.hstack([green, yellow, orange]))
    cv2.imshow("RPB", np.hstack([red, purple, blue]))
    
    #gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    #p2 = 30
    #lo = -1
    #success = 0
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
    #if circles is not None:
    #    a, b, c = circles.shape
    #    for i in range(b):
    #        cv2.circle(output, (circles[0][i][0], circles[0][i][1]), circles[0][i][2],
    #                   (0, 255, 0), 3, cv2.LINE_AA)
    #    cv2.imshow("Output", np.hstack([image, output]))

    #cv2.imshow("Gray", gray)
    # when you need to store image, please use following command
    #cv2.imwrite("image.jpg", image)
    key = cv2.waitKey(1)
    rawCapture.truncate(0)

    if key == 27:
        break
    elif key == ord('c'):
        name = 'capture%02d.jpg' % num
        num = num + 1;
        cv2.imwrite(name, image)


camera.close()
cv2.destroyAllWindows()
