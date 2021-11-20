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
#camera.hflip = True
#camera.vflip = True
camera.hflip = False
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
    org = image.copy()
    
    # face detection
    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);
    #faces = face_cascade.detectMultiScale(gray, 1.1, 5);
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch,a_ch,b_ch = cv2.split(lab)
    # l_ch: intensity
    # a_ch: green - red/magenta
    # b_ch: blue - yellow
    #cv2.imshow("Lab", np.hstack([l_ch,a_ch,b_ch]))
    
    a_inv = cv2.bitwise_not(a_ch)
    b_inv = cv2.bitwise_not(b_ch)

    # green
    #cv2.imshow("ab", np.hstack([a_inv, b_ch]))
    mask1 = cv2.inRange(a_inv, 140, 255)
    mask2 = cv2.inRange(b_ch, 80, 176)
    #cv2.imshow("Mask", np.hstack([mask1, mask2]))
    org = image.copy()
    res1 = cv2.bitwise_and(org, org, mask=mask1)
    green = cv2.bitwise_and(res1, res1, mask=mask2)
    
    # yellow
    #cv2.imshow("ab", np.hstack([a_ch, b_ch]))
    mask1 = cv2.inRange(a_ch, 80, 176)
    mask2 = cv2.inRange(b_ch, 140, 255)
    org = image.copy()
    res1 = cv2.bitwise_and(org, org, mask=mask1)
    yellow = cv2.bitwise_and(res1, res1, mask=mask2)
    
    # orange
    #cv2.imshow("ab", np.hstack([a_ch, b_ch]))
    mask1 = cv2.inRange(a_ch, 155, 195)
    mask2 = cv2.inRange(b_ch, 155, 225)
    #cv2.imshow("Mask", np.hstack([mask1, mask2]))
    org = image.copy()
    res1 = cv2.bitwise_and(org, org, mask=mask1)
    orange = cv2.bitwise_and(res1, res1, mask=mask2)
    
    # red
    #cv2.imshow("ab", np.hstack([a_ch, b_ch]))
    mask1 = cv2.inRange(a_ch, 195, 255)
    mask2 = cv2.inRange(b_ch, 100, 156)
    #cv2.imshow("Mask", np.hstack([mask1, mask2]))
    org = image.copy()
    res1 = cv2.bitwise_and(org, org, mask=mask1)
    red = cv2.bitwise_and(res1, res1, mask=mask2)

    # purple
    #cv2.imshow("ab", np.hstack([a_ch, b_inv]))
    mask1 = cv2.inRange(a_ch, 170, 190)
    mask2 = cv2.inRange(b_inv, 155, 190)
    #cv2.imshow("Mask", np.hstack([mask1, mask2]))
    org = image.copy()
    res1 = cv2.bitwise_and(org, org, mask=mask1)
    purple = cv2.bitwise_and(res1, res1, mask=mask2)

    # blue
    #cv2.imshow("ab", np.hstack([a_ch, b_inv]))
    mask1 = cv2.inRange(a_ch, 80, 170)
    mask2 = cv2.inRange(b_inv, 180, 255)
    #cv2.imshow("Mask", np.hstack([mask1, mask2]))
    org = image.copy()
    res1 = cv2.bitwise_and(org, org, mask=mask1)
    blue = cv2.bitwise_and(res1, res1, mask=mask2)
    cv2.imshow("Image", output)
    cv2.imshow("GYO", np.hstack([green, yellow, orange]))
    cv2.imshow("RPB", np.hstack([red, purple, blue]))
    
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
