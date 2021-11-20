from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import glob

w = 640
h = 480
camera = PiCamera()
camera.resolution = (w, h)
#camera.framerate = 30
camera.framerate = 30
camera.hflip = True
#camera.vflip = True
camera.vflip = False
rawCapture = PiRGBArray(camera, size=(w, h))

display_window = cv2.namedWindow("Images")

# face detection
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

time.sleep(1)

num = 1
img_list = glob.glob('./*.jpg')
num = len(img_list)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    # face detection
    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);
    #faces = face_cascade.detectMultiScale(gray, 1.1, 5);
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Images", image)
    # when you need to store image, please use following command
    #cv2.imwrite("image.jpg", image)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)

    if key == 27:
        camera.close()
        cv2.destroyAllWindows()
        break
    elif key == ord('c'):
        name = 'capture%02d.jpg' % num
        num = num + 1;
        cv2.imwrite(name, image)

cv2.destroyAllWindows()
