import numpy as np
import time
import cv2
import cv2.aruco as aruco
from picamera.array import PiRGBArray
from picamera import PiCamera

iw = 640
ih = 480
# pi camera
camera = PiCamera()
camera.framerate = 32
camera.resolution = (iw, ih)
#camera.hflip = True
#camera.vflip = True
camera.hflip = False
camera.vflip = False
rawCapture = PiRGBArray(camera, size=(iw, ih))

time.sleep(1)

num = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    #print(frame.shape) #480x640
    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()

    #print(parameters)

    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
        #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print(corners)

    #It's working.
    # my problem was that the cellphone put black all around it. The alrogithm
    # depends very much upon finding rectangular black blobs

    gray = aruco.drawDetectedMarkers(gray, corners)

    #print(rejectedImgPoints)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    rawCapture.truncate(0)

# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()
