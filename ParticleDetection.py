import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
from pylibfreenect2 import OpenGLPacketPipeline

pipeline = OpenGLPacketPipeline()

logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

# select image to use
types = FrameType.Color | FrameType.Depth  
listener = SyncMultiFrameListener(types)
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# use preset calibration parameters
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

while True:
    frames = listener.waitForNewFrame()
    color = frames["color"]
    ir = frames["ir"]
    depth = frames["depth"]
    registration.apply(color, depth, undistorted, registered)

    ### image processing ###
    img = cv2.resize(color.asarray(), (int(1920 / 3), int(1080 / 3)))
    img = cv2.medianBlur(img, 5)
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    depth = depth.asarray()
    depth = cv2.medianBlur(depth, 5)

    mask = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    #mask = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    #kernel = np.ones((1,1), np.uint8)
    #mask = cv2.erode(mask, kernel, iterations = 1)

    fgbg = cv2.createBackgroundSubtractorMOG2()
    mog = fgbg.apply(mask)
    retval, mask = cv2.threshold(mog, 127, 255, cv2.THRESH_BINARY)

    # attempt at circle detection
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 120, param1=50, param2=50, minRadius=30, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(mask,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(mask,(i[0],i[1]),2,(0,0,255),3)

    #cv2.imshow("original", img)
    #cv2.imshow('depth', depth / 4500.)
    cv2.imshow("grayscale", grayscale)
    cv2.imshow('mask', mask)
    #cv2.imshow('MOG', mog)

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)
