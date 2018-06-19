#!/usr/bin/env python2
import numpy
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
from pylibfreenect2 import OpenGLPacketPipeline

import ImageProc

pipeline = OpenGLPacketPipeline()

#logger = createConsoleLogger(LoggerLevel.Debug)
#setGlobalLogger(logger)

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
registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

maxDepth = []

#Discard 4 frames to clear any initialisation problems
for i in range(4):
    frames = listener.waitForNewFrame()
    listener.release(frames)    

while True:
    frames = listener.waitForNewFrame()
    color = frames["color"]
    ir = frames["ir"]
    depth = frames["depth"]
    registration.apply(color, depth, undistorted, registered)
    depth = depth.asarray()
    if len(maxDepth) == 0:
        maxDepth = depth
        listener.release(frames)
        continue

    maxDepth = numpy.maximum(depth, maxDepth)

    foreground = registered.asarray(numpy.uint8).copy()

    ImageProc.depthFilter(maxDepth, depth, foreground)
    
    cv2.imshow("registered", registered.asarray(numpy.uint8))
    #cv2.imshow("depth", depth / 4500)
    #cv2.imshow("max depth", maxDepth / 4500)
    #cv2.imshow("color", color.asarray())

    cv2.imshow("greenscreen", foreground)
        
    listener.release(frames)
    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

