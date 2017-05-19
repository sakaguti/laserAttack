#!/usr/bin/python
#
# read image from mjpg-streamer
#
#
import cv2
import urllib 
import numpy as np

# stream is too slow, do not use stream
#stream=urllib.urlopen('http://sensor.local:8080/?action=stream')

bytes=''
while True:
    stream=urllib.urlopen('http://sensor.local:8080/?action=snapshot')
    bytes=stream.read(320*240)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
	encode_param=int(cv2.IMWRITE_JPEG_QUALITY)
        c = np.fromstring(jpg, dtype=np.uint8)
        encimg = cv2.imdecode(c, encode_param)

        cv2.imshow('i',encimg)

        if cv2.waitKey(1) ==27:
            exit(0)    

