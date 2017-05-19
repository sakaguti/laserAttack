#!/usr/bin/python
#coding: utf-8
from __future__ import division
import cv2 as cv
import numpy as np
import time
import sys
import urllib
import math
#from matplotlib import pyplot as plt
from PIL import Image
from itertools import product

def exec_cmd(cmd):
    from subprocess import Popen, PIPE
    p = Popen(cmd, stdout=PIPE)
    out = p.communicate()
    return out

def LeptonCapture():
#  return 640X480 gray image
# scale factors of Lepton
    scl = 5.5
    mx = 130
    my = 110
    img=''
    for temp in exec_cmd('/home/pi/src/LeptonModule/software/raspberrypi_capture/leptoncapture'):
        if temp is None:
                #print 'Lepton Error'
                return img
        temp = temp.split(' ')
        temp = temp[:-1]
        #print u'temp ',temp

        # type is list
        temp = map(lambda x: float(x), temp)
        pmax = np.max(temp)
        pmin = np.min(temp)
        temp = (temp-pmin)/(pmax-pmin)*255
        temp = map(np.uint8, temp)
        print u'temp',temp
        temp = np.array(temp)

        temp = temp.reshape(60, 80)
        #print u'temp ',temp
        print u'temp shape',temp.shape
        print u'temp ndim',temp.ndim
        print temp[0,0] 
	LeptonImg=temp
        print u'LeptonImg shape',LeptonImg.shape
        print u'LeptonImg dtype',LeptonImg.dtype

        # make scaled image
	dispSW = False

        if dispSW == True:
                cv.imshow('comp image',LeptonImg)  
                #cv.moveWindow('comp image',640,0)
                key = cv.waitKey(0)
                if key == ESC_KEY:
                        cv.destroyAllWindows()   
                        break
        #
	print LeptonImg[:0]
        LeptonImg = cv.resize(LeptonImg,None,fx=5.5, fy=5.5,interpolation=cv.INTER_CUBIC)

        '''
        # Equlize image
        LeptonImg = cv.equalizeHist(LeptonImg)
        # Gamma
        LeptonImg = gammaImage(0.01, LeptonImg)
        '''
        #print u'Lepton ',LeptonImg.shape

        lheight,lwidth  = LeptonImg.shape[:2]

        # make scaled image
        CanvasImg = np.zeros([480,640],np.uint8)
        height,width  = CanvasImg.shape[:2]

        gw = mx + lwidth
        gh = my + lheight
        if gw > width:
                gw = width
        if gh > height:
                gh = height

        '''
        print u'mx ',mx,' my ',my
        print u'gh ',gh,' gw ',gw
        print u'Canvas ',CanvasImg.shape
        print u'Lepton ',LeptonImg.shape
        print u'Lepton type',type(LeptonImg)
        '''

        CanvasImg[my:gh,mx:gw] = LeptonImg  

        #print u'Canvas type',type(CanvasImg)
    return CanvasImg


'''
data = np.array(data, dtype=np.uint8)
data = data.reshape(28,28)
print data.shape
image = Image.fromarray(data)
'''

LeptonCapture()

