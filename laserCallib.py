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

# Import the PCA9685 module.
import Adafruit_PCA9685

# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

# Configure min and max servo pulse lengths
servoR_min = 150  # Min pulse length out of 4096
servoR_max = 600  # Max pulse length out of 4096

# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)

# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)

# 定数定義
ESC_KEY = 27     # Escキー
INTERVAL= 33     # インターバル
#INTERVAL= 0     # インターバル
FRAME_RATE = 30  # fps


WINDOW_ORG = "org"
WINDOW_BACK = "back"
WINDOW_DIFF = "diff"

ON = True
OFF = False
nFrame = 0
import signal
import RPi.GPIO as GPIO

def receive_signal(signum, stack):
    print 'Received:', signum
    laser.ChangeDutyCycle(0)
    laserServoCtl(0,staticVar.LaserSERVO_0_ORG)
    laserServoCtl(1,staticVar.LaserSERVO_1_ORG)
    sensorServoCtl(staticVar.SensorSERVO_ORG)
    laser.stop(0.0);
    sys.exit()

signal.signal(signal.SIGHUP, receive_signal)
signal.signal(signal.SIGINT, receive_signal)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
gp_out=27
GPIO.setup(gp_out, GPIO.OUT)
laser = GPIO.PWM(gp_out,100)

laser.start(0.0);

def laserCtl(pwm):
    staticVar.LaserPWM=pwm
    laser.ChangeDutyCycle(pwm)

def servoCtl(no, x):
    #print u'servoCtl no=',no
    #print u'servoCtl x=',x
    pwm.set_pwm(no, 0, int(x))

def sensorServoCtl(xz):
    servoCtl(1,int(xz))

def laserServoCtl(no,xz):
    servoCtl(2+no*2,int(xz[1]))
    servoCtl(3+no*2,int(xz[0]))
    staticVar.laserServoPos=xz

class staticVar:
#　レーザーサーボの原点位置
    LaserSERVO_0_ORG= np.array([365,300])
    LaserSERVO_1_ORG= np.array([350,370])
    #LaserSERVO_0_ORG= np.array([376,300])
    #LaserSERVO_1_ORG= np.array([370,350])
#  レーザーの画面位置
    LaserPos= np.zeros(2)
#  レーザーのPWM
    LaserPWM=0
#  カメラサーボの原点位置
    SensorSERVO_ORG=335
#  サーボ変数
    laserServoPos = np.array([370,300]) 
    firstLaserPos = np.array([0.0,0.0]) 
    firstError = np.array([0.0,0.0]) 
    finalLaserPos = np.array([0.0,0.0]) 
    finalError = np.array([0.0,0.0]) 
    oldErrVec = np.array([0.0,0.0]) 
    oldErrEst = np.array([0.0,0.0])
    oldErr = 0.0 
    Coeff = np.array([[0.075,0.075],[0.075,0.075]])
    sAngle = np.array([0.0,0.0])
#  近似式
    funcXSX = np.zeros((2,2,2))
    funcXSY = np.zeros((2,2,2))
    funcYSX = np.zeros((2,2,2))
    funcYSY = np.zeros((2,2,2))

def getLaserPoint():
    #time.sleep(0.5)
    laserPWM=staticVar.LaserPWM
    laserCtl(0) # PWM 0%
    # read backgroud image
    back_frame = MjpgStreamerCapture()
	
    #back_frame = cv.resize(back_frame,None,fx=2, fy=2, \
    #                                    interpolation=cv.INTER_CUBIC)
    #cv.imwrite('BackGroudImage.jpg', back_frame)

    laserCtl(20) # PWM 30%
    #time.sleep(0.1)
    f_frame = MjpgStreamerCapture()
    #laserCtl(0) # PWM 0%

    #f_frame = cv.resize(f_frame, None, fx=2, fy=2, \
    #                                    interpolation=cv.INTER_CUBIC)
    #diff_frame = cv.absdiff(f_frame, back_frame)
    diff_frame = cv.subtract(f_frame, back_frame)

    #  Color to GRAY
    diff_frame = cv.cvtColor(diff_frame,cv.COLOR_RGB2GRAY)

    #  max position
    min, max, min_loc, max_loc = cv.minMaxLoc(diff_frame)

    #print u'max =',max
    #print u'min =',min
    #print u'laser position  =',max_loc

    center = np.array(f_frame.shape[:2])/2 # [h,w]
    laser  = np.array([max_loc[1],max_loc[0]]) # [y,x]

    # laser is Gree circle and Center is Red circle
    # BGR   Red is center of image
    cv.circle(diff_frame, (int(center[1]),int(center[0])),5,  (0, 0, 255), 1)
    cv.circle(diff_frame, (int(laser[1]),int(laser[0])),  10,  (0, 255, 0), 5)
    #print dispSW
    
    '''
    if dispSW == True:
	    #cv.imshow('comp image',f_frame)  
	    #cv.moveWindow('comp image',640,0)
            cv.imshow('diff image',diff_frame)
	    #cv.moveWindow('diff', 0,480)
            key = cv.waitKey(0)
            if key == ESC_KEY:
                    cv.destroyAllWindows()
                    return
            elif key == ord('s'):
                    fileName = 'DiffImage'+str(iter)+'.jpg'
                    cv.imwrite(fileName, f_frame)
    '''
    laserCtl(staticVar.LaserPWM)
    return laser, center


def laserCaliblationAxis(checkXZ, no):

    # checkXZ = 0/1  0 then Z axis, 1 then X axis rotate
    filename = 'laserServo'+str(checkXZ)+'.csv'
    f = open(filename, 'w')

    # data for interpolate
    xData = np.zeros(31)
    yData = np.zeros(31)
    sxData = np.zeros(31)
    syData = np.zeros(31)

    # rotae to ORIGIN position
    if no == 0:
        staticVar.laserServoPos = staticVar.LaserSERVO_0_ORG
        laserServoCtl(no,staticVar.LaserSERVO_0_ORG)
    elif no == 1:
        staticVar.laserServoPos = staticVar.LaserSERVO_1_ORG
        laserServoCtl(no,staticVar.LaserSERVO_1_ORG)

    if checkXZ == 0:
    	staticVar.laserServoPos[0] -=  50
    else:
        staticVar.laserServoPos[1] -=  30
    	staticVar.laserServoPos[0] +=  50

    laserServoCtl(no,staticVar.laserServoPos)
    sensorServoCtl(staticVar.SensorSERVO_ORG)

    iter=0
    sStep = 3
    reTry = 0
    oldxy = [0,0]
    while True:
        #print u'Iterete = ',iter
	laser, center=getLaserPoint()

	# data analysis
	msg = '%d,%d,%d,%d,%d' % (iter,laser[1],laser[0],\
			staticVar.laserServoPos[1],staticVar.laserServoPos[0])

   	# レーザー点の位置がおかしければやり直す
        if iter > 1 and (abs(laser[1]-oldxy[0]) > 25 or abs(laser[0]-oldxy[1]) > 25):
		print u'RETRY:',msg
		#print abs(xData[0]-oldxy[0])
		#print abs(yData[1]-oldxy[1])
		if reTry < 10:
			reTry += 1
			continue

	oldxy=[laser[1],laser[0]]

	xData[iter]=laser[1]
	yData[iter]=laser[0]
        sxData[iter]=staticVar.laserServoPos[1]
        syData[iter]=staticVar.laserServoPos[0]

        print msg
        f.write(msg+'\n')
        f.flush()

        # X axis
        staticVar.sAngle = [sStep, 0]
    	if checkXZ == 0:
        	staticVar.laserServoPos = staticVar.laserServoPos + staticVar.sAngle
        # Z axis
        staticVar.sAngle = [0, sStep]
    	if checkXZ == 1:
        	staticVar.laserServoPos = staticVar.laserServoPos + staticVar.sAngle

        laserServoCtl(no, staticVar.laserServoPos)
        iter += 1
        if iter > 30:
                break
    f.close()
    if dispSW == True:
        cv.destroyAllWindows()

    staticVar.funcXSX[no][checkXZ] = np.polyfit(xData, sxData, 1)
    staticVar.funcXSY[no][checkXZ] = np.polyfit(xData, syData, 1)
    staticVar.funcYSX[no][checkXZ] = np.polyfit(yData, sxData, 1)
    staticVar.funcYSY[no][checkXZ] = np.polyfit(yData, syData, 1)

    filename = 'laserServo'+str(checkXZ)+'Relations.txt'
    f = open(filename, 'w')
    if staticVar.funcXSX[no][checkXZ] is not None:
    	f.write('funcXSX axis '+str(checkXZ)+'\n')
    	f.write(str(staticVar.funcXSX[no][checkXZ])+'\n')
 
    if staticVar.funcXSY[no][checkXZ] is not None:
    	f.write('funcXSY axis '+str(checkXZ)+'\n')
    	f.write(str(staticVar.funcXSY[no][checkXZ])+'\n')

    if staticVar.funcYSX[no][checkXZ] is not None:
    	f.write('funcYSX axis '+str(checkXZ)+'\n')
    	f.write(str(staticVar.funcYSX[no][checkXZ])+'\n')

    if staticVar.funcYSY[no][checkXZ] is not None:
    	f.write('funcYSY axis '+str(checkXZ)+'\n')
    	f.write(str(staticVar.funcYSY[no][checkXZ])+'\n')
    f.close()
    staticVar.laserServoPos[1] +=  30

def funcXSX(n,x):
	func = np.poly1d(staticVar.funcXSY[n][0])
	return func(x)

def funcXSY(n,x):
	func = np.poly1d(staticVar.funcYSX[n][0])
	return func(x)

def funcYSX(n,x):
	func = np.poly1d(staticVar.funcXSY[n][1])
	return func(x)

def funcYSY(n,x):
	func = np.poly1d(staticVar.funcYSX[n][1])
	return func(x)

# ralserNo, servoPos, targetPos
def pixelToAngle(no,tPos):
    cPos=staticVar.laserServoPos

    #print u'laser pos ',cPos
    #print u'target pos ',tPos

    # tPos -> sPos
    tSy=funcYSY(no,tPos[1])
    tx =funcYSX(no,tPos[1])
    tSx=funcXSX(no,tPos[0])
    ty =funcXSY(no,tPos[0])
    dx =cPos[0]-funcXSX(no,tPos[0])
    dy =cPos[1]-funcYSY(no,tPos[1])

    #print u'tSy=',tSy,' tSx=', tSx

    # Add Error correction
    print u'dx=',dx,' dy=',dy 
    dy = 0
    dx = 0

    nSx=tSx-dx
    nSy=tSy-dy
    staticVar.sAngle=[nSx,nSy] 
    return [nSx,nSy] 

    # Add Error correction

def laserAttack(cx,cy,rx,ry,rw,rh):
    print'Laser Attack!!'
    laserCtl(20)

    if testSW == True:
    	#laser,center=getLaserPoint()
    	print u'Resion x y ',rx,ry,' w h ',rw,rh
	for y in range(ry,ry+rh,10):
		for x in range(rx,rx+rw,10):
			targetPos = [float(x),float(y)]
    			angle=pixelToAngle(0,targetPos)
    			print u'Target ',targetPos,' Angle ',angle
    			laserServoCtl(0,angle)
			# get PiCam image
			PiCamImg = MjpgStreamerCapture()
    			#PiCamImg = cv.resize(PiCamImg,None,fx=2, fy=2, \
                        #                interpolation=cv.INTER_CUBIC)
    			# laser is Gree circle and Center is Red circle
    			# BGR   Red is center of image
    			cv.circle(PiCamImg, (int(x),int(y)),\
							5,  (0, 0, 255), 1)
			cv.rectangle(PiCamImg,(rx,ry),\
						((rx+rw),(ry+rh)),(0,255,0),1)
    			#print dispSW
    			if dispSW == True:
        			cv.imshow('attack image',PiCamImg)
				cv.moveWindow('attack image',0,0)
            			key = cv.waitKey(INTERVAL)
            			if key == ESC_KEY:
                    			cv.destroyAllWindows()
					sys.exit()
                    			break
			time.sleep(0.5)
    else:
	center=[cx,cy]
    	print u'Target Canter',center
    	angle=pixelToAngle(0,[cx,cy])
    	print u'Angle ',angle
    	laserServoCtl(0,angle)
	# get PiCam image
	PiCamImg = MjpgStreamerCapture()
    	#PiCamImg = cv.resize(PiCamImg,None,fx=2, fy=2, \
        #                               interpolation=cv.INTER_CUBIC)
    	# laser is Gree circle and Center is Red circle
    	# BGR   Red is center of image
    	cv.circle(PiCamImg, (cx,cy),\
				5,  (0, 0, 255), 1)
	cv.rectangle(PiCamImg,(cx-50,cy-50),\
				((cx+50),(cy+50)),(0,255,0),2)
    	#print dispSW
    	if dispSW == True:
        	cv.imshow('attack image',PiCamImg)
		cv.moveWindow('attack image',0,0)
            	key = cv.waitKey(INTERVAL)
            	if key == ESC_KEY:
               			cv.destroyAllWindows()
				sys.exit()

def selectBlob(label):
    n = label[0] - 1
    if n <= 2:
	return -1 

    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    maxArea = 0
    index = selectedBlobIndex=2

    for blob in data[2:4]:
        print u'blob area = ',blob[4]
	if maxArea < blob[4]:
		maxArea = blob[4]
    		selectedBlobIndex=index
	index += 1
	
    print u'selectedBlobIndex = ',selectedBlobIndex
    return selectedBlobIndex
	
def exec_cmd(cmd):
    from subprocess import Popen, PIPE
    p = Popen(cmd, stdout=PIPE)
    out = p.communicate()
    return out

def MjpgStreamerCapture():
    img=''
    stream=urllib.urlopen('http://sensor.local:8080/?action=snapshot')
    bytes=stream.read(320*240)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        encode_param=int(cv.IMWRITE_JPEG_QUALITY)
        c = np.fromstring(jpg, dtype=np.uint8)
        img = cv.imdecode(c, encode_param)
    	img = cv.resize(img,None,fx=2, fy=2, \
                                        interpolation=cv.INTER_CUBIC)
    return img

def FtoDeg(F):
	t = int((float(F-30))*1.8) 
	if t < 0:
		t = 0
	return t 

def LeptonCapture():
    img=''
    for png in exec_cmd('/home/pi/src/flirpi-mjpeg/leptgraypng'):
    	if png is None:
		#print 'Lepton Error'
		return img

    	png = np.fromstring(png, dtype=np.uint8)

	'''
        print u'png',png
        print u'max',np.max(png)
        print u'min',np.min(png)
        print u'pngType',type(png)
        print u'pngSize',png.ndim

	temp = map(FtoDeg, png) # from F to ℃
        print u'tempDeg type',type(temp)
	pmax = np.max(temp)
	pmin = np.min(temp)
	temp = (png-pmin)/(pmax-pmin)*255
	print u'temp ',temp
        print u'tempType',type(temp)
    	img = cv.imdecode(temp, cv.IMREAD_GRAYSCALE)
	# ここが動かない
	if dispSW == True:
		cv.imshow('comp image',img)  
		cv.moveWindow('comp image',640,0)
    		key = cv.waitKey(0)
    		if key == ESC_KEY:
    			cv.destroyAllWindows()   
			break
	#
	'''

    	img = cv.imdecode(png, cv.IMREAD_GRAYSCALE)
    return img

def gammaImage(gamma, img):
    look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
    for i in range(256):
       	look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    return cv.LUT(img, look_up_table)

def readCalibFile():
    for fileno in range(2):
        filename='laserServo'+str(fileno)+'Relations.txt'
        f = open(filename)
        txt=f.read().split('\n')
        i = 0
        for line in txt:
                if line.find('axis') > 0:
                        i += 1
                        continue

                data = line.replace('[','').replace(']','').split(' +')
                data = data[0].split(' ')
                data = filter(lambda s:s != '', data)
                if len(data) == 0:
                        i += 1
                        continue
                #print u'DATA',data[0],',',data[1]

                data[0]=float(data[0])
                data[1]=float(data[1])
		#print u'readCalibFile= ',data
    		if i == 1:
			staticVar.funcXSX[0][fileno]=[data[0],data[1]]
    		if i == 3:
    			staticVar.funcXSY[0][fileno]=[data[0],data[1]]
    		if i == 5:
    			staticVar.funcYSX[0][fileno]=[data[0],data[1]]
    		if i == 7:
    			staticVar.funcYSY[0][fileno]=[data[0],data[1]]
		i = i+1
    	f.close()

laserServoCtl(1,staticVar.LaserSERVO_0_ORG) # 3mWレーザーを元に戻す

scl = 5.5
mx = 130
my = 110
dispSW = True
verbosSW = False
calSW = True
testSW = False

argv=sys.argv
argc = len(argv)
if argc > 1:
	for av in argv:
		if av == '-nd':
			dispSW = False
		elif av == '-v':
			verbosSW = True
		elif av == '-nv':
			verbosSW = False
		elif av == '-d':
			dispSW = True
		if av == '-nc':
			calSW = False
		if av == '-c':
			calSW = True
		if av == '-t':
			testSW = True
		if av == '-nt':
			testSW = False

#レーザーOFF
laserCtl(0)

laserServoCtl(1,[500,staticVar.LaserSERVO_0_ORG[1]]) # 3mWレーザーを横に向ける

if calSW == True:
    laserCaliblationAxis(0, 0)
    laserCaliblationAxis(1, 0)
else:
# read caliblation file
    readCalibFile()


'''
#
# calibrate 3mW laser
laserServoCtl(0,[500,staticVar.LaserSERVO_1_ORG[1]]) # 30mWレーザーを横に向ける
camImg = MjpgStreamerCapture()
#laserCaliblation(0, camImg)
laserServoCtl(0,staticVar.LaserSERVO_1_ORG) # 30mWレーザーを元に戻す
'''

#レーザーOFF
laserCtl(0)

# 正面に向ける
laserServoCtl(0,staticVar.LaserSERVO_0_ORG)
laserServoCtl(1,staticVar.LaserSERVO_1_ORG)
sensorServoCtl(staticVar.SensorSERVO_ORG)

while True:
	cx = cy = 0
	rrx = rry = rw = rh = 0

	# get Lepton image
	LeptonImg = LeptonCapture()

	# get PiCam image
	PiCamImg = MjpgStreamerCapture()

	if PiCamImg is None or LeptonImg is None:
		continue

	# max 255 min 0 (0,159), (0,8)
	min, max, min_loc, max_loc = cv.minMaxLoc(LeptonImg)

        '''
	if verbosSW == True:
		print 'max='+str(max)+' min='+str(min)
		print 'max_loc='+str(max_loc)+' min_loc='+str(min_loc)
        '''

	# avoid noise
	if testSW == False and (max > 250 or max <= 31):
		'''
		if dispSW == True:
			cv.imshow('comp image',PiCamImg)  
			cv.moveWindow('comp image',640,0)
    			key = cv.waitKey(INTERVAL)
    			if key == ESC_KEY:
    				cv.destroyAllWindows()   
				break
		'''
		continue

	if testSW == False:
		# Equlize image
		LeptonImg = cv.equalizeHist(LeptonImg)
		# Gamma
        	LeptonImg = gammaImage(0.01, LeptonImg)
		# make scaled image
		LeptonImg = cv.resize(LeptonImg,None,fx=scl, fy=scl, \
					interpolation=cv.INTER_CUBIC)

	# calc max and min
		min, max, min_loc, max_loc = cv.minMaxLoc(LeptonImg)
		if verbosSW == True:
			print 'max='+str(max)+' min='+str(min)
			print 'max_loc='+str(max_loc)+' min_loc='+str(min_loc)

		# binary image
		#ret,thresh = cv.threshold(LeptonImg,31,255,0)
		thresh = cv.adaptiveThreshold(LeptonImg,255, \
			cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

		if thresh is None:
			continue

		# calc blob
    		label = cv.connectedComponentsWithStats(thresh)
    		n = label[0] - 1
		if n <= 2:
			continue
    		data = np.delete(label[2], 0, 0)
    		center = np.delete(label[3], 0, 0)
		'''
    		print u"ブロブの個数:", n
    		print u"各ブロブの外接矩形の左上x座標", data[:,0]
    		print u"各ブロブの外接矩形の左上y座標", data[:,1]
    		print u"各ブロブの外接矩形の幅", data[:,2]
    		print u"各ブロブの外接矩形の高さ", data[:,3]
    		print u"各ブロブの面積", data[:,4]
		print u"各ブロブの中心座標:\n",center
		'''

		imax = selectBlob(label) 
		if verbosSW == True:
			print u"max =",imax

    		data = np.delete(label[2], 0, 0)
    		print u'data=',data

		i = imax
    		print u"各ブロブの外接矩形の左上x座標", data[i,0]
    		print u"各ブロブの外接矩形の左上y座標", data[i,1]
    		print u"各ブロブの外接矩形の幅", data[i,2]
    		print u"各ブロブの外接矩形の高さ", data[i,3]
    		print u"各ブロブの面積", data[i,4]
		print u"各ブロブの中心座標:\n",center[i]
		rx=data[i,0] # 外接矩形の最小x
		ry=data[i,1] # 外接矩形の最小y
		rw=data[i,2] # 外接矩形の幅
		rh=data[i,3] # 外接矩形の高さ
		cx=int(center[i][0]) # 外接矩形の中心 x
		cy=int(center[i][1]) # 外接矩形の中心 y

        else:
	#	testSW ON
		laserCtl(50)
		# RECT  (240, 320, 3)
		print u'RECT ',PiCamImg.shape
		center = np.zeros((2,2))
		center[0][0]=PiCamImg.shape[1]/2 # 外接矩形の中心 y
		center[0][1]=PiCamImg.shape[0]/2 # 外接矩形の中心 x
		cx=int(center[0][0]) # 外接矩形の中心 y
		cy=int(center[0][1]) # 外接矩形の中心 y

		LeptonImg = cv.resize(LeptonImg,None,fx=scl, fy=scl, \
					interpolation=cv.INTER_CUBIC)
		rx=cx-25 # 外接矩形の最小x
		ry=cy-25 # 外接矩形の最小y
		rw=50 # 外接矩形の幅
		rh=50 # 外接矩形の高さ

	# Gray to Color
	# convert gray image to color image
	LeptonRGB = cv.cvtColor(LeptonImg,cv.COLOR_GRAY2RGB)

	# make same size image
	height,width  = LeptonRGB.shape[:2]
	#print u"Lepton width ",width
	#print u"Lepton height ",height

	CanvasImg = np.zeros(PiCamImg.shape,np.uint8)
	lheight,lwidth  = PiCamImg.shape[:2]

	# crop image
	sh=height-lheight
	sw=width-lwidth
	if sh < 0:
		sh = 0
	if sw < 0:
		sw = 0

	LeptonRGB=LeptonRGB[sh:height-sh,sw:width-sw]
	cx = cx-sw
	cy = cy-sh
	rx = rx-sw
	ry = ry-sh

	gw = mx + width
	gh = my + height
	if gw > lwidth:
		gw = lwidth
	if gh > lheight:
		gh = lheight

	#cv.imshow('Lep image',LeptonImg)  
	#key = cv.waitKey(INTERVAL0)

	CanvasImg[my:gh,mx:gw] = LeptonRGB  
	cx = cx+mx
	cy = cy+my
	rx = rx+gw
	ry = ry+gh

	# make composite image
	CompImg = cv.addWeighted(PiCamImg, 1.0, CanvasImg, 0.5, 0.8)

	# add maker
	if testSW == False:
		cv.circle(CompImg, (cx, cy), 2,  (0, 0, 255), 5)
		cv.rectangle(CompImg,(cx-int(rw/2),cy-int(rh/2)),\
				(cx+int(rw/2),cy+int(rh/2)),(0,255,255),1)
	else:
	# testSW ON
		cv.circle(CompImg, (cx, cy), 10,  (0, 0, 255), 1)
		cv.rectangle(CompImg,(rx,ry),\
				(rx+rw,ry+rh),(0,255,0),1)

	if verbosSW == True:
		print u'center ',cx,' ',cy
		print u'rect min',rx,' ',ry
		print u'rect max',rx+rw,' ',ry+rh

         
        # 背景フレーム
	if nFrame == 0:
		back_frame = np.zeros_like(CanvasImg, np.float32)
	nFrame += 1
   	# 差分計算
	f_frame = CanvasImg.astype(np.float32)
    	diff_frame = cv.absdiff(f_frame, back_frame)

    	# 背景の更新
    	cv.accumulateWeighted(CanvasImg, back_frame, 0.025)
	# フレーム表示
    	#cv.imshow(WINDOW_BACK, back_frame.astype(np.uint8))
    	if dispSW == True:
    		cv.imshow(WINDOW_DIFF, diff_frame.astype(np.uint8))
		cv.moveWindow('comp image',640,480)
		cv.imshow('comp image',CompImg)  
		cv.moveWindow('comp image',640,0)

    	# Escキーで終了
    	key = cv.waitKey(INTERVAL)
    	if key == ESC_KEY:
    		cv.destroyAllWindows()   
		break
        

	# レーザーで攻撃する
	laserAttack(cx,cy,rx,ry,rw,rh)

	if dispSW == True:
		cv.imshow('comp image',CompImg)  
		cv.moveWindow('comp image',640,0)
    		key = cv.waitKey(INTERVAL)
    		if key == ESC_KEY:
    			cv.destroyAllWindows()   
			break
		elif key == ord('s'):
    			cv.imwrite('saveimage.jpg', CompImg)
    			cv.destroyAllWindows()
			break
