#!/usr/bin/python
#coding: utf-8
import cv2 as cv
import numpy as np
import time
import sys
import urllib

# 定数定義
ESC_KEY = 27     # Escキー
INTERVAL= 33     # インターバル
FRAME_RATE = 30  # fps

WINDOW_ORG = "org"
WINDOW_BACK = "back"
WINDOW_DIFF = "diff"
ON = True
OFF = False
Coeff = 0.6
nFrame = 0

def laserCtl(pwm):

    return

def servoCtl(no, angle):

    return

def pixelToAngle(no,sPos,lPos,tPos):
#  image pixel to servo angle
    errVec= tPos-lPos
    err =  np.linalg.norm(errVec) 

    sAngle = Coeff * errVec 

    oldErrorVec = errVec
    return sAngle


def laserCaliblation(no, camImg):

    laserCtl(0)
    time.sleep(0.5)

    # rotae to ORIGIN position
    if no == 0:
        servoPos = SERVO_0_ORG
    elif no == 1:
        servoPos = SERVO_1_ORG
    servoCtl(no,servoPos)

    # read backgroud image
    back_frame = MjpgStreamerCapture()
    back_frame = cv.resize(back_frame,None,fx=2, fy=2, \
					interpolation=cv.INTER_CUBIC)
    iter=0
    while True:
    	laserCtl(60)
    	time.sleep(0.5)

    	f_frame=MjpgStreamerCapture()
    	f_frame = cv.resize(bf,None,fx=2, fy=2, \
					interpolation=cv.INTER_CUBIC)
    	laserCtl(0)
    	time.sleep(0.5)

    	diff_frame = cv.absdiff(f_frame, back_frame)
    	min, max, min_loc, max_loc = cv.minMaxLoc(diff_frame)
    	center = np.array(f_frame.shape[:2])/2
	laser = nd.array(max_loc)
    	errVec= center-laser
    	err =  np.linalg.norm(errVec) 
	print u'Error = ',err
    	if err <= 1:
	# complite   
		break
	angle=pixelToAngle(err,no,servoPos,laser,center)
	servoPos = servoPos + angle
	servoCtl(no, servoPos)
    
    return
	
def laserAttack(cx,cy,rx,ry,rxw,ryh, termoImg, camImg):
    laserCtl(100)
    
    return


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
    return img

def FtoDeg(F):
	return int((F-3000)*1.8) 

def LeptonCapture():
    img=''
    for png in exec_cmd('/home/pi/src/flirpi-mjpeg/leptgraypng'):
    	if png is None:
		#print 'Lepton Error'
		return img
    	png = np.fromstring(png, dtype=np.uint8)

	## should be normalize
	'''
	# ToDo
	# more dinamic range need
	pmax = np.max(png)
	pmin = np.min(png)
	coef1=np.ones(png.shape)*1000
	coef2=np.ones(png.shape)*pmin
	coef3=np.ones(png.shape)*(pmax-pmin)
	png = (png-coef1)*coef1/coef3
	'''

    	img = cv.imdecode(png, cv.IMREAD_GRAYSCALE)
    return img

def gammaImage(gamma, img):
    look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
    for i in range(256):
       	look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    return cv.LUT(img, look_up_table)


scl = 5.5
mx = 130
my = 110
dispSW = True
verbosSW = True

argv=sys.argv
argc = len(argv)
if argc > 2:
	for av in argv:
		if av == '-nd':
			dispSW = False
		elif av == '-v':
			verbosSW = True
		elif av == '-nv':
			verbosSW = False
		elif av == '-d':
			dispSW = True

laserCtl(OFF)

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
	if verbosSW == True:
		print 'max='+str(max)+' min='+str(min)
		print 'max_loc='+str(max_loc)+' min_loc='+str(min_loc)

	# avoid noise
	if max > 250 or max <= 31:
		PiCamImg = cv.resize(PiCamImg,None,fx=2, fy=2, \
					interpolation=cv.INTER_CUBIC)
		if dispSW == True:
			cv.imshow('comp image',PiCamImg)  
    			key = cv.waitKey(INTERVAL)
    			if key == ESC_KEY:
    				cv.destroyAllWindows()   
				break
		continue

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
	#imax = np.argmax(data[:,4])
	if n > 1:
		imax = 1
	else:
		imax = 0
	if verbosSW == True:
		print u"max =",imax

	i = imax
	'''
    	print u"各ブロブの外接矩形の左上x座標", data[i,0]
    	print u"各ブロブの外接矩形の左上y座標", data[i,1]
    	print u"各ブロブの外接矩形の幅", data[i,2]
    	print u"各ブロブの外接矩形の高さ", data[i,3]
    	print u"各ブロブの面積", data[i,4]
	print u"各ブロブの中心座標:\n",center[i]
	'''
	rx=data[i,0] # 外接矩形の最小x
	ry=data[i,1] # 外接矩形の最小y
	rw=data[i,2] # 外接矩形の幅
	rh=data[i,3] # 外接矩形の高さ
	cx=int(center[i][0]) # 外接矩形の中心 x
	cy=int(center[i][1]) # 外接矩形の中心 y

	# Grey to Color
	LeptonRGB = cv.cvtColor(LeptonImg,cv.COLOR_GRAY2RGB)
	# make same size image
	height,width  = LeptonRGB.shape[:2]
	#print u"Lepton width ",width
	#print u"Lepton height ",height

	# make scaled image
	PiCamImg = cv.resize(PiCamImg,None,fx=2, fy=2, interpolation=cv.INTER_CUBIC)
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

	gw = mx + width
	gh = my + height
	if gw > lwidth:
		gw = lwidth
	if gh > lheight:
		gh = lheight

	#cv.imshow('Lep image',LeptonImg)  
	#key = cv.waitKey(0)

	CanvasImg[my:gh,mx:gw] = LeptonRGB  

	# make composite image
	CompImg = cv.addWeighted(PiCamImg, 1.0, CanvasImg, 0.5, 0.8)

	# add maker
	cv.circle(CompImg, (cx+mx, cy+my), 2,  (0, 0, 255), 5)
	rrx = rx+int(scl*mx)
	rry = ry+int(scl*my)
	rrx = rx+mx
	rry = ry+my
	CompImg = cv.rectangle(CompImg,(rrx,rry),(rrx+rw,rry+rh),(0,255,0),1)

	if verbosSW == True:
		print u'center ',cx,' ',cy
		print u'rect min',rrx,' ',rry
		print u'rect max',rrx+rw,' ',rry+rh

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
    	cv.imshow(WINDOW_DIFF, diff_frame.astype(np.uint8))

    	# Escキーで終了
    	key = cv.waitKey(INTERVAL)
    	if key == ESC_KEY:
    		cv.destroyAllWindows()   
		break

	laserAttack(cx,cy,rrx,rry,rrx+rw,rry+rh, CanvasImg, PiCamImg)

	if dispSW == True:
		cv.imshow('comp image',CompImg)  
    		key = cv.waitKey(INTERVAL)
    		if key == ESC_KEY:
    			cv.destroyAllWindows()   
			break
		elif key == ord('s'):
    			cv.imwrite('saveimage.jpg', CompImg)
    			cv.destroyAllWindows()
			break
