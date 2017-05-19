#!/usr/bin/python
# coding: utf-8
from __future__ import division
import cv2 as cv
import numpy as np
import time
import sys
import urllib
import math
# from matplotlib import pyplot as plt
#
from pylepton import Lepton

# GPIO
import RPi.GPIO as GPIO

# Configure min and max servo pulse lengths
servoR_min = 150  # Min pulse length out of 4096
servoR_max = 600  # Max pulse length out of 4096

# Import the PCA9685 module.
import Adafruit_PCA9685

# 定数定義
ESC_KEY = 27  # Escキー
INTERVAL = 33  # インターバル
FRAME_RATE = 30  # fps

ON = True
OFF = False

import signal
import RPi.GPIO as GPIO

# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()
# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
gp_out = 27
GPIO.setup(gp_out, GPIO.OUT)
laser = GPIO.PWM(gp_out, 100)
laser.start(0.0);


def receive_signal(signum, stack):
    # print u'Received:', signum
    laser.ChangeDutyCycle(0)
    laserServoCtl(0, staticVar.LaserSERVO_0_ORG)
    laserServoCtl(1, staticVar.LaserSERVO_1_ORG)
    # laserServoCtl(0,staticVar.LaserSERVO_ORG[0])
    # laserServoCtl(1,staticVar.LaserSERVO_ORG[1])
    sensorServoCtl(staticVar.SensorSERVO_ORG)
    laser.stop(0.0);
    GPIO.cleanup()
    sys.exit()


signal.signal(signal.SIGHUP, receive_signal)
signal.signal(signal.SIGINT, receive_signal)


def laserCtl(pwm):
    staticVar.LaserPWM = pwm
    laser.ChangeDutyCycle(pwm)


def servoCtl(no, x):
    # print u'servoCtl no=',no
    # print u'servoCtl x=',x
    pwm.set_pwm(no, 0, int(x))
    # set_servo_pulse(no, int(x))


def sensorServoCtl(xz):
    servoCtl(1, int(xz))


def laserServoCtl(no, xz):
    servoCtl(2 + no * 2, int(xz[1]))
    servoCtl(3 + no * 2, int(xz[0]))
    staticVar.laserServoPos = xz


class staticVar:
    # 　レーザーサーボの原点位置
    LaserSERVO_0_ORG = np.array([365, 300])
    LaserSERVO_1_ORG = np.array([350, 370])
    LaserSERVO_ORG = np.array([[365, 300], [350, 370]])
    #  レーザーの画面位置
    LaserPos = np.zeros(2)
    #  レーザーのPWM
    LaserPWM = 0
    #  カメラサーボの原点位置
    SensorSERVO_ORG = 300
    #  サーボ変数
    laserServoPos = np.array([370, 300])

    #  近似式
    funcXSX = np.zeros((2, 2, 2))
    funcXSY = np.zeros((2, 2, 2))
    funcYSX = np.zeros((2, 2, 2))
    funcYSY = np.zeros((2, 2, 2))
    #  Background
    back_img = np.zeros((60, 80), np.float32)


# レーザーの照射位置を得る
def getLaserPoint():
    laserPWM = staticVar.LaserPWM
    laserCtl(0)  # PWM 0%
    # read backgroud image
    back_frame = MjpgStreamerCapture()

    laserCtl(30)  # PWM 30%
    f_frame = MjpgStreamerCapture()
    # laserCtl(0) # PWM 0%

    diff_frame = cv.subtract(f_frame, back_frame)

    #  Color to GRAY
    diff_frame = cv.cvtColor(diff_frame, cv.COLOR_RGB2GRAY)

    #  max position
    min, max, min_loc, max_loc = cv.minMaxLoc(diff_frame)

    if verbosSW == True:
        print u'getLaserPoint max =', max, 'min =', min, 'laser position  =', max_loc

    center = np.array(f_frame.shape[:2]) / 2  # [h,w]
    laser = np.array([max_loc[1], max_loc[0]])  # [y,x]

    # laser is Gree circle and Center is Red circle
    # BGR   Red is center of image
    cv.circle(diff_frame, (int(center[1]), int(center[0])), 5, (0, 0, 255), 1)
    cv.circle(diff_frame, (int(laser[1]), int(laser[0])), 10, (0, 255, 0), 5)

    cv.imwrite('img.jpg', diff_frame)

    '''
    if dispSW == True:
        cv.imｓ('comp image',f_frame)
        #cv.moveWindow('comp image',640,0)
        cv.imshow('diff',diff_frame)
        cv.moveWindow('diff', 0,480)
            key = cv.waitKey(INTERVAL)
            if key == ESC_KEY:
                    cv.destroyAllWindows()
                    return
            elif key == ord('s'):
                    fileName = 'DiffImage'+str(iter)+'.jpg'
                    cv.imwrite(fileName, f_frame)
    '''

    laserCtl(staticVar.LaserPWM)
    return laser, center

# staticVar.laserServoPos = staticVar.LaserSERVO_ORG[no]

def funcXSX(n, x):
    func = np.poly1d(staticVar.funcXSY[n][0])
    return func(x)


def funcXSY(n, x):
    func = np.poly1d(staticVar.funcYSX[n][0])
    return func(x)


def funcYSX(n, x):
    func = np.poly1d(staticVar.funcXSY[n][1])
    return func(x)


def funcYSY(n, x):
    func = np.poly1d(staticVar.funcYSX[n][1])
    return func(x)


# ralserNo, servoPos, targetPos
def pixelToAngle(no, tPos):
    cPos = staticVar.laserServoPos
    # tPos -> sPos
    tSy = funcYSY(no, tPos[1])
    tx = funcYSX(no, tPos[1])
    tSx = funcXSX(no, tPos[0])
    ty = funcXSY(no, tPos[0])
    dx = cPos[0] - funcXSX(no, tPos[0])
    dy = cPos[1] - funcYSY(no, tPos[1])

    # print u'tSy=',tSy,' tSx=', tSx

    # Add Error correction
    # print u'dx=',dx,' dy=',dy
    dy = 0
    dx = 0

    nSx = tSx - dx
    nSy = tSy - dy
    return [nSx, nSy]


# 動画出力
class export_movie:
    def __init__(self):
        # 入力する動画と出力パスを指定。
        self.result = "attack.xdvi"
        # 形式はMP4Vを指定
        self.fourcc = cv.VideoWriter_fourcc(*'XDVI')
        # 出力先のファイルを開く
        self.out = cv.VideoWriter(self.result, int(self.fourcc), 3.0, (640, 480))

    def writeFrame(self, frame):
        # 読み込んだフレームを書き込み
        self.out.write(frame)

    def close(self):
        self.out.release()


# レーザーで攻撃する
def laserAttack(cx, cy, rx, ry, rw, rh):
    if verbosSW == True:
        print 'Laser Attack'
    laserCtl(80)
    # laserServoCtl(1,[500,staticVar.LaserSERVO_0_ORG[1]])
    # laserServoCtl(1,[500,staticVar.LaserSERVO_ORG[1]])

    if testSW == True:
        # laser,center=getLaserPoint()
        # print u'Resion x y ',rx,ry,' w h ',rw,rh
        for y in range(ry, ry + rh, 10):
            for x in range(rx, rx + rw, 10):
                targetPos = [float(x), float(y)]
                angle = pixelToAngle(0, targetPos)
                # print u'Target ',targetPos,' Angle ',angle
                laserServoCtl(0, angle)
                if dispSW == True:
                    # get PiCam image
                    PiCamImg = MjpgStreamerCapture()
                    # laser is Gree circle and Center is Red circle
                    # BGR   Red is center of image
                    cv.circle(PiCamImg, (int(x), int(y)), \
                              5, (0, 0, 255), 1)
                    cv.rectangle(PiCamImg, (rx, ry), \
                                 ((rx + rw), (ry + rh)), (0, 255, 0), 1)
                    # print dispSW
                    # X11VNCが遅いので小さな画像を表示する
                    PiCamImg = cv.resize(PiCamImg, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
                    cv.imshow('attack image', PiCamImg)
                    # cv.moveWindow('attack image',0,0)
                    key = cv.waitKey(INTERVAL)
                    if key == ESC_KEY:
                        cv.destroyAllWindows()
                        sys.exit()
                        break
    else:
        center = [cx, cy]
        # print u'Target Canter',center
        # angle=pixelToAngle(0,[cx,cy])
        # print u'Angle ',angle
        if rw < 20:
            rw = 20
        if rh < 20:
            rh = 20
        for y in range(ry, ry + rh, 10):
            for x in range(rx, rx + rw, rw - 1):
                targetPos = [float(x), float(y)]
                angle = pixelToAngle(0, targetPos)
                # print u'Target ',targetPos,' Angle ',angle
                laserServoCtl(0, angle)
                if dispSW == True:
                    # get PiCam image
                    PiCamImg = MjpgStreamerCapture()
                    # laser is Gree circle and Center is Red circle
                    # BGR   Red is center of image
                    cv.circle(PiCamImg, (int(x), int(y)), \
                              5, (0, 0, 255), 1)
                    cv.rectangle(PiCamImg, (rx, ry), \
                                 ((rx + rw), (ry + rh)), (0, 255, 0), 1)

                    movieWiter.writeFrame(PiCamImg);
                    # X11VNCが遅いので小さな画像を表示する
                    PiCamImg = cv.resize(PiCamImg, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
                    cv.imshow('attack image', PiCamImg)
                    # cv.moveWindow('attack image',0,0)
                    key = cv.waitKey(INTERVAL)
                    if key == ESC_KEY:
                        cv.destroyAllWindows()
                        sys.exit()
    laserCtl(0)
    laserServoCtl(1, staticVar.LaserSERVO_0_ORG)


# 最大面積のブロッブを選ぶ
def selectBlob(label):
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    # print u'data=',data

    center = np.delete(label[3], 0, 0)
    maxArea = 0
    selectedBlobIndex = 0

    # print u"ブロブの個数:", n
    # print u"各ブロブの外接矩形の左上x座標", data[:,0]
    # print u"各ブロブの外接矩形の左上y座標", data[:,1]
    # print u"各ブロブの外接矩形の幅", data[:,2]
    # print u"各ブロブの外接矩形の高さ", data[:,3]
    # print u"各ブロブの面積", data[:,4]
    # print u"各ブロブの中心座標:\n",center
    i = 1
    for blob in data[1:4]:
        if maxArea < blob[4]:
            maxArea = blob[4]
            selectedBlobIndex = i
        i += 1

    return selectedBlobIndex


def MjpgStreamerCapture():
    #  return 640X480 color image
    img = ''
    stream = urllib.urlopen('http://sensor.local:8080/?action=snapshot')
    bytes = stream.read(320 * 240)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b + 2]
        bytes = bytes[b + 2:]
        encode_param = int(cv.IMWRITE_JPEG_QUALITY)
        c = np.fromstring(jpg, dtype=np.uint8)
        img = cv.imdecode(c, encode_param)
        img = cv.resize(img, None, fx=2, fy=2, \
                        interpolation=cv.INTER_CUBIC)
    return img


import subprocess
from subprocess import Popen, PIPE


def FtoDeg(F):
    t = int((float(F - 30)) * 1.8) / 100
    if t < 0:
        t = 0
    return t


# キャリブレーションファイルを読込む
def readCalibFile():
    for fileno in range(2):
        filename = 'laserServo' + str(fileno) + 'Relations.txt'
        f = open(filename)
        txt = f.read().split('\n')
        i = 0
        for line in txt:
            if line.find('axis') > 0:
                i += 1
                continue

            data = line.replace('[', '').replace(']', '').split(' +')
            data = data[0].split(' ')
            data = filter(lambda s: s != '', data)
            if len(data) == 0:
                i += 1
                continue
            # print u'DATA',data[0],',',data[1]

            data[0] = float(data[0])
            data[1] = float(data[1])
            # print u'readCalibFile= ',data
            if i == 1:
                staticVar.funcXSX[0][fileno] = [data[0], data[1]]
            if i == 3:
                staticVar.funcXSY[0][fileno] = [data[0], data[1]]
            if i == 5:
                staticVar.funcYSX[0][fileno] = [data[0], data[1]]
            if i == 7:
                staticVar.funcYSY[0][fileno] = [data[0], data[1]]
            i = i + 1
        f.close()


# 差分背景の更新
def updateBackground(img):
    # type of img is <list>
    img = np.array(img)  # list to array
    staticVar.back_img = 0.8 * staticVar.back_img + 0.2 * img
    return staticVar.back_img


from pylepton import Lepton

# 　サーマル画像の取得
def LeptonCapture():
    # return 640X480 gray image

    # scale factors of Lepton
    scl = 5.5
    mx = 130
    my = 110

    with Lepton() as l:
    	LeptonImg, _ = l.capture()

    
    for j in range(LeptonImg.shape[0]):
               for i in range(LeptonImg.shape[1]):
                       q = FtoDeg(LeptonImg[j][i])
                       LeptonImg[j][i]=q
                       if LeptonImg[j][i] > 147:
                          #print q,'deg'
                               LeptonImg[j][i]=(q-147)*10+147
                       if LeptonImg[j][i] > 255:
                               LeptonImg[j][i] = 255
                       if LeptonImg[j][i] < 147:
                               LeptonImg[j][i]=0
    if verbosSW == True:
    	print u'mainLoop max', np.max(LeptonImg), np.argmax(LeptonImg), 'min', np.min(LeptonImg), np.argmin(LeptonImg)
    

    LeptonImg = LeptonImg.astype(np.float32)

    # 配列を[60][80]にする。
    LeptonImg = LeptonImg.reshape(60, 80)

    '''
    if dispSW == True:
    	cv.imshow('Temp image',LeptonImg.astype(np.uint8))
    	key = cv.waitKey(INTERVAL)
    	if key == ESC_KEY:
        	cv.destroyAllWindows()
    		sys.exit()
    '''

    # 背景を差し引く
    #back_img = updateBackground(LeptonImg)
    #temp = abs(LeptonImg - back_img)
    temp=LeptonImg

    temp = np.array(temp)
    pmax = np.max(temp)
    ipmax = np.argmax(temp)
    pmin = np.min(temp)
    ipmin = np.argmin(temp)

    '''
    if verbosSW== True:
    	print u'LeptonCapture array max=',pmax,\
        'index=',ipmax,'min=',pmin,'index=',ipmin
    '''

# temp = (temp-pmin)/(pmax-pmin)*255   # float list
    temp = np.array(map(np.uint8, temp))  # list to array
    LeptonImg = temp.reshape(60, 80)

# make scaled image

    LeptonImg = cv.resize(LeptonImg, None, fx=5.5, fy=5.5, interpolation=cv.INTER_CUBIC)
    lheight, lwidth = LeptonImg.shape[:2]
# make scaled image
    CanvasImg = np.zeros([480, 640], np.uint8)
    height, width = CanvasImg.shape[:2]

    gw = mx + lwidth
    gh = my + lheight
    if gw > width:
    	gw = width
    if gh > height:
    	gh = height

    CanvasImg[my:gh, mx:gw] = LeptonImg

    if verbosSW == True:
    	pmax = np.max(CanvasImg)
    	pmin = np.min(CanvasImg)
    	print u'LeptonCapture  CanvasImg   max=', pmax, ' min=', pmin
    return CanvasImg

#  main loop
dispSW = True
verbosSW = False
calSW = True
testSW = False

argv = sys.argv
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

laserServoCtl(1, staticVar.LaserSERVO_0_ORG)  # 3mWレーザーを元に戻す
laserServoCtl(1, staticVar.LaserSERVO_ORG[1])  # 3mWレーザーを元に戻す

# レーザーOFF
laserCtl(0)

laserServoCtl(1, [500, staticVar.LaserSERVO_0_ORG[1]])  # 3mWレーザーを横に向ける
# laserServoCtl(1,[500,staticVar.LaserSERVO_ORG[1][1]]) # 3mWレーザーを横に向ける

# レーザーOFF
laserCtl(0)

# 正面に向ける
laserServoCtl(0, staticVar.LaserSERVO_0_ORG)
laserServoCtl(1, staticVar.LaserSERVO_1_ORG)
sensorServoCtl(staticVar.SensorSERVO_ORG)
time.sleep(0.5)

# ムービーを作る
movieWiter = export_movie()
rng=50
for angle in range(staticVar.SensorSERVO_ORG-rng,staticVar.SensorSERVO_ORG+rng,int(2*rng/10)):
    print u'angle=',angle
    sensorServoCtl(angle)

    # get Lepton image

    # print 'Lepton Capture'
    LeptonImg = LeptonCapture()
    if LeptonImg is None:
        print 'LeptonImg', LeptonImg
        continue

    # get PiCam image
    # print 'PiCam Capture'
    PiCamImg = MjpgStreamerCapture()

    if PiCamImg is None or LeptonImg is None:
        print 'PiCamImg', PiCamImg
        continue

    CanvasImg = cv.cvtColor(LeptonImg, cv.COLOR_GRAY2BGR)

    # make composite image
    CompImg = cv.addWeighted(PiCamImg, 1.0, CanvasImg, 0.5, 0.8)


    # サーマル画像の書き出し
    filename = 'lept%d.jpg' % (angle)
    cv.imwrite(filename,LeptonImg)
    filename = 'img%d.jpg' % (angle)
    cv.imwrite(filename, CompImg)
    movieWiter.writeFrame(CompImg);

    # フレーム表示
    if dispSW == True:
        # X11VNCが遅いので小さな画像を表示する
        CompImg = cv.resize(CompImg, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
        cv.imshow('comp image', CompImg)
        key = cv.waitKey(INTERVAL)
        if key == ESC_KEY:
            cv.destroyAllWindows()
            break
        elif key == ord('s'):
            cv.imwrite('img.jpg', CompImg)
            cv.destroyAllWindows()
            break

    time.sleep(0.1)
sensorServoCtl(staticVar.SensorSERVO_ORG)

