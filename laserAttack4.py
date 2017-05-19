#!/usr/bin/python
#coding: utf-8
from __future__ import division
import cv2 as cv
import numpy as np
from scipy import ndimage
import time
import sys
import urllib
import math
#from matplotlib import pyplot as plt
#
from pylepton import Lepton

# timeout
from timeout import on_timeout

# GPIO
import RPi.GPIO as GPIO

# Import the PCA9685 module.
import Adafruit_PCA9685

# 定数定義
ESC_KEY = 27     # Escキー
INTERVAL= 33     # インターバル
#INTERVAL= 0     # インターバル
FRAME_RATE = 30  # fps

# Calibration sampling number
nSample=20
#
laserPWM=20

#
# Lepton Factors
#
#scl=5.5
#mx = 130
#my = 110
#
SclX=5.4
SclY=5.4
Mx=130
My=60
Rot=1.1
#
#
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
gp_out=27
GPIO.setup(gp_out, GPIO.OUT)
laser = GPIO.PWM(gp_out,100)
laser.start(0.0);

import spidev

def receive_signal(signum, stack):
    #print u'Received:', signum
    laser.ChangeDutyCycle(0)
    laserServoCtl(LaserSERVO_ORG)
    sensorServoCtl(SensorSERVO_ORG)
    laser.stop(0.0);
    GPIO.cleanup() 
    sys.exit()

# 終了時にレーザーをOFFにする
signal.signal(signal.SIGHUP, receive_signal)
signal.signal(signal.SIGINT, receive_signal)

def laserCtl(pwm):
    global LaserPWM
    LaserPWM = pwm
    laser.ChangeDutyCycle(pwm)

def servoCtl(no, x):
    # リミットを付けて暴走しないようにする
    x= limit(x)
    pwm.set_pwm(no, 0, int(x))
    #time.sleep(0.1) # sleep 100msec

def sensorServoCtl(xz):
    # リミットを付けて暴走しないようにする
    global SensorServoPos
    #
    if xz < 270:
        xz = 270
    if xz > 500:
        xz = 500
    #
    SensorServoPos=int(xz)
    servoCtl(1,int(xz))

def laserServoCtl(xz):
    global laserServoPos
    xz[0]=limit(xz[0])
    xz[1]=limit(xz[1])
    laserServoPos = xz
    servoCtl(3,int(xz[0])) # 水平回転
    servoCtl(2,int(xz[1])) # 仰角回転
    return xz

#　レーザーサーボの原点位置
LaserServoORGX = 375
LaserServoORGY = 375
'''
# 384, 404
LaserServoORGX = 384
LaserServoORGY = 404 
'''
LaserSERVO_ORG = np.array([LaserServoORGX,LaserServoORGY])
#  レーザーの画面位置
LaserPos = np.zeros(2)
#  レーザーのPWM
LaserPWM=20

#  カメラサーボの原点位置
SensorSERVO_ORG=300
SensorServoPos=0
#  サーボ変数
laserServoPos = np.array([430,200]) 
#  近似式
funcXSX = np.zeros(2)
funcXSY = np.zeros(2)
funcYSX = np.zeros(2)
funcYSY = np.zeros(2)
matrixM = np.ones([3,3])
#  Background 
back_img = np.zeros((60,80),np.float32)
# precisionTableが使えるかどうか判定
precisionTableState=False 
# 正確な位置合わせテーブル
precTable= np.zeros((6,8,3),np.uint16)
#
#
Thresh=8

def displayLaserPoint(target, lp):
    if lp is None:
        return
    target = tuple(np.array(target,dtype=np.uint16))
    lp = tuple(np.array(lp,dtype=np.uint16))

    # get PiCam image
    #PiCamImg = MjpgStreamerCapture()
    PiCamImg=PiCamCapture()
    # laser is Gree circle and Center is Red circle
    # BGR   Red is center of image
    #
    # Greep Circle is laser point
    cv.line(PiCamImg, (lp[0],lp[1]),  (target[0],target[1]), (0, 255, 0), 1)
    # Red Circle
    cv.circle(PiCamImg, (target[0],target[1]),\
              int(Thresh),  (0, 0, 255), 1)
        
    # Blue Circle
    cv.circle(PiCamImg,  (lp[0],lp[1]),\
                        12,  (255, 0, 0), 1)
              
    # Yellow circle
    #cv.circle(PiCamImg,  (target[0],target[1]),\
    #                    3,  (0, 255, 255), 1)
    # Lepton Image範囲
    mx = Mx 
    my = My
    mw=int(80*SclY)
    mh=int(60*SclX)
    kw=8
    kh=6
    dx = int(mw/(kw-1)+0.5)
    dy = int(mh/(kh-1)+0.5)
    # grid
    for yy in range(my,my+mh+int(dy/2),dy):
        cv.line(PiCamImg, (mx,yy), (mx+mw,yy), (255, 255, 255), 1)
    for xx in range(mx,mx+mw+int(dx/2),dx):
        cv.line(PiCamImg, (xx,my), (xx,my+mh), (255, 255, 255), 1)
            
    #print dispSW
    cv.imshow('Laser Position',PiCamImg)
    key = cv.waitKey(INTERVAL)
    if key == ESC_KEY:
        cv.destroyAllWindows()
        sys.exit()

##
def adjustLaser(target):
    global precisionTableState
    target=np.array(target,dtype=np.uint16)
    if verbosSW==True:
        print u'adjustLaser  precisionTableState =',precisionTableState,'reLearn =',relearnSW,'target=',target
    
    # Lepton Image範囲
    mx = Mx
    my = My
    #
    searchMax = 10
    #
    mw=int(80*SclX)
    mh=int(60*SclY)
    
    # 修正した目標座標
    target    = np.array(target,dtype=np.float32)
    # 目標座標
    targetPos = target.copy()
    position=np.array(2,dtype=np.uint16)
    
    for loop in range(searchMax):
        sensorServoCtl(SensorSERVO_ORG)
        laserServoPos=pixelToAngle(target)
        laserServoPos=laserServoCtl(laserServoPos)
        
        lp = getLaserPoint()
        if lp is None:
            # 除外 
            position=np.zeros(3)
            if verbosSW == True:
                print u'adjustLaser save ',position
                #print u'ORG [',LaserSERVO_ORG,'] laser=',lp
            break
            
        lp = np.array(lp,dtype=np.float32)
        
        err = lp-targetPos # 誤差ベクトル
        errNorm = math.sqrt(np.dot(err,err))
        
        if verbosSW:
            #print u'Pos=',[x,y],'target=',target,'nowPos=',lp,\
            #'err=',err,'Enorm=',errNorm,'OldNorm=',oldErr
            print u'Loop=',loop,'err=',err, 'adjustLaser Norm=',errNorm,'targetPos=',targetPos,'Laser=',lp

        # 画像表示して確認する
        if dispSW == True:
            displayLaserPoint(targetPos, lp)
        
        if errNorm > Thresh:
            # 現在のサーボの角度を少し回転させる
            target = target-0.8*err
            
            if loop == searchMax/2:
                # global search
                target=globalSearch(target)
                continue
            if loop == searchMax-1:
                position=laserServoPos
                if verbosSW == True:
                    print u'adjustLaser save ',position
                break
        elif errNorm <= Thresh:
            # 正確なサーボ制御角度を保存する
            laserServoPos=pixelToAngle(target)
            position=laserServoCtl(laserServoPos)
            if verbosSW == True:
                print u'adjustLaser save ',position
                #print u'ORG [',LaserSERVO_ORG,'] laser=',lp
            break
    return position


def getLaserPoint():
    global laserPWM
    laserCtl(0) # PWM 0%
    time.sleep(0.5)
    # read backgroud image
    #back_frame = MjpgStreamerCapture()
    back_frame=PiCamCapture()
    
    # 朝方、明るいときはレーザーパワーを高くする
    # 夜、暗い時は、ハレーションが起こるので弱くする
    laserCtl(laserPWM) # PWM 20%
    time.sleep(0.5)
    #f_frame = MjpgStreamerCapture()
    f_frame=PiCamCapture()
    #laserCtl(0) # PWM 0%
    
    diff_frame = cv.subtract(f_frame, back_frame)
    
    #  Color to GRAY
    diff_frame = cv.cvtColor(diff_frame,cv.COLOR_RGB2GRAY)
    
    #  max position
    min, max, min_loc, max_loc = cv.minMaxLoc(diff_frame)
    
    # 最大輝度点が見つからなければ、レーザーパワーを高くして再トライする
    laser  = max_loc;
    if max < 10.0:
        laserPWM = laserPWM + 10
        print u'Power up',laserPWM
        if laserPWM >= 60:
            laserPWM = 20.0
            return None
        laser=getLaserPoint()

    if verbosSW == True and testSW == False:
        print u'getLaserPoint max =',max,'min =',min,'laser position  =',max_loc
    
    # laser is Gree circle and Center is Red circle
    # BGR   Red is center of image
    #if dispSW == True: 
    #    displayLaserPoint(laser, laser)
    
    #laserPWM=0.0
    laserCtl(0.0)
    
    return np.array(laser)

def savePrecisionTable(precImg):
    filename='PrecisionTableImg.png'
    cv.imwrite(filename, precImg)

def globalSearch(target):
    laserServoCtl(pixelToAngle(target))
    time.sleep(0.1)

    # -10, +10
    # -10, +10
    # search min norm
    targetPos=target
    spot = targetPos.copy()
    normMin=1000000.0
    for y in range(-10,10,10):
        for x in range(-10,10,10):
            target = targetPos + [x,y]
        #print 'target=',target
            laserServoCtl(pixelToAngle(target))
        time.sleep(0.1)
        lp=getLaserPoint()
        if lp is None:
            print u'Can not get laser position'
            #原点に戻す
            laserServoCtl(LaserSERVO_ORG)
            continue
        #
        lp=np.array(lp,dtype=np.float32)
        # 誤差ベクトル
        err = lp-targetPos 
        errNorm = math.sqrt(np.dot(err,err))
        if verbosSW==True:
            print u'globalSearch error [',x,y,'] norm=',errNorm
        if errNorm < normMin:
            normMin=errNorm
            spot=target
    if verbosSW==True:
        print u'globalSearch restart=',spot
    return spot

def getLeptonCenter():
    mx = Mx
    my = My
    mw=int(80*SclX)
    mh=int(60*SclY)
    LeptonCenter=[int(mx+mw/2),int(my+mh/2)]
    return LeptonCenter

def getPiCamCenter():
    PiCamCenter=[320,240]

# Lepton 撮像範囲の四隅に正確にレーザーを照射する
def precisionLaser():
    global precTable
    global precisionTableState
    global Thresh

    #precisionTableState = False
    if verbosSW==True:
        print u'precisionTableState =',precisionTableState,'reLearn =',relearnSW
    pixelval=np.zeros(3)
    # Lepton Image範囲  
    #
    mx = Mx
    my = My
    # しきい値
    # 10 > Thresh > 6
    #Thresh = 10
    #
    searchMax = 10 
    #
    mw=int(80*SclX)
    mh=int(60*SclY)
    target=np.zeros([2])
    err=np.zeros([2])
    laserCtl(20)

    #kw=8
    #kh=6

    kw=4
    kh=3

    #kw=2
    #kh=2
    dx = mw/(kw-1)
    dy = mh/(kh-1)

    pprecTable= np.zeros((kh,kw,3),np.uint16)
    
    for iy in range(0,kh):
        for ix in range(0,kw):
            # 修正した目標座標
            x = int(mx+ix*dx+0.5)
            y = int(my+iy*dy+0.5)
            target = np.array([x,y],dtype=np.float32)
            pprecTable[iy][ix][:2]=adjustLaser(target)

    # 内挿処理
    # (8+1)X(6+1) -> 80X60 -> SclX*80 X SclY*60

    if verbosSW:
        print u'precTable=',pprecTable
        print u'precTable.shape=',pprecTable.shape
        print u'precTable.dtype=',pprecTable.dtype

    np.save('precTable.npy', pprecTable)

    with open('precTable.csv','wb') as f:
        for row in pprecTable:
            for xyz in row:
                    np.savetxt(f, xyz, delimiter=',', fmt='%d')

    precTable=pprecTable.astype(np.uint16)

    precTable = cv.resize(precTable,None,fx=int(SclX*80.0/precTable.shape[1]+0.5), \
                          fy=int(SclY*60.0/precTable.shape[0]+0.5), interpolation=cv.INTER_LINEAR)

    gw = int(80*SclX+0.5)
    gh = int(60*SclY+0.5)
    height=480
    width =640
    if gw+mx > width:
        gw = width-mx
    if gh+my > height:
        gh = height-my

    precTable=precTable.astype(np.uint16)
    tmp = np.zeros([480,640,3],np.uint16)
    tmp[my:my+gh,mx:mx+gw] = precTable
    precTable = tmp.copy()

    # 保存
    np.save('precImg.npy', precTable)
    savePrecisionTable(precTable.astype(np.uint8))
    precisionTableState = True

    laserCtl(0)

    return precTable


def Verification():
    global precTable
    global precImg
    global precisionTableState
    mx = Mx
    my = My
    mw=int(80*SclX)
    mh=int(60*SclY)
    if verbosSW==True:
        print u'Verification'
    '''
    precTable=np.load('precTable.npy')
    print u'precTable.shape=',precTable.shape
    print u'precTable=',precTable
    laserCtl(50)
    for  pxy in precTable: 
        for xy in pxy: 
                print u'sx,sy =',xy
                laserServoCtl(xy)
        time.sleep(1)

    laserCtl(0)
    '''
    precTable=np.load('precImg.npy')
    precisionTableState = True
    laserCtl(50)
    skh = 3
    skw = 4
    dx = int(mw/(skw-1))
    dy = int(mh/(skh-1))
    errMap=np.zeros([skh,skw,3],dtype=np.uint16)
    print 'errMap.shape=',errMap.shape
    # 画像表示して確認する
    for iy in range(0,skh):
        for ix in range(0,skw):
        # 目標座標
            x = int(ix*dx+mx+0.5)
            y = int(iy*dy+my+0.5)
            
            if y == 440:
                y -= 1
                
            target = np.array([x,y],dtype=np.float32)
            if precTable[y][x][0] == 0 and precTable[y][x][1] == 0:
                print u'skip x,y =',x,y
                continue
            print u'target =',target
            #laserServoCtl(pixelToAngle(target))
            laserServoCtl(precTable[y][x])
            lp = getLaserPoint()
            if lp is None:
                continue
            err = lp-target # 誤差ベクトル
            errNorm = math.sqrt(np.dot(err,err))
            #ix = int((x-mx)/dx)
            #iy = int((y-my)/dy)
            errMap[iy][ix][0]=errNorm
            
            # 画像表示して確認する
            if dispSW == True:
                displayLaserPoint(target,lp)
                time.sleep(0.25)
                
    errMap = cv.resize(errMap,None,fx=SclX*80.0/errMap.shape[1], \
                          fy=SclY*60.0/errMap.shape[0], interpolation=cv.INTER_LINEAR)

    gw = int(80*SclX)
    gh = int(60*SclY)
    height=480
    width =640
    if gw+mx > width:
        gw = width-mx
    if gh+my > height:
        gh = height-my

    errMap=errMap.astype(np.uint16)
    tmp = np.zeros([480,640,3],np.uint16)
    tmp[my:my+gh,mx:mx+gw] = errMap
    errMap = tmp
    cv.imwrite('errMap.png',errMap.astype(np.uint8))
    laserCtl(0)
           
# レーザーの照射位置と画像の画素との対応を取る
def laserCaliblationAxis(checkXZ):
    # checkXZ = 0/1  0 then Z axis, 1 then X axis rotate

    # ３以下だと、サーボ位置が変わらないことがある
    if checkXZ == 0:
        sStep = 3
    if checkXZ == 1:
        sStep = 3
        
    reTry = 0
    oldxy = [0,0]

    # data for interpolate
    xData  = np.zeros(nSample)
    yData  = np.zeros(nSample)
    sxData = np.zeros(nSample)
    syData = np.zeros(nSample)

    # rotae to ORIGIN position
    global LaserSERVO_ORG
    LaserSERVO_ORG = np.array([LaserServoORGX, LaserServoORGY])
    laserServoPos = np.array([LaserServoORGX, LaserServoORGY]) 
    sensorServoCtl(SensorSERVO_ORG)

    while True:
        laser=getLaserPoint()
        if laser is not None:
            break
    
    if checkXZ == 0:
        laserServoPos[0] = laserServoPos[0]-int(nSample*sStep/2) 
        laserServoPos[1] = LaserServoORGY
    else: # checkXZ == 1
        laserServoPos[0] = LaserServoORGX 
        laserServoPos[1] = laserServoPos[1]-int(nSample*sStep/2) 

    laserServoCtl(laserServoPos)
    if verbosSW==True:
        print u'ServoPos',laserServoPos,'laser=',laser

    iter=0
    while True:
        # check laser point
        laser=getLaserPoint()
        if laser is None:
            continue

        # レーザー点の位置がおかしければやり直す
        if iter > 1 and (abs(laser[0]-oldxy[0]) > 150 or abs(laser[1]-oldxy[1]) > 150):    
            laserServoPos = laserServoPos + sAngle
            continue

        oldxy=[laser[0],laser[1]]
        
        xData[iter]=laser[0]
        yData[iter]=laser[1]
        sxData[iter]=laserServoPos[0]
        syData[iter]=laserServoPos[1]
        
        # 一度、原点に戻す（サーボが回転するように）
        laserServoCtl(LaserSERVO_ORG)
        time.sleep(0.2)
    
        # X axis
        if checkXZ == 0:
            sAngle = [sStep, 0]
        # Z axis
        if checkXZ == 1:
            sAngle = [0, sStep]
            
        laserServoPos = laserServoPos + sAngle
        laserServoCtl(laserServoPos)

        if verbosSW==True:
            print u'ServoPos',laserServoPos,'laser=',laser

        time.sleep(0.5)

        if iter >= (nSample-1):
            break
        iter += 1

    if dispSW == True:
        cv.destroyAllWindows()

    # resize buffer
    xData=np.resize(xData,iter)
    yData=np.resize(yData,iter)
    sxData=np.resize(sxData,iter)
    syData=np.resize(syData,iter)

    # ファイルに書き出す
    filename = 'laserServo'+str(checkXZ)+'.csv'
    f = open(filename, 'w')
    for i in range(len(xData)):
        msg = '%d,%d,%d,%d,%d\n' % (i,xData[i],yData[i],sxData[i],syData[i]) 
        f.write(msg)
    f.close()
    laserServoCtl(LaserSERVO_ORG)

# ralserNo, servoPos, targetPos
def pixelToAngle(tPos):
    tPos = np.array(tPos)
    # 精密な位置合わせ
    if precisionTableState == True or relearnSW == True:
        tPos = tPos.astype(np.uint16)
        ix = tPos[0]
        iy = tPos[1]
        if ix < precTable.shape[1] and iy < precTable.shape[0]:
            # Leptonの範囲内
            if precTable[iy][ix][0] > 0:
                xz = np.zeros(2)
                xz[0] = precTable[iy][ix][0]
                xz[1] = precTable[iy][ix][1]
                # Leptonの範囲外
            else:
                xz = pixelToAngle1(tPos)
        else:
            xz = pixelToAngle1(tPos)
    else:
        xz = pixelToAngle1(tPos)
 
    return xz

# ralserNo, servoPos, targetPos
def pixelToAngle1(tPos):
    cPos=laserServoPos
    tPos = np.array(tPos,dtype=np.float32)

    # tPos -> sPos
    tSy=funcYSY(tPos[1])
    tx =funcYSX(tPos[1])
    tSx=funcXSX(tPos[0])
    ty =funcXSY(tPos[0])
    
    dx =cPos[0]-funcXSX(tPos[0])
    dy =cPos[1]-funcYSY(tPos[1])
    #dx =funcXSX(tPos[0])
    #dy =funcYSY(tPos[1])

    #print u'tSy=',tSy,' tSx=', tSx
    # Add Error correction
    #print u'dx=',dx,' dy=',dy
    dy = 0 
    dx = 0

    nSx=int(tSx-dx)
    nSy=int(tSy-dy)
    return [nSx,nSy] 

    # Add Error correction
class export_movie:
    def __init__(self):
    # 入力する動画と出力パスを指定。
        self.result = "attack.xdvi" 
    # 形式はMP4Vを指定
        self.fourcc = cv.VideoWriter_fourcc(*'MP4')
    # 出力先のファイルを開く
        self.out = cv.VideoWriter(self.result, int(self.fourcc), 3.0, (640, 480))

    def writeFrame(self,frame):
    # 読み込んだフレームを書き込み
        self.out.write(frame)

    def close(self):
        self.out.release()

smax=500
smin=270
    
def limit(x):
    x = int(x)
    if x < smin:
        x = smin
    if x > smax:
        x = smax
    return x

def servoLimit(cPos):
    # limit
    cPos=np.array(cPos,dtype=np.uint16)
    if cPos[0] > smax:
        cPos[0]=smax
    if cPos[1] > smax:
        cPos[1]=smax
    if cPos[0] < smin:
        cPos[0]=smin
    if cPos[1] < smin:
        cPos[1]=smin
    return cPos

# laserAttack
def laserAttack():
    global Thresh
    (cx,cy,rx,ry,rw,rh)=Target_Resion
    
    if verbosSW == True:
        print 'Laser Attack'

    laserCtl(60)

    center=[cx,cy]
    # ショットガン
    if verbosSW == True:
        print u'Shut Gun  Target Center',center
    Thresh=50  
    cPos=adjustLaser(center)

    # get PiCam image
    #PiCamImg = MjpgStreamerCapture()
    PiCamImg=PiCamCapture()
    # laser is Gree circle and Center is Red circle
    # BGR   Red is center of image
    cv.circle(PiCamImg, (int(cx),int(cy)),\
                            5,  (0, 0, 255), 1)
    cv.rectangle(PiCamImg,(rx,ry),\
                        ((rx+rw),(ry+rh)),(0,255,0),1)

    fileName = 'images/Attack'+str(int(time.time()*100))+'.jpg'
    cv.imwrite(fileName,PiCamImg);

    #print dispSW
    if dispSW == True:
            cv.imshow('attack image',PiCamImg)
            #cv.moveWindow('attack image',0,0)
            key = cv.waitKey(INTERVAL)
            if key == ESC_KEY:
                cv.destroyAllWindows()
                sys.exit()

    # limit
    cPos = servoLimit(cPos)

    # マシンガン
    # machin gun
    print u'Machine Gun'
    laserCtl(90)
    angle=np.zeros(2)
    for loop in range(30):
        for y in range(-5,6,5):
            for x in range(-5,6,5):
                angle = [cPos[0]+x,cPos[1]+y]
                laserServoCtl(angle)
                print u'Machine Gun[servo]',angle, 'center ',cPos
                '''
                # get PiCam image
                #PiCamImg = MjpgStreamerCapture()
                PiCamImg=PiCamCapture()
                # laser is Gree circle and Center is Red circle
                # BGR   Red is center of image
                cv.circle(PiCamImg, (int(x),int(y)),\
                          5,  (0, 0, 255), 1)
                cv.rectangle(PiCamImg,(rx,ry),\
                             ((rx+rw),(ry+rh)),(0,255,0),1)

                fileName = 'images/Attack'+str(int(time.time()*100))+'.jpg'
                cv.imwrite(fileName,PiCamImg);
                #print dispSW
                if dispSW == True:
                    cv.imshow('attack image',PiCamImg)
                    key = cv.waitKey(INTERVAL)
                    if key == ESC_KEY:
                        cv.destroyAllWindows()
                        sys.exit()
               '''
    laserCtl(0)

def selectBlob(label):
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    #print u'data=',data

    center = np.delete(label[3], 0, 0)
    maxArea = 0
    selectedBlobIndex=0

    #print u"ブロブの個数:", n
    #print u"各ブロブの外接矩形の左上x座標", data[:,0]
    #print u"各ブロブの外接矩形の左上y座標", data[:,1]
    #print u"各ブロブの外接矩形の幅", data[:,2]
    #print u"各ブロブの外接矩形の高さ", data[:,3]
    #print u"各ブロブの面積", data[:,4]
    #print u"各ブロブの中心座標:\n",center
    i=0
    for blob in data[:4]:
        if i==0:
            i += 1
            continue

        if blob[4] > 150000:
            i += 1
            continue

        if maxArea < blob[4]:
            maxArea = blob[4]
            selectedBlobIndex=i
            
        i += 1
    
    return selectedBlobIndex
    
def MjpgStreamerCapture():
#  return 640X480 color image
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

'''
from picamera.array import PiRGBArray
from picamera import PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.sharpness = 0
camera.contrast = 0
camera.brightness = 50
camera.saturation = 0
camera.ISO = 0
camera.video_stabilization = False
camera.exposure_compensation = 0
camera.exposure_mode = 'auto'
camera.meter_mode = 'average'
camera.awb_mode = 'auto'
camera.image_effect = 'none'
camera.color_effects = None
camera.rotation = 0
camera.hflip = True
camera.vflip = True
camera.crop = (0.0, 0.0, 1.0, 1.0)
'''

def PiCamCapture():
    rawCapture=None
    while rawCapture is None:
        rawCapture = PiRGBArray(camera, size=(640, 480))
        #time.sleep(0.1)
    # grab an image from the camera
    camera.capture(rawCapture, format="bgr")
    img = rawCapture.array
    cv.imwrite('/mnt/photo/PiCamImg.jpg',img.astype(np.uint8))
    return img

def PiCamCaptureORG():
    #img = PiCamCapture()
    #img = cv.resize(img,None,fx=0.5, fy=0.5,interpolation=cv.INTER_CUBIC)
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

import subprocess
from subprocess import Popen, PIPE

def FtoDeg(F):
        t = int((float(F-30))*1.8)/100
        if t < 0:
                t = 0
        return t

def updateBackground(img):
    # type of img is <list>
    img = np.array(img) # list to array
    global back_img
    back_img=0.8*back_img+0.2*img
    return back_img

import threading

class LeptonCaptureThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        ret=LeptonCapture()
        self.return_value = ret   # SET RETURN VALUE

    def get_value(self):  # GETTER METHOD
        return self.return_value

class PiCamCaptureThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        #ret=MjpgStreamerCapture()
        ret=PiCamCapture()
        self.return_value = ret   # SET RETURN VALUE

    def get_value(self):  # GETTER METHOD
        return self.return_value
    
class LaserAttackThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        ret=laserAttack()
        self.return_value = ret   # SET RETURN VALUE

    def get_value(self):  # GETTER METHOD
        return self.return_value
    

def LeptonCapture():
# return 640X480 gray image
# scale factors of Lepton
    mx = Mx
    my = My
    with Lepton() as l:
        LeptonImg,_ = l.capture()
    '''
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
    '''
    errorNo=0
    for j in range(LeptonImg.shape[0]):
        for i in range(LeptonImg.shape[1]):
            if LeptonImg[j][i] >= 80000:
                errorNo += 1

    '''
    if verbosSW == True and errorNo > 0:
        print u'mainLoop max',np.max(LeptonImg),np.argmax(LeptonImg),\
        'min',np.min(LeptonImg), np.argmin(LeptonImg),'Error ',errorNo
    '''
        
    if errorNo > 10:
        print u'Lepton Error Data:',errorNo
        return None
        
    LeptonImg = LeptonImg.astype(np.float32)
    LeptonImg= LeptonImg.reshape(60, 80)
    cv.imwrite('/mnt/photo/LeptonImg.jpg',LeptonImg.astype(np.uint8))

    '''
    if dispSW == True:
        cv.imshow('Temp image',LeptonImg.astype(np.uint8))  
        key = cv.waitKey(INTERVAL)
        if key == ESC_KEY:
            cv.destroyAllWindows()   
            sys.exit()
    '''

    # 背景を差し引く
    if subSW==True:
        back_img=updateBackground(LeptonImg)
   	#temp=abs(LeptonImg - back_img)
        temp = cv.subtract(LeptonImg, back_img)
    else:
        temp =LeptonImg

    temp = np.array(temp) 
    pmax = np.max(temp)
    ipmax=np.argmax(temp)
    pmin = np.min(temp)
    ipmin=np.argmin(temp)

    '''
    if verbosSW== True:
    print u'LeptonCapture array max=',pmax,\
            'index=',ipmax,'min=',pmin,'index=',ipmin
    '''

    #temp = (temp-pmin)/(pmax-pmin)*255   # float list
    temp = np.array(map(np.uint8, temp)) # list to array
    LeptonImg = temp.reshape(60, 80)

    # make scaled image
    
    LeptonImg = cv.resize(LeptonImg,None,fx=SclX, fy=SclY,interpolation=cv.INTER_CUBIC)
    lheight,lwidth = LeptonImg.shape[:2]
    # make scaled image
    CanvasImg = np.zeros([480,640],np.uint8)
    height,width = CanvasImg.shape[:2]

    gw = mx + lwidth
    gh = my + lheight
    if gw > width:
        gw = width
    if gh > height:
        gh = height

    CanvasImg[my:gh,mx:gw] = LeptonImg  

    '''
    if dispSW == True:
        cv.imshow('Lepton image',CanvasImg)  
        #cv.moveWindow('Lepton image',640,0)
        key = cv.waitKey(INTERVAL)
        if key == ESC_KEY:
            cv.destroyAllWindows()   
            sys.exit()
    '''
    '''
    if verbosSW == True:
        pmax = np.max(CanvasImg)
        pmin = np.min(CanvasImg)
        print u'LeptonCapture  CanvasImg   max=',pmax,' min=',pmin
    '''
    cv.imwrite('/mnt/photo/CanvasImg.jpg',CanvasImg.astype(np.uint8))
    return CanvasImg

def LeptonCaptureORG():
# return 640X480 gray image
# scale factors of Lepton
    mx = Mx
    my = My
    with Lepton() as l:
        LeptonImg,_ = l.capture()
    errorNo=0
    for j in range(LeptonImg.shape[0]):
        for i in range(LeptonImg.shape[1]):
            if LeptonImg[j][i] >= 80000:
                errorNo += 1
        
    if errorNo > 10:
        print u'Lepton Error Data:',errorNo
        return None
        
    LeptonImg = LeptonImg.astype(np.float32)
    LeptonImg = LeptonImg.reshape(60, 80)
    cv.imwrite('/mnt/photo/LeptonImgORG.jpg',LeptonImg.astype(np.uint8))
    return LeptonImg

def gammaImage(gamma, img):
    look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
    for i in range(256):
           look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    return cv.LUT(img, look_up_table)

def readCalibFile():
    global funcXSX
    global funcXSY
    global funcYSX
    global funcYSY

    PLT =   False
    if PLT:
        import matplotlib.pyplot as plt

    #####  laserServo0 #####
    f = open('laserServo0.csv')
    data1 = f.read()  # ファイル終端まで全て読んだデータを返す
    f.close()
    lines1 = data1.split('\n') # 改行で区切る(改行文字そのものは戻り値のデータには含まれない)
    n=len(lines1)-1
    
    #
    xData0  = np.zeros(n)
    yData0  = np.zeros(n)
    sxData0 = np.zeros(n)
    syData0 = np.zeros(n)

    i=0
    for line in lines1:
        data = line.split(',')
        if data[0] is None or i >= n:
            break
        xData0[i]  = data[1]
        yData0[i]  = data[2]
        sxData0[i] = data[3]
        syData0[i] = data[4]
        i += 1
    if PLT:
        #print('xData',xData0)
        #print('sxData',sxData0)
        plt.subplot(221)
        plt.plot(xData0, sxData0, '.')
    p = np.polyfit(xData0, sxData0, 1)
    funcXSX = np.poly1d(p)
    if PLT:
        plt.plot(xData0, funcXSX(xData0), '-')
        plt.legend((funcXSX,),loc='lower right') # 凡例

        plt.xlim(xmin=0,xmax=500)
        plt.ylim(ymin=0,ymax=500)

        plt.subplot(222)
        plt.plot(yData0, sxData0, '*')
        
    p = np.polyfit(yData0, sxData0, 1)
    funcYSX = np.poly1d(p)
    
    if PLT:
        plt.plot(yData0, funcYSX(yData0), '-')
        plt.legend((funcYSX,),loc='lower right') # 凡例
        plt.xlim(xmin=0,xmax=500)
        plt.ylim(ymin=0,ymax=500)


    #####  laserServo1 #####
    f = open('laserServo1.csv')
    data1 = f.read()  # ファイル終端まで全て読んだデータを返す
    f.close()

    lines1 = data1.split('\n') # 改行で区切る(改行文字そのものは戻り値のデータには含まれない)
    n=len(lines1)-1
    #
    xData1  = np.zeros(n)
    yData1  = np.zeros(n)
    sxData1 = np.zeros(n)
    syData1 = np.zeros(n)

    i=0
    for line in lines1:
        data = line.split(',')
        if data[0] is None or i >= n:
            break
        xData1[i]  = data[1]
        yData1[i]  = data[2]
        sxData1[i] = data[3]
        syData1[i] = data[4]
        i += 1

    if PLT:
        plt.subplot(223)
        plt.plot(yData1,syData1, '.')
    
    q = np.polyfit(yData1, syData1, 1)
    funcYSY = np.poly1d(q)
    
    if PLT:
        plt.plot(yData1,funcYSY(yData1), '-')
        plt.legend((funcYSY,),loc='lower right') # 凡例
        plt.xlim(xmin=0,xmax=500)
        plt.ylim(ymin=0,ymax=500)

        plt.subplot(224)
        plt.plot(xData1, syData1, '*')
    
    p = np.polyfit(xData1, syData1, 1)
    funcXSY = np.poly1d(p)
    
    if PLT:
        plt.plot(xData1, funcXSY(xData1), '-')
        plt.legend((funcXSY,),loc='lower right') # 凡例
        plt.xlim(xmin=0,xmax=500)
        plt.ylim(ymin=0,ymax=500)
        plt.show()
    
    
    return funcXSX,funcXSY,funcYSX,funcYSY

def funcM(M, xy):
    # xy =[lx,ly]
    # M [3X3] matrix, xy [3] vector
    vec=np.ones(3)
    vec[0]=xy[0]
    vec[1]=xy[1]
    res = np.dot(M,vec)
    #print('res=',res)
    return res

M_Affine=np.load('M.npy')
    
def getCompImg(LeptonImg,PiCamImg):
    rows = LeptonImg.shape[0]
    cols = LeptonImg.shape[1]
    dst = cv.warpAffine(LeptonImg,M_Affine,(cols,rows))
    # return 320X240 image
    CompImg = cv.addWeighted(PiCamImg, 1.0, LeptonImg, 0.8, 0.8)

    # return 640X480 image
    # CompImg = cv.resize(CompImg,None,fx=2.0, fy=2.0, interpolation=cv.INTER_LINEAR)
    return CompImg


def LeptonCalibImg():
    print u'Lepton ImageとPiCam Imageを重ね合わせるキャリブレーションを実行します。電子ライター３つを配置してください。'
    # 背景撮影
    P0=PiCamCaptureORG()
    cv.imwrite('/mnt/photo/P0.jpg',P0.astype(np.uint8))
    L0=LeptonCaptureORG()
    for n in range(5):
        L0 += LeptonCaptureORG()
    L0 = L0*0.2
    cv.imwrite('/mnt/photo/L0.jpg',L0.astype(np.uint8))
    # 点熱源ON
    print u'電子ライター1をONにしてください'
    raw_input()
    P01=PiCamCaptureORG()
    cv.imwrite('/mnt/photo/P01.jpg',P01.astype(np.uint8))
    L01=LeptonCaptureORG()
    for n in range(5):
        L01 += LeptonCaptureORG()
    L01 = L01*0.2
    cv.imwrite('/mnt/photo/L01.jpg',L01.astype(np.uint8))
    print u'電子ライター2をONにしてください'
    raw_input()
    P02=PiCamCaptureORG()
    cv.imwrite('/mnt/photo/P02.jpg',P02.astype(np.uint8))
    L02=LeptonCaptureORG()
    for n in range(5):
        L02 += LeptonCaptureORG()
    L02 = L02*0.2
    cv.imwrite('/mnt/photo/L02.jpg',L02.astype(np.uint8))
    print u'電子ライター3をONにしてください'
    raw_input()
    P03=PiCamCaptureORG()
    cv.imwrite('/mnt/photo/P03.jpg',P03.astype(np.uint8))
    L03=LeptonCaptureORG()
    for n in range(5):
        L03 += LeptonCaptureORG()
    L03 = L03*0.2
    cv.imwrite('/mnt/photo/L03.jpg',L03.astype(np.uint8))
  
    # 点熱源
    LeptonImg = cv.subtract(L01, L0)+cv.subtract(L02, L0)+cv.subtract(L03, L0)
    #
    cv.imwrite('/mnt/photo/LeptonImg0123.jpg',LeptonImg.astype(np.uint8))

    print u'`photoshopで、L01.jpg, L02.jpg, L03.jpgの電子ライターの位置座標を読み取る。'
    print u'`photoshopで、P01.jpg, P02.jpg, P03.jpgの電子ライターの位置座標を読み取る。'
    print u'Affine.ipynbを開いて、pts1、pts2に位置座標を入力して実行する。'
    print u'Lepton ImageとPiCam Imageを重ね合わせる、M.npyファイルが作られる。'
    return


    
#レーザーが写る範囲
servoAngleRange=np.zeros((2,2))

# レーザーが写っている範囲を探す
def getLaserServoLimit():
    global servoAngleRange
    laserCtl(30)
    laserServoCtl(LaserSERVO_ORG)
    smax=500
    smin=250
    # 真ん中付近から2度づつ回転させて、レーザーが見えなくなる角度を探す
    for x in range(375,smin,-2):    
        angle=[x,375]
        # [smin,smin]
        laserServoCtl(angle)
        laserCtl(30)
        p0=PiCamCaptureORG().astype(np.uint8)
        cv.imwrite('/mnt/photo/p0%d.jpg' % x,p0)
        lp=getLaserPoint()
        if lp is None:
            print('[smin,smin] Error')
            sXmin = x+2
            break
            
    for x in range(375,smax,2):    
        angle=[x,375]
        # [smin,smin]
        laserServoCtl(angle)
        laserCtl(30)
        p0=PiCamCaptureORG().astype(np.uint8)
        cv.imwrite('/mnt/photo/p1%d.jpg' % x,p0)
        lp=getLaserPoint()
        if lp is None:
            print('[smin,smin] Error')
            sXmax = x-2
            break
            
    for y in range(375,smin,-2):    
        angle=[375,y]
        # [smin,smin]
        laserServoCtl(angle)
        laserCtl(30)
        p0=PiCamCaptureORG().astype(np.uint8)
        cv.imwrite('/mnt/photo/p2%d.jpg' % x,p0)
        lp=getLaserPoint()
        if lp is None:
            print('[smin,smin] Error')
            sYmin = y+2
            break
            
    for y in range(375,smax,2):    
        angle=[375,y]
        # [smin,smin]
        laserServoCtl(angle)
        laserCtl(30)
        p0=PiCamCaptureORG().astype(np.uint8)
        cv.imwrite('/mnt/photo/p3%d.jpg' % x,p0)
        lp=getLaserPoint()
        if lp is None:
            print('[smin,smin] Error')
            sYmax = y-2
            break

    servoAngleRange[0][0]=sXmin
    servoAngleRange[0][1]=sXmax
    servoAngleRange[1][0]=sYmin
    servoAngleRange[1][1]=sYmax
    
    
# Main Loop
laserServoCtl(LaserSERVO_ORG) # 3mWレーザーを原点に戻す
sensorServoCtl(SensorSERVO_ORG) # センサーサーボを原点に戻す

dispSW = True    # X11 windowに表示するかしないか
verbosSW = False # 情報出力するかしないか
calSW = True     # キャリブレーションするかしないか
testSW = False   # 精密位置合わせをするかしないか
verifiSW = False   # 精密位置合わせ確認
relearnSW = False
loadORGSW = False
subSW = True
leptonCalibSW = False # Lepton ImageとPiCam imageとのキャリブレーション　M_Affineマトリックスを作成する

# Global
# cx,cy,rx,rymrw,rh
Target_Resion=np.zeros(26)

argv=sys.argv
argc = len(argv)
i = 0
if argc > 1:
    for av in argv:
        if av == '-nd': # X11表示をしない
            dispSW = False
        if av == '-v': # 情報表示する
            verbosSW = True
        if av == '-nv':# 情報表示しない
            verbosSW = False
        if av == '-d': # X11表示をする
            dispSW = True
        if av == '-nc': # レーザーのキャリブレーションをしない
            calSW = False
        if av == '-c': # レーザーのキャアリブレーションをする
            calSW = True
        if av == '-t': # レーザーの精密キャリブレーションをする
            testSW = True
        if av == '-nt': # レーザーの精密キャリブレーションをしない
            testSW = False
        if av == '-V':  # レーザーの精密キャリブレーションの確認
            verifiSW = True
        if av == '-a': # レーザーの精密キャリブレーションの追加学習
            relearnSW = True
        if av == '-thresh': # レーザーの精密キャリブレーションの閾値
            Thresh=float(argv[i+1])
            print u'Thresh=',Thresh
        if av == '-O': # レーザーの原点読込
            loadORGSW = True
        if av == '-ns': 
            subSW = False
        i += 1
            
#レーザーOFF
laserCtl(0)

#レーザーが検出できる限界角度を探す
# [smin,smin] [smin,smax][smax,smin][smax,smax]
# サーボ回転限界内でレーザーは検出できているか？
# 検出できていなければ、回転限界を狭める

#getLaserServoLimit()


# LeptonとPiCamの記録
# 80X60と160X120画像
if leptonCalibSW==True:
    LeptonCalibImg()


# getCompImg(LeptonImg,PiCamImg)

# レーザーのキャアリブレーションをする
if calSW == True:
    if loadORGSW==True:
        LaserSERVO_ORG=np.load('LaserSERVO_ORG.npy')
    laserCaliblationAxis(0)
    laserCaliblationAxis(1)

# レーザーのキャリブレーションをしない
# read caliblation file
readCalibFile()
#
#
#print u'LaserSERVO_ORG=',LaserSERVO_ORG
if loadORGSW==True:
    LaserSERVO_ORG=np.load('LaserSERVO_ORG.npy')
laserServoCtl(LaserSERVO_ORG)
#LaserSERVO_ORG=adjustLaser([320,240])
LaserSERVO_ORG=adjustLaser(getLeptonCenter())
np.save('LaserSERVO_ORG.npy',LaserSERVO_ORG)
if testSW == True:
    # 精密な位置合わせを行う
    precTable= precisionLaser()
    # precisionTableが使えるかどうかの判定
    precisionTableState=True 
# レーザーの精密キャリブレーションを読み込む
else:
    precTable=np.load('precImg.npy')
    precisionTableState=True 

# 確認
if verifiSW==True:
    Verification()

#レーザーOFF
laserCtl(0)

# 正面に向ける
laserServoCtl(LaserSERVO_ORG)
sensorServoCtl(SensorSERVO_ORG)

nAttack=0
startTime=time.time()

laserCtl(0)
while True:
    cx = cy = 0
    rrx = rry = rw = rh = 0

    # get Lepton image
    # thread
    thLepton = LeptonCaptureThread()
    thLepton.setDaemon(True)
    thLepton.start()

    # get PiCam image
    # thread
    
    thLepton.join()
    LeptonImg=thLepton.get_value()

    if LeptonImg is None:
        print 'LeptonImg',LeptonImg
        continue

    # calc max and min
    min, max, min_loc, max_loc = cv.minMaxLoc(LeptonImg)

    if verbosSW == True:
        print 'mainLoop max='+str(max)+' min='+str(min),\
        'max_loc='+str(max_loc)+' min_loc='+str(min_loc)
    if max < 10.0:
        laserCtl(0)

    # binary image
    ret,thresh = cv.threshold(LeptonImg,80,255,0)

    if thresh is None:
        print u"thresh:", thresh 
        continue

    # calc blob
    label = cv.connectedComponentsWithStats(thresh)
    n = label[0] - 1

    if n > 1:
        data = np.delete(label[2], 0, 0)
        center = np.delete(label[3], 0, 0)
         
        #print u"ブロブの個数:", n
        #print u"各ブロブの外接矩形の左上x座標", data[:,0]
        #print u"各ブロブの外接矩形の左上y座標", data[:,1]
        #print u"各ブロブの外接矩形の幅", data[:,2]
        #print u"各ブロブの外接矩形の高さ", data[:,3]
        #print u"各ブロブの面積", data[:,4]
        #print u"各ブロブの中心座標:\n",center
        
        imax = selectBlob(label) 
        #print u"------- imax =",imax,' n= ',n

        rx=data[imax,0] # 外接矩形の最小x
        ry=data[imax,1] # 外接矩形の最小y
        rw=data[imax,2] # 外接矩形の幅
        rh=data[imax,3] # 外接矩形の高さ
        cx=int(center[imax][0]) # 外接矩形の中心 x
        cy=int(center[imax][1]) # 外接矩形の中心 y

    # 異常値の排除
    if rw > 50 or rh > 50:
        continue

    if testSW == True:
        #    testSW ON
        laserCtl(50)
        # RECT  (240, 320, 3)
        center = np.zeros((2,2))
        center[0][0]=LeptonImg.shape[1]/2 # 外接矩形の中心 y
        center[0][1]=LeptonImg.shape[0]/2 # 外接矩形の中心 x
        cx=int(center[0][0]) # 外接矩形の中心 y
        cy=int(center[0][1]) # 外接矩形の中心 y
        rx=cx-50 # 外接矩形の最小x
        ry=cy-50 # 外接矩形の最小y
        rw=100 # 外接矩形の幅
        rh=100 # 外接矩形の高さ

    CanvasImg = cv.cvtColor(LeptonImg,cv.COLOR_GRAY2BGR)

    PiCamImg=PiCamCapture()
    
    if PiCamImg is None:
        print 'PiCamImg',PiCamImg
        continue

    # make composite image
    CompImg = cv.addWeighted(PiCamImg, 1.0, CanvasImg, 0.8, 0.8)
    cv.imwrite('/mnt/photo/CompImg.jpg',CompImg.astype(np.uint8))

    # add maker
    if n > 1:
        cv.circle(CompImg, (int(cx),int(cy)),\
                                10,  (0, 0, 255), 1)
        cv.rectangle(CompImg,(int(cx-rw),int(cy-rh)),\
                            (int(cx+rw),int(cy+rh)),(0,255,0),1)

        if verbosSW == True:
            print u'mainLoop center ',cx,' ',cy,'rect min',\
                    rx,' ',ry,'rect max',rx+rw,' ',ry+rh

        currentTime=time.time()
        #print u'currentTime-startTime=',currentTime-startTime
        # wait 1min fo stable
        if currentTime - startTime < 10:
            if verbosSW==True:
                print u'Warming Up Now'
            continue
        Target_Resion=[cx,cy,rx,ry,rw,rh]
        
        # laserAttack thread
        # thread
        # レーザーで攻撃する
        if verbosSW == True:
            print u'max=',max,'min=',min,'rw=',rw,'rh=',rh

    
        laserAttack()
        nAttack += 1

    # サーマル画像の書き出し
    cv.imwrite('img.jpg', CompImg)

    # フレーム表示
    if dispSW == True:
        cv.imshow('comp image',CompImg)  
        #cv.moveWindow('comp image',640,0)
        key = cv.waitKey(INTERVAL)
        if key == ESC_KEY:
            cv.destroyAllWindows()   
            sys.exit()
        elif key == ord('s'):
            cv.imwrite('/mnt/photo/thermal_simg.jpg', CompImg)
            cv.destroyAllWindows()
            sys.exit()


