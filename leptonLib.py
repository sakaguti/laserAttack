#!/usr/bin/python
import cv2
import time
import numpy as np

def main():
    scl=5.5
    while True:
    	for png in exec_cmd('/home/pi/src/flirpi-mjpeg/leptgraypng'):
		if png is not None:
	        	png = np.fromstring(png, dtype=np.uint8)
       	 		LeptonImg = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)

			'''
        		cv2.imshow('o',LeptonImg)
        		if cv2.waitKey(1) ==27:
          			exit(0)	
			'''

			####
			max = 0
        		min = 100000
			th=8300
			w=LeptonImg.shape[0]
			h=LeptonImg.shape[1]
        		for j in range(h):
                		for i in range(w):
					# F to deg
                                	#LeptonImg[i][j]=int((float(LeptonImg[i][j])/100.0-35.0)*1.8)
                        		if LeptonImg[i][j] > max:
                                		max = LeptonImg[i][j]
                        		if LeptonImg[i][j] < min:
                                		min = LeptonImg[i][j]

			th = (max-min)/2+min
        		print 'max='+str(max)+' min='+str(min)+' th='+str(th)
        		for j in range(h):
                		for i in range(w):
                        		LeptonImg[i][j] = int(float(LeptonImg[i][j]-min)/float(max-min))*128
					'''
                        		if LeptonImg[i][j] <= th:
                                		LeptonImg[i][j]=0
					else:
                                		LeptonImg[i][j]=128
					'''
			####

        		LeptonRGB = cv2.cvtColor(LeptonImg,cv2.COLOR_GRAY2RGB)
        		LeptonRGB = cv2.resize(LeptonRGB,None,fx=scl, fy=scl, interpolation=cv2.INTER_CUBIC)

			'''
        		cv2.imshow('i',LeptonRGB)
        		if cv2.waitKey(1) ==27:
          			exit(0)	
			'''

			time.sleep(0.5)
	

def exec_cmd(cmd):
    from subprocess import Popen, PIPE
    p = Popen(cmd, stdout=PIPE)
    out = p.communicate()
    return out

if __name__ == '__main__':
    main()
