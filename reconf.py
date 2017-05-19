#!/usr/bin/python
#coding: utf-8
import numpy as np

precTable=np.load('precTable.npy')
print precTable[0][0][0]
print precTable[0][0][1]
print precTable[0][0][2]
precImg=np.load('precImg.npy')

for y in range(precImg.shape[0]): 
    for x in range(precImg.shape[1]): 
	if precImg[y][x][0] == precTable[0][0][0]:
		if precImg[y][x][1] == precTable[0][0][1]:
			print u'[ ',x,y,'] table =',precTable[0][0][0],precTable[0][0][1], \
			'img   =',precImg[y][x][0],precImg[y][x][1] 	
			break
