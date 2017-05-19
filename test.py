#!/usr/bin/python
#coding: utf-8
import numpy as np
import sys 
import time 

from timeout import on_timeout
from subprocess import Popen, PIPE

def handler_func(msg):
    print msg 
    time.sleep(0.01)
    sub()


@on_timeout(limit=1, handler=handler_func, hint=u'retry')
def sub():
    cmd='/home/pi/src/LeptonModule/software/raspberrypi_capture/leptoncapture'
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    #temp,err = p.communicate()
    temp = []

    while True:
        # バッファから1行読み込む.
        line = p.stdout.readline()
        temp.append(line)
        #sys.stdout.write(line)
        #sys.stdout.write('\n')

        # バッファが空 + プロセス終了.
        if not line and p.poll() is not None:
                break

        '''
        if err:
                print u'Lepton Error',err

        if temp is None:
                print 'Lepton Error !!'
                continue
        else:
                break
        '''
    temp=temp[0].replace('\n','')
    #print u'temp ',temp
    temp = temp.split(' ')
    temp = temp[:-1]

    # type is list
    temp = map(lambda x: float(x), temp)
    pmax = np.max(temp)
    pmin = np.min(temp)

    print u'max=',pmax,' min=',pmin

while True:
    sub()
    time.sleep(0.1)
