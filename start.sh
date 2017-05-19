#!/bin/sh
cd /home/pi/src/opencv
./app.js &
/home/pi/src/opencv/laserAttack3.py -nd -nc -nt -v
#/home/pi/src/opencv/laserAttack.py  -nd  -v
