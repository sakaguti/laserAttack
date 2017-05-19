#!/usr/bin/python
# Simple demo of of the PCA9685 PWM servo/LED controller library.
# This will move channel 0 from min to max position repeatedly.
# Author: Tony DiCola
# License: Public Domain
from __future__ import division
import sys
import signal
import RPi.GPIO as GPIO

def receive_signal(signum, stack):
    print 'Received:', signum
    laser.ChangeDutyCycle(0)
    laser.stop(0.0);
    sys.exit()

signal.signal(signal.SIGHUP, receive_signal)
signal.signal(signal.SIGINT, receive_signal)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
gp_out=27
GPIO.setup(gp_out, GPIO.OUT)
laser = GPIO.PWM(gp_out,1000)

laser.start(0.0);
args=sys.argv
dc = int(args[1])
while True:
    laser.ChangeDutyCycle(dc)
