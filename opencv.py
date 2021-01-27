#!/usr/bin/python3
import numpy as np
import cv2
import math
import ht301
import utils
import time

draw_temp = True

cap = ht301.HT301()
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while(True):
    ret, frame = cap.read()

    info, lut = cap.info()
    frame = frame.astype(np.float32)

    # Sketchy auto-exposure
    frame -= frame.min()
    frame /= frame.max()
    frame = (np.clip(frame, 0, 1)*255).astype(np.uint8)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

    if draw_temp:
        utils.drawTemperature(frame, info['Tmin_point'], info['Tmin_C'], (55,0,0))
        utils.drawTemperature(frame, info['Tmax_point'], info['Tmax_C'], (0,0,85))
        utils.drawTemperature(frame, info['Tcenter_point'], info['Tcenter_C'], (0,255,255))

    #frame2 = frame2.reshape(288, 384)
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('u'):
        cap.calibrate()
    if key == ord('s'):
        cv2.imwrite(time.strftime("%Y-%m-%d_%H:%M:%S") + '.png', frame)

cap.release()
cv2.destroyAllWindows()
