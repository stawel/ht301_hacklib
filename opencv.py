#!/usr/bin/python3
import numpy as np
import cv2
import math
import ht301
import utils
import time


draw_temp = True

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
# Use raw mode
cap.set(cv2.CAP_PROP_ZOOM, 0x8004)
# Calibrate
#cap.set(cv2.CAP_PROP_ZOOM, 0x8000)

#cap.set(cv2.CAP_PROP_ZOOM, 0x8020)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)



while(True):
    ret, frame = cap.read()

    frame = frame.reshape(292,384,2) # 0: LSB. 1: MSB

    # Remove the four extra rows
    # Convert to uint16
    dt = np.dtype(('<u2', [('x', np.uint8, 2)]))
    frame = frame.view(dtype=dt)

    info, lut = ht301.info(frame)
    frame = frame.astype(np.float32)
    frame = frame[:288,...]

    # Sketchy auto-exposure
    frame -= frame.min()
    frame /= frame.max()
    frame = (np.clip(frame, 0, 1)*255).astype(np.uint8)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

    if draw_temp:
        utils.drawTemperature(frame, info['Tmin_point'], info['Tmin_C'], (55,0,0))
        utils.drawTemperature(frame, info['Tmax_point'], info['Tmax_C'], (0,0,85))
        utils.drawTemperature(frame, info['Tcenter_point'], info['Tcenter_C'], (0,255,255))

    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('u'):
        cap.set(cv2.CAP_PROP_ZOOM, 0x8000)
    if key == ord('s'):
        cv2.imwrite(time.strftime("%Y-%m-%d_%H:%M:%S") + '.png', frame)

cap.release()
cv2.destroyAllWindows()
