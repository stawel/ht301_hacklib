#!/usr/bin/python3
import numpy as np
import cv2
import math
import ht301_hacklib
import utils
import time
from skimage.exposure import rescale_intensity, equalize_hist
draw_temp = True

# cap = ht301_hacklib.HT301()
cap = ht301_hacklib.T2SPLUS()
cv2.namedWindow("HT301", cv2.WINDOW_NORMAL)

def increase_luminance_contrast(frame):
    lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    frame = enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return frame

while(True):
    ret, frame = cap.read()

    info, lut = cap.info()
    frame = frame.astype(np.float32)

    # Sketchy auto-exposure
    frame = rescale_intensity(equalize_hist(frame), in_range='image', out_range=(0,255)).astype(np.uint8)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
    
    frame = increase_luminance_contrast(frame)

    if draw_temp:
        # utils.drawTemperature(frame, info['Tmin_point'], info['Tmin_C'], (55,0,0))
        utils.drawTemperature(frame, info['Tmax_point'], info['Tmax_C'], (255,255,255))
        utils.drawTemperature(frame, info['Tcenter_point'], info['Tcenter_C'], (255,255,255))


    cv2.imshow('HT301',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('u'):
        cap.calibrate()
    if key == ord('z'):
        cap.temperature_range_normal()
        cap.calibrate()
    if key == ord('x'):
        cap.temperature_range_high()
        cap.calibrate()
    if key == ord('s'):
        cv2.imwrite(time.strftime("%Y-%m-%d_%H:%M:%S") + '.png', frame)

cap.release()
cv2.destroyAllWindows()
