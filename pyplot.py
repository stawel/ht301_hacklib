#!/usr/bin/python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ht301

video_dev = 2
fps = 25
T_margin = 2.0

cap = ht301.HT301(video_dev)

fig = plt.figure()

ret, frame = cap.read()
info, lut = cap.info()

ax = plt.gca()
im = ax.imshow(lut[frame],cmap='magma')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)

tmin, tmax = 0., 0.
paused = False

def animate_func(i):
    global paused, tmin, tmax
    ret, frame = cap.read()
    if not paused:
        info, lut = cap.info()
        lut_frame = lut[frame]
        im.set_array(lut_frame)

        # Sketchy auto-exposure
        lmin, lmax = lut_frame.min(), lut_frame.max()
        refresh = False

        if tmin                > lmin: refresh, tmin = True, lmin-T_margin
        if tmin + 2 * T_margin < lmin: refresh, tmin = True, lmin-T_margin
        if tmax                < lmax: refresh, tmax = True, lmax+T_margin
        if tmax - 2 * T_margin > lmax: refresh, tmax = True, lmax+T_margin
        if refresh:
            im.set_clim(tmin, tmax)
            fig.canvas.resize_event()  #force update all, even with blit=True
            return []
        return [im]
    return []

anim = animation.FuncAnimation(fig, animate_func, interval = 1000 / fps, blit=True)

def press(event):
    global paused
    if event.key == ' ': paused ^= True
    if event.key == 'u': cap.calibrate()


fig.canvas.mpl_connect('key_press_event', press)

plt.show()
cap.release()
