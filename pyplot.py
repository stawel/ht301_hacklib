#!/usr/bin/python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ht301
import time

video_dev = 2
fps = 25

T_margin = 2.0
auto_exposure = True
T_min, T_max = 0., 50.

#see https://matplotlib.org/tutorials/colors/colormaps.html
cmaps_idx = 0
cmaps = ['inferno', 'coolwarm', 'cividis', 'jet', 'nipy_spectral', 'binary', 'gray', 'tab10']

cap = ht301.HT301(video_dev)

fig = plt.figure()

ret, frame = cap.read()
info, lut = cap.info()

ax = plt.gca()
im = ax.imshow(lut[frame],cmap=cmaps[cmaps_idx])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)

paused = False
update_colormap = True

def animate_func(i):
    global paused, update_colormap, T_min, T_max
    ret, frame = cap.read()
    if not paused:
        info, lut = cap.info()
        lut_frame = lut[frame]
        im.set_array(lut_frame)

        # Sketchy auto-exposure
        lmin, lmax = lut_frame.min(), lut_frame.max()
        if auto_exposure:
            if T_min                > lmin: update_colormap, T_min = True, lmin-T_margin
            if T_min + 2 * T_margin < lmin: update_colormap, T_min = True, lmin-T_margin
            if T_max                < lmax: update_colormap, T_max = True, lmax+T_margin
            if T_max - 2 * T_margin > lmax: update_colormap, T_max = True, lmax+T_margin
        if update_colormap:
            im.set_clim(T_min, T_max)
            fig.canvas.resize_event()  #force update all, even with blit=True
            update_colormap = False
            return []
        return [im]
    return []

anim = animation.FuncAnimation(fig, animate_func, interval = 1000 / fps, blit=True)

def press(event):
    global paused, auto_exposure, update_colormap, cmaps_idx
    if event.key == ' ': paused ^= True; print('paused:', paused)
    if event.key == 'a': auto_exposure ^= True; print('auto exposure:', auto_exposure)
    if event.key == 'u': print('calibrate'); cap.calibrate()
    if event.key == 'w':
        filename = time.strftime("%Y-%m-%d_%H:%M:%S") + '.png'
        plt.savefig(filename)
        print('saved to:', filename)
    if event.key in [',', '.']:
        if event.key == '.': cmaps_idx= (cmaps_idx + 1) % len(cmaps)
        else:                cmaps_idx= (cmaps_idx - 1) % len(cmaps)
        print('color map:', cmaps[cmaps_idx])
        im.set_cmap(cmaps[cmaps_idx])
        update_colormap = True


fig.canvas.mpl_connect('key_press_event', press)

plt.show()
cap.release()
