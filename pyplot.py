#!/usr/bin/python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ht301
import utils
import time

fps = 25
T_margin = 2.0
auto_exposure = True
auto_exposure_type = 'ends'  # 'center' or 'ends'
T_min, T_max = 0., 50.
draw_temp = True

#see https://matplotlib.org/tutorials/colors/colormaps.html
cmaps_idx = 1
cmaps = ['inferno', 'coolwarm', 'cividis', 'jet', 'nipy_spectral', 'binary', 'gray', 'tab10']

cap = ht301.HT301()
ret, frame = cap.read()
info, lut = cap.info()

fig = plt.figure()
ax = plt.gca()
im = ax.imshow(lut[frame],cmap=cmaps[cmaps_idx])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
astyle = dict(s='', xy=(0, 0), xytext=(0, 0), textcoords='offset pixels', arrowprops=dict(facecolor='black', arrowstyle="->"))
a_min = ax.annotate(**astyle, bbox=dict(boxstyle='square', fc='lightblue', alpha=0.3, lw=0))
a_max = ax.annotate(**astyle, bbox=dict(boxstyle='square', fc='red' , alpha=0.3, lw=0))
a_cen = ax.annotate(**astyle, bbox=dict(boxstyle='square', fc='yellow', alpha=0.3, lw=0))

paused = False
update_colormap = True

def animate_func(i):
    global paused, update_colormap, T_min, T_max, im
    ret, frame = cap.read()
    if not paused:
        info, lut = cap.info()
        lut_frame = lut[frame]
        im.set_array(lut_frame)

        utils.setAnnotate(a_min, frame, info, 'Tmin', draw_temp)
        utils.setAnnotate(a_max, frame, info, 'Tmax', draw_temp)
        utils.setAnnotate(a_cen, frame, info, 'Tcenter', draw_temp)

        if auto_exposure:
            update_colormap, T_min, T_max = utils.autoExposure(update_colormap, T_min, T_max, T_margin, auto_exposure_type, lut_frame)

        if update_colormap:
            im.set_clim(T_min, T_max)
            fig.canvas.resize_event()  #force update all, even with blit=True
            update_colormap = False
            return []
        return [im, a_min, a_max, a_cen]
    return [im, a_min, a_max, a_cen]

anim = animation.FuncAnimation(fig, animate_func, interval = 1000 / fps, blit=True)

def print_help():
    print('''keys:
    'h'      - help
    ' '      - pause, resume
    'u'      - calibrate
    't'      - draw min, max, center temperature
    'a', 'z' - auto exposure on/off, auto exposure type
    'w'      - save to file date.png
    ',', '.' - change color map
    left, right, up, down - set exposure limits
''')

#keyboard
def press(event):
    global paused, auto_exposure, auto_exposure_type, update_colormap, cmaps_idx, draw_temp, T_min, T_max
    if event.key == 'h': print_help()
    if event.key == ' ': paused ^= True; print('paused:', paused)
    if event.key == 't': draw_temp ^= True; print('draw temp:', draw_temp)
    if event.key == 'u': print('calibrate'); cap.calibrate()
    if event.key == 'a': auto_exposure ^= True; print('auto exposure:', auto_exposure, ', type:', auto_exposure_type)
    if event.key == 'z':
        if auto_exposure_type == 'center':
            auto_exposure_type = 'ends'
        else:
            auto_exposure_type = 'center'
        print('auto exposure:', auto_exposure, ', type:', auto_exposure_type)
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
    if event.key in ['left', 'right', 'up', 'down']:
        auto_exposure = False
        T_cent = int((T_min + T_max)/2)
        d = int(T_max - T_cent)
        if event.key == 'up':    T_cent += T_margin/2
        if event.key == 'down':  T_cent -= T_margin/2
        if event.key == 'left':  d -= T_margin/2
        if event.key == 'right': d += T_margin/2
        d = max(d, T_margin)
        T_min, T_max = T_cent - d, T_cent + d
        print('auto exposure off, T_min:', T_min, 'T_cent:', T_cent, 'T_max:', T_max)
        update_colormap = True

fig.canvas.mpl_connect('key_press_event', press)

print_help()
plt.show()
cap.release()
