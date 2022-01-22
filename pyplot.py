#!/usr/bin/python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ht301_hacklib
import utils
import time

fps = 40
T_margin = 2.0
auto_exposure = True
auto_exposure_type = 'ends'  # 'center' or 'ends'
T_min, T_max = 0., 50.
draw_temp = True

#see https://matplotlib.org/tutorials/colors/colormaps.html
cmaps_idx = 1
cmaps = ['inferno', 'coolwarm', 'cividis', 'jet', 'nipy_spectral', 'binary', 'gray', 'tab10']

cap = ht301_hacklib.HT301()
ret, frame = cap.read()
info, lut = cap.info()

fig = plt.figure()
fig.canvas.set_window_title('HT301')
ax = plt.gca()
im = ax.imshow(lut[frame],cmap=cmaps[cmaps_idx])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
astyle = dict(s='', xy=(0, 0), xytext=(0, 0), textcoords='offset pixels', arrowprops=dict(facecolor='black', arrowstyle="->"))

def get_ann(color):
    return ax.annotate(**astyle, bbox=dict(boxstyle='square', fc=color, alpha=0.3, lw=0))

temp_std_annotations =  {
    'Tmin': get_ann('lightblue'),
    'Tmax': get_ann('red'),
    'Tcenter': get_ann('yellow')
}
temp_extra_annotations = {}

# Add the patch to the Axes
roi_patch = ax.add_patch(patches.Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='black', facecolor='none'))
roi_patch.set_visible(False)

paused = False
update_colormap = True
enable_diff, enable_annotation_diff = False, False
lut_frame = diff_frame = np.zeros(frame.shape)

def animate_func(i):
    global paused, update_colormap, T_min, T_max, im, diff_frame, lut_frame, enable_diff, enable_annotation_diff
    ret, frame = cap.read()
    if not paused:
        info, lut = cap.info()
        lut_frame = lut[frame]

        if enable_diff: show_frame = lut_frame - diff_frame
        else:           show_frame = lut_frame
        if enable_annotation_diff:
                        annotation_frame = lut_frame - diff_frame
                        utils.updateInfo(info, annotation_frame)
        else:           annotation_frame = lut_frame
        if roi_patch.get_visible(): utils.updateInfo(info, annotation_frame, (roi_patch.xy[::-1], (roi_patch.get_height(), roi_patch.get_width())))

        im.set_array(show_frame)

        for name, annotation in temp_std_annotations.items():
            utils.setAnnotate(annotation, frame, info[name + '_point'], info[name+'_C'], draw_temp)

        for pos, annotation in temp_extra_annotations.items():
            utils.setAnnotate(annotation, frame, pos, annotation_frame[pos[1],pos[0]], True)

        if auto_exposure:
            update_colormap, T_min, T_max = utils.autoExposure(update_colormap, T_min, T_max, T_margin, auto_exposure_type, show_frame)

        if update_colormap:
            im.set_clim(T_min, T_max)
            fig.canvas.resize_event()  #force update all, even with blit=True
            update_colormap = False
            return []

    return [im, roi_patch] + list(temp_std_annotations.values()) + list(temp_extra_annotations.values())

def print_help():
    print('''keys:
    'h'      - help
    ' '      - pause, resume
    'd'      - set diff
    'x','c'  - enable/disable diff, enable/disable annotation diff
    'f'      - full screen
    'u'      - calibrate
    't'      - draw min, max, center temperature
    'e'      - remove extra annotations
    'a', 'z' - auto exposure on/off, auto exposure type
    'w'      - save to file date.png
    ',', '.' - change color map
    left, right, up, down - set exposure limits

mouse:
    left  button - add region of interest (ROI)
    right button - add extra temperature annotation
''')

#keyboard
def press(event):
    global paused, auto_exposure, auto_exposure_type, update_colormap, cmaps_idx, draw_temp, T_min, T_max, temp_extra_annotations
    global lut_frame, diff_frame, enable_diff, enable_annotation_diff
    if event.key == 'h': print_help()
    if event.key == ' ': paused ^= True; print('paused:', paused)
    if event.key == 'd': diff_frame = lut_frame; enable_annotation_diff = enable_diff = True; print('set   diff')
    if event.key == 'x': enable_diff ^= True; print('enable diff:', enable_diff)
    if event.key == 'c': enable_annotation_diff ^= True; print('enable annotation diff:', enable_annotation_diff)
    if event.key == 't': draw_temp ^= True; print('draw temp:', draw_temp)
    if event.key == 'e':
        print('removing extra annotations: ', len(temp_extra_annotations))
        for ann in temp_extra_annotations.values(): ann.remove()
        temp_extra_annotations = {}
    if event.key == 'u': print('calibrate'); cap.calibrate()
    if event.key == 'a': auto_exposure ^= True; print('auto exposure:', auto_exposure, ', type:', auto_exposure_type)
    if event.key == 'z':
        types = ['center', 'ends']
        auto_exposure_type = types[types.index(auto_exposure_type)-1]
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

mouse_action_pos = (0,0)
mouse_action = None
def onclick(event):
    global mouse_action, mouse_action_pos
    if event.inaxes == ax:
        pos = (int(event.xdata), int(event.ydata))
        if event.button == MouseButton.RIGHT:
            print('add extra annotation at pos:', pos)
            temp_extra_annotations[pos] = get_ann('white')
        if event.button == MouseButton.LEFT:
            if utils.inRoi(roi_patch, pos, frame.shape):
                mouse_action = 'move_roi'
                mouse_action_pos = (roi_patch.xy[0] - pos[0], roi_patch.xy[1] - pos[1])
            else:
                mouse_action = 'create_roi'
                roi_patch.xy = mouse_action_pos = pos
                roi_patch.set_visible(False)

def onmotion(event):
    global mouse_action, mouse_action_pos
    if event.inaxes == ax and event.button == MouseButton.LEFT:
        pos = (int(event.xdata), int(event.ydata))
        if mouse_action == 'create_roi':
            w,h = pos[0] - mouse_action_pos[0], pos[1] - mouse_action_pos[1]
            roi_patch.set_width(w)
            roi_patch.set_height(h)
            roi_patch.set_visible(w!=0 or h!=0)
        if mouse_action == 'move_roi':
            roi_patch.xy = (pos[0] + mouse_action_pos[0], pos[1] + mouse_action_pos[1])


anim = animation.FuncAnimation(fig, animate_func, interval = 1000 / fps, blit=True)
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', onmotion)
fig.canvas.mpl_connect('key_press_event', press)

print_help()
plt.show()
cap.release()
