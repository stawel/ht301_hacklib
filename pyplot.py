#!/usr/bin/python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ht301_hacklib
import utils
import time
import sys

fps = 40
exposure = {'auto': True,
            'auto_type': 'ends',  # 'center' or 'ends'
            'T_min': 0.,
            'T_max': 50.,
            'T_margin': 2.0,
}
draw_temp = True

# choose the camera class
camera:ht301_hacklib.Camera
if sys.argv[-1].endswith('.npy'):
    camera = ht301_hacklib.CameraEmulator(sys.argv[-1])
else:
    camera = ht301_hacklib.Camera()



#see https://matplotlib.org/tutorials/colors/colormaps.html
cmaps_idx = 1
cmaps = ['inferno', 'coolwarm', 'cividis', 'jet', 'nipy_spectral', 'binary', 'gray', 'tab10']

matplotlib.rcParams['toolbar'] = 'None'

# temporary fake frame
lut_frame = frame = np.full((camera.height, camera.width), 25.)
info = {}
lut = None # will be defined later

fig = plt.figure()

try:
    fig.canvas.set_window_title("Thermal Camera")
except:
    # does not work on windows
    pass

ax = plt.gca()
im = ax.imshow(lut_frame, cmap=cmaps[cmaps_idx])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)

annotations = utils.Annotations(ax, patches)
temp_annotations =  {
    'std': {
        'Tmin': 'lightblue',
        'Tmax': 'red',
        'Tcenter': 'yellow'
        },
    'user': {}
}

# Add the patch to the Axes
roi = ((0,0),(0,0))


paused = False
update_colormap = True
diff = { 'enabled': False,
         'annotation_enabled': False,
         'frame': np.zeros(frame.shape)
}

import csv
from datetime import datetime #to easily get miliseconds
csv_filename = None
def log_annotations_to_csv(annotation_frame):
    anns_data = []
    for type in ['std', 'user']:
        for ann_name in temp_annotations[type]:
            pos = annotations.get_pos(ann_name)
            val = round(annotations.get_val(ann_name, annotation_frame), 2)
            anns_data += [pos[0], pos[1], val]  # store each position and value
    if csv_filename is not None:
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now()] + anns_data)

import csv
from datetime import datetime #to easily get miliseconds
csv_filename = None
def log_annotations_to_csv(annotation_frame):
    anns_data = []
    for type in ['std', 'user']:
        for ann_name in temp_annotations[type]:
            pos = annotations.get_pos(ann_name)
            val = round(annotations.get_val(ann_name, annotation_frame), 2)
            anns_data += [pos[0], pos[1], val]  # store each position and value
    if csv_filename is not None:
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now()] + anns_data)


def animate_func(i):
    global lut, frame, info, paused, update_colormap, exposure, im, diff, lut_frame
    ret, frame = camera.read()
    if not paused:
        info, lut = camera.info()
        lut_frame = lut[frame]

        if diff['enabled']: show_frame = lut_frame - diff['frame']
        else:               show_frame = lut_frame
        if diff['annotation_enabled']:
                        annotation_frame = lut_frame - diff['frame']
        else:           annotation_frame = lut_frame

        im.set_array(show_frame)

        annotations.update(temp_annotations, annotation_frame, draw_temp)

        if exposure['auto']:
            update_colormap = utils.autoExposure(update_colormap, exposure, show_frame)

        log_annotations_to_csv(annotation_frame)

        if update_colormap:
            im.set_clim(exposure['T_min'], exposure['T_max'])
            fig.canvas.resize_event()  #force update all, even with blit=True
            update_colormap = False
            return []

    return [im] + annotations.get()

def print_help():
    print('''keys:
    'h'      - help
    'q'      - quit
    ' '      - pause, resume
    'd'      - set diff
    'x','c'  - enable/disable diff, enable/disable annotation diff
    'f'      - full screen
    'u'      - calibrate
    't'      - draw min, max, center temperature
    'e'      - remove user temperature annotations
    'w'      - save to file date.png
    'r'      - save raw data to file date.npy
    'v'      - record annotations data to file date.csv
    ',', '.' - change color map
    'a', 'z' - auto exposure on/off, auto exposure type
    'k', 'l' - set the thermal range to normal/high (supported by T2S+/T2L)  
    left, right, up, down - set exposure limits

mouse:
    left  button - add Region Of Interest (ROI)
    right button - add user temperature annotation
''')

FILE_NAME_FORMAT = "%Y-%m-%d_%H-%M-%S"

#keyboard
def press(event):
    global paused, exposure, update_colormap, cmaps_idx, draw_temp, temp_extra_annotations, csv_filename
    global lut_frame, lut, frame, diff, annotations, roi
    if event.key == 'h': print_help()
    if event.key == ' ': paused ^= True; print('paused:', paused)
    if event.key == 'd': diff['frame'] = lut_frame; diff['annotation_enabled'] = diff['enabled'] = True; print('set   diff')
    if event.key == 'x': diff['enabled'] ^= True; print('enable diff:', diff['enabled'])
    if event.key == 'c': diff['annotation_enabled'] ^= True; print('enable annotation diff:', diff['annotation_enabled'])
    if event.key == 't': draw_temp ^= True; print('draw temp:', draw_temp)
    if event.key == 'e':
        print('removing user annotations: ', len(temp_annotations['user']))
        annotations.remove(temp_annotations['user'])
    if event.key == 'u': print('calibrate'); camera.calibrate()
    if event.key == 'a': exposure['auto'] ^= True; print('auto exposure:', exposure['auto'], ', type:', exposure['auto_type'])
    if event.key == 'z':
        types = ['center', 'ends']
        exposure['auto_type'] = types[types.index(exposure['auto_type'])-1]
        print('auto exposure:', exposure['auto'], ', type:', exposure['auto_type'])
    if event.key == 'w':
        filename = time.strftime(FILE_NAME_FORMAT) + '.png'
        plt.savefig(filename)
        print('saved to:', filename)
    if event.key == 'r':
        filename = time.strftime(FILE_NAME_FORMAT) + '.npy'
        np.save(filename, camera.frame_raw_u16.reshape(camera.height+4, camera.width))
        print('saved to:', filename)
    if event.key == 'v':
        if csv_filename is None:
            csv_filename = time.strftime(FILE_NAME_FORMAT) + '.csv'
            with open(csv_filename, 'w', newline='') as f:
                header = ["time"]
                header += [f'{a} {x}' for a in temp_annotations['std'].keys() for x in ['x', 'y', 'val']] #t, tmin x, tmin y, tmin val, etc
                header += [f'Point{i} {x}' for i, key in enumerate(temp_annotations['user'].keys()) for x in ['x', 'y', 'val']]
                csv.writer(f).writerow(header)
            print('Annotation recording started in:', csv_filename)
        else:
            print('Annotation recording  saved  in:', csv_filename)
            csv_filename = None
        
    if event.key in [',', '.']:
        if event.key == '.': cmaps_idx= (cmaps_idx + 1) % len(cmaps)
        else:                cmaps_idx= (cmaps_idx - 1) % len(cmaps)
        print('color map:', cmaps[cmaps_idx])
        im.set_cmap(cmaps[cmaps_idx])
        update_colormap = True
    if event.key in ['k', 'l']:
        if event.key == 'k':
            camera.temperature_range_normal()
        else:
            camera.temperature_range_high()
        camera.calibrate()
    if event.key in ['left', 'right', 'up', 'down']:
        exposure['auto'] = False
        T_cent = int((exposure['T_min'] + exposure['T_max'])/2)
        d = int(exposure['T_max'] - T_cent)
        if event.key == 'up':    T_cent += exposure['T_margin']/2
        if event.key == 'down':  T_cent -= exposure['T_margin']/2
        if event.key == 'left':  d -= exposure['T_margin']/2
        if event.key == 'right': d += exposure['T_margin']/2
        d = max(d, exposure['T_margin'])
        exposure['T_min'] = T_cent - d
        exposure['T_max'] = T_cent + d
        print('auto exposure off, T_min:', exposure['T_min'], 'T_cent:', T_cent, 'T_max:', exposure['T_max'])
        update_colormap = True

mouse_action_pos = (0,0)
mouse_action = None
def onclick(event):
    global mouse_action, mouse_action_pos
    if event.inaxes == ax:
        pos = (int(event.xdata), int(event.ydata))
        if event.button == MouseButton.RIGHT:
            print('add user temperature annotation at pos:', pos)
            temp_annotations['user'][pos] = 'white'
        if event.button == MouseButton.LEFT:
            if utils.inRoi(annotations.roi, pos, frame.shape):
                mouse_action = 'move_roi'
                mouse_action_pos = (annotations.roi[0][0] - pos[0], annotations.roi[0][1] - pos[1])
            else:
                mouse_action = 'create_roi'
                mouse_action_pos = pos
                annotations.set_roi((pos, (0,0)))

def onmotion(event):
    global mouse_action, mouse_action_pos, roi
    if event.inaxes == ax and event.button == MouseButton.LEFT:
        pos = (int(event.xdata), int(event.ydata))
        if mouse_action == 'create_roi':
            w,h = pos[0] - mouse_action_pos[0], pos[1] - mouse_action_pos[1]
            roi = (mouse_action_pos, (w,h))
            annotations.set_roi(roi)
        if mouse_action == 'move_roi':
            roi = ((pos[0] + mouse_action_pos[0], pos[1] + mouse_action_pos[1]), annotations.roi[1])
            annotations.set_roi(roi)


anim = animation.FuncAnimation(fig, animate_func, interval = 1000 / fps, blit=True)
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', onmotion)
fig.canvas.mpl_connect('key_press_event', press)

print_help()
plt.show()
camera.release()
