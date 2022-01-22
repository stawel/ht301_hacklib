import numpy as np
import cv2




def drawTemperature(img, point, T, color = (0,0,0)):
    d1, d2 = 2, 5
    dsize = 1
    font = cv2.FONT_HERSHEY_PLAIN
    (x, y) = point
    t = '%.2fC' % T
    cv2.line(img,(x+d1, y),(x+d2,y),color, dsize)
    cv2.line(img,(x-d1, y),(x-d2,y),color, dsize)
    cv2.line(img,(x, y+d1),(x,y+d2),color, dsize)
    cv2.line(img,(x, y-d1),(x,y-d2),color, dsize)

    text_size = cv2.getTextSize(t, font, 1, dsize)[0]
    tx, ty = x+d1, y+d1+text_size[1]
    if tx + text_size[0] > img.shape[1]: tx = x-d1-text_size[0]
    if ty                > img.shape[0]: ty = y-d1

    cv2.putText(img, t, (tx,ty), font, 1, color, dsize, cv2.LINE_8)

def setAnnotate(a, img, pos, value, visible):
    (x,y) = pos
    a.xy  = pos
    a.set_text('%.2f$^\circ$C' % value)
    a.set_visible(visible)
    tx,ty = 20, 15
    if x > img.shape[1]-50: tx = -80
    if y < 30: ty = -15
    a.xyann = (tx, ty)

def autoExposure(update, T_min, T_max, T_margin, auto_exposure_type, frame):
    # Sketchy auto-exposure
    lmin, lmax = frame.min(), frame.max()
    if auto_exposure_type == 'center':
        T_cent = int((T_min+T_max)/2)
        d = int(max(T_cent-lmin, lmax-T_cent, 0) + T_margin)
        if lmin < T_min or T_max < lmax or (T_min + 2 * T_margin < lmin and T_max - 2 * T_margin > lmax):
#            print('d:',d, 'lmin:', lmin, 'lmax:', lmax)
            update = True
            T_min, T_max = T_cent - d, T_cent + d
#            print('T_min:', T_min, 'T_cent:', T_cent, 'T_max:', T_max)
    if auto_exposure_type == 'ends':
        if T_min                > lmin: update, T_min = True, lmin-T_margin
        if T_min + 2 * T_margin < lmin: update, T_min = True, lmin-T_margin
        if T_max                < lmax: update, T_max = True, lmax+T_margin
        if T_max - 2 * T_margin > lmax: update, T_max = True, lmax+T_margin

    return update, T_min, T_max

def correctRoi(roi, shape):
    if roi is None:
        roi = ((0,0), shape)
    ((x,y),(w,h)) = roi
    x1,x2 = max(0, min(x,x+w)), max(0, x,x+w)
    y1,y2 = max(0, min(y,y+h)), max(0, y,y+h)
    if x1 == x2: x2 += 1
    if y1 == y2: y2 += 1
    return ((x1,y1),(x2,y2))


def inRoi(roi_patch, point, shape):
    result = False
    roi = (roi_patch.xy, (roi_patch.get_width(), roi_patch.get_height()))
    ((x1,y1),(x2,y2)) = correctRoi(roi, shape)
    if x1 < point[0] and point[0] < x2:
        if y1 < point[1] and point[1] < y2:
            result = True
    return result

def updateInfo(info, frame, roi = None):
    ((x1,y1),(x2,y2)) = correctRoi(roi, frame.shape)
    roi_frame = frame[x1:x2,y1:y2]
    if roi_frame.size <= 0:
        x1,y1 = 0, 0
        roi_frame = frame
    pos = np.unravel_index(roi_frame.argmin(), roi_frame.shape)
    info['Tmin_C'] = roi_frame[pos]
    info['Tmin_point'] = (pos[1]+y1, pos[0]+x1)
    pos = np.unravel_index(roi_frame.argmax(), roi_frame.shape)
    info['Tmax_C'] = roi_frame[pos]
    info['Tmax_point'] = (pos[1]+y1, pos[0]+x1)
    pos = (frame.shape[0]//2, frame.shape[1]//2)
    info['Tcenter_C'] = frame[pos]
    info['Tcenter_point'] = (pos[1],pos[0])
