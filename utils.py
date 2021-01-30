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

def setAnnotate(a, img, info, name, visible):
    (x,y) = info[name + '_point']
    a.xy = (x,y)
    a.set_text('%.2f$^\circ$C' % info[name+'_C'])
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
