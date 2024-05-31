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

def autoExposure(update, exposure, frame):
    # Sketchy auto-exposure
    lmin, lmax = frame.min(), frame.max()
    T_min, T_max, T_margin = exposure['T_min'], exposure['T_max'], exposure['T_margin']
    if exposure['auto_type'] == 'center':
        T_cent = int((T_min+T_max)/2)
        d = int(max(T_cent-lmin, lmax-T_cent, 0) + T_margin)
        if lmin < T_min or T_max < lmax or (T_min + 2 * T_margin < lmin and T_max - 2 * T_margin > lmax):
#            print('d:',d, 'lmin:', lmin, 'lmax:', lmax)
            update = True
            T_min, T_max = T_cent - d, T_cent + d
#            print('T_min:', T_min, 'T_cent:', T_cent, 'T_max:', T_max)
    if exposure['auto_type'] == 'ends':
        if T_min                > lmin: update, T_min = True, lmin-T_margin
        if T_min + 2 * T_margin < lmin: update, T_min = True, lmin-T_margin
        if T_max                < lmax: update, T_max = True, lmax+T_margin
        if T_max - 2 * T_margin > lmax: update, T_max = True, lmax+T_margin

    exposure['T_min'] = T_min
    exposure['T_max'] = T_max
    return update

def correctRoi(roi, shape):
    ((x,y),(w,h)) = roi
#    if w == 0 and h == 0:
#        ((x,y),(w,h)) = ((0,0), shape)
    x1,x2 = max(0, min(x,x+w)), max(0, x,x+w)
    y1,y2 = max(0, min(y,y+h)), max(0, y,y+h)
#    if x1 == x2: x2 += 1
#    if y1 == y2: y2 += 1
    return ((x1,y1),(x2,y2))


def inRoi(roi, point, shape):
    result = False
    ((x1,y1),(x2,y2)) = correctRoi(roi, shape)
    if x1 < point[0] and point[0] < x2:
        if y1 < point[1] and point[1] < y2:
            result = True
    return result

def subdict(d, l):
    return dict((k,d[k]) for k in l if k in d)





class Annotations:
    def __init__(self, ax, patches):
        self.ax = ax
        self.astyle = dict(text='', xy=(0, 0), xytext=(0, 0), textcoords='offset pixels', arrowprops=dict(facecolor='black', arrowstyle="->"))
        self.anns = {}
        self.roi_patch = ax.add_patch(patches.Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='black', facecolor='none'))
        self.set_roi(((0,0),(0,0)))

    def set_roi(self, roi):
        self.roi = roi
        ((x,y), (w,h)) = roi
        self.roi_patch.xy = (x,y)
        self.roi_patch.set_width(w)
        self.roi_patch.set_height(h)
        self.roi_patch.set_visible(w!=0 and h!=0)

    def get_ann(self, name, color):
        if name not in self.anns:
            self.anns[name] = self.ax.annotate(**self.astyle, bbox=dict(boxstyle='square', fc=color, alpha=0.3, lw=0))
        return self.anns[name]

    def get_pos(self, name):
        pos = self.get_ann(name, "").xy
        return pos
    
    def get_val(self, name, annotation_frame):
        annotation_pos = self.get_pos(name)
        val = annotation_frame[annotation_pos[::-1]]
        return val

    def update(self, temp_annotations, annotation_frame, draw_temp):
        l = temp_annotations['std'].items() | temp_annotations['user'].items()
        for name, color in l:
            pos = self._get_pos(name, annotation_frame, self.roi)
            self._ann_set_temp(self.get_ann(name, color), pos, annotation_frame, draw_temp)

    def get(self):
        return list(self.anns.values()) + [self.roi_patch]

    def remove(self, d):
        for name in d:
            if name in self.anns:
                self.anns[name].remove()
                del self.anns[name]
        d.clear()

    def _ann_set_temp(self, ann, pos, annotation_frame, draw_temp):
        (x,y) = pos
        ann.xy  = pos
        value = annotation_frame[pos[1], pos[0]]
        ann.set_text('%.2f$^\circ$C' % value)
        ann.set_visible(draw_temp)
        tx,ty = 20, 15
        if x > annotation_frame.shape[1]-50: tx = -80
        if y < 30: ty = -15
        ann.xyann = (tx, ty)

    
    def _get_pos(self, name, annotation_frame, roi):
        ((x1,y1),(x2,y2)) = correctRoi(roi, annotation_frame.shape)
        roi_frame = annotation_frame[y1:y2,x1:x2]
        if roi_frame.size <= 0:
            x1,y1 = 0, 0
            roi_frame = annotation_frame
        if name == 'Tmin':
            pos = np.unravel_index(roi_frame.argmin(), roi_frame.shape)
            pos = (pos[1]+x1, pos[0]+y1)
        elif name == 'Tmax':
            pos = np.unravel_index(roi_frame.argmax(), roi_frame.shape)
            pos = (pos[1]+x1, pos[0]+y1)
        elif name == 'Tcenter':
            pos = (annotation_frame.shape[1]//2, annotation_frame.shape[0]//2)
        else:
            pos = name
        return pos

def pixel_to_temp(pixel_value, min_temp, max_temp):
    return ((pixel_value / 255) * (max_temp - min_temp)) + min_temp


def temp_img(frame, min_temp, max_temp):
    normalized_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.normalize(normalized_image, None, 0, 255, cv2.NORM_MINMAX)
    return pixel_to_temp(normalized_image, min_temp, max_temp)


def hot_area(frame, temperature_image, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.where(temperature_image < threshold, 0, gray)
    return mask, cv2.bitwise_and(frame, frame, mask=mask)


def contoured_hottest_area(frame, mask):
    cont_frame = frame.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(cont_frame, contours, -1, (0, 255, 0), 2)
    return cont_frame


def temp_clustering(hottest_area, cluster_no):
    pic = cv2.cvtColor(hottest_area, cv2.COLOR_BGR2RGB)
    pixel_vals = pic.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.9)
    k = cluster_no
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 20, cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(pic.shape)
    return k, centers, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)


def cluster_centroids(cluster_centers, clustered_temp):
    centroid_locations = []

    for row in cluster_centers:
        cluster_i = np.where(clustered_temp == row)
        center_x = np.mean(cluster_i[0]).astype(int)
        center_y = np.mean(cluster_i[1]).astype(int)
        centroid_locations.append([center_x, center_y])

    return np.array(centroid_locations)
