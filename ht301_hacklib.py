#!/usr/bin/python3
import numpy as np
import math
import cv2

debug = 0

def f32(m3, idx):
    v = m3[idx:idx+4].view(dtype=np.dtype(np.float32))
    return float(v[0])

def u16(m3, idx):
    v = m3[idx:idx+4].view(dtype=np.dtype(np.uint16))
    return int(v[0])


Fix_, Distance_, refltmp_, airtmp_, Humi_, Emiss_ = 0., 0., 0., 0., 0., 0.
fpatmp_, fpaavg_, orgavg_, coretmp_ = 0., 0., 0., 0.

part_emi_t_1, part_Tatm_Trefl = 0., 0.
flt_10003360, flt_1000335C, flt_1000339C, flt_100033A4, flt_10003398 = 0., 0., 0., 0., 0.
flt_10003394 = 0., 0., 0.

ABSOLUTE_ZERO_CELSIUS = -273.15


#fpa - focal-plane array (sensor)


def sub_10001180(fpatmp_, coretmp_, cx):
    global Distance_, refltmp_, airtmp_, Humi_, Emiss_
    global flt_1000335C, flt_10003360, flt_1000339C, flt_10003394, flt_10003398

    # based on:
    # https://www.mdpi.com/1424-8220/17/8/1718 page: 4
    # https://github.com/mcguire-steve/ht301_ircam

    # w - coefficient showing the content of water vapour in atmosphere
    h0 = 1.5587
    h1 = 0.06938999999999999
    h2 = -0.00027816
    h3 = 0.00000068455

    w = math.exp(h3 * airtmp_ ** 3 + h2  * airtmp_ ** 2 + h1 * airtmp_ + h0) * Humi_

    # K_atm - scaling factor for the atmosphere damping
    # a1,a2 - attenuation for atmosphere without water vapor
    # b1,b2 - attenuation for water vapor
    
    K_atm = 1.9
    a1, a2 = 0.0066, 0.0126
    b1, b2 = -0.0023, -0.0067

    #t - transmittance of the atmosphere 
    d_ = -Distance_**0.5
    w_ = w ** 0.5
    t =  K_atm * math.exp(d_ * (a1 + b1 * w_)) + (1. - K_atm) * math.exp(d_ * (a2 + b2 * w_))

    if debug > 0:
        print('water vapour content coefficient:', w)
        print('transmittance of atmosphere:     ', t)

    part_emi_t_1 = 1.0 / (Emiss_ * t)
    part_Tatm_Trefl = (1.0 - Emiss_) * t * (refltmp_ - ABSOLUTE_ZERO_CELSIUS)**4  +  (1.0 - t) * (airtmp_ - ABSOLUTE_ZERO_CELSIUS)**4

#---------------------
    # a1 = coretmp_
    # flt_100033A4 = fpatmp_ 

    l_flt_1000337C = flt_1000335C / (2.0 * flt_10003360)
    l_flt_1000337C_2 = l_flt_1000337C **2


    v23 = flt_10003360 * coretmp_**2 + flt_1000335C * coretmp_;
    v22 = flt_1000339C * fpatmp_**2 + flt_10003398 * fpatmp_ + flt_10003394;

    type_ = 0
    if type_ != 0:
        v2 = 0;
    else:
        v2 = int(390.0 - fpatmp_ * 7.05)
    v4 = cx - v2;
    v5 = -v4;
    p = [];
    if (Distance_ >= 20):
        distance_c = (20        * 0.85 - 1.125) / 100.
    else:
        distance_c = (Distance_ * 0.85 - 1.125) / 100.

    np_v5 = np.arange(16384.0) - v4
    np_v8 = (np_v5 * v22 + v23) / flt_10003360 + l_flt_1000337C_2
    np_Ttot = np_v8**0.5 - l_flt_1000337C - ABSOLUTE_ZERO_CELSIUS
    np_Tobj_C = ((np_Ttot**4 - part_Tatm_Trefl) * part_emi_t_1)**0.25 + ABSOLUTE_ZERO_CELSIUS
    np_result = np_Tobj_C + distance_c * (np_Tobj_C - airtmp_)

    if debug > 1:
        v = np_result.tolist()
        print('cx:', cx, 'v2:', v2)
        print('v5:', v5)
        print('flt_1000339C', flt_1000339C, 'flt_10003398', flt_10003398, 'flt_10003394', flt_10003394, 'fpatmp_', fpatmp_)
        print('v22:', v22)
        print('v23:', v23)
        print('np1:', v[:10])
        print('np2:', v[-10:])
    return np_result



def temperatureLut(fpatmp_, meta3):

    global Fix_, Distance_, refltmp_, airtmp_, Humi_, Emiss_
    global fpaavg_, orgavg_, coretmp_ 

    global part_emi_t_1, part_Tatm_Trefl
    global flt_10003360, flt_1000335C, flt_1000339C, flt_100033A4, flt_10003398
    global flt_10003394

    m3 = meta3.view(dtype=np.dtype(np.uint8))

    v5 = meta3[0];
    coretmp_ = float(meta3[1]) / 10.0 + ABSOLUTE_ZERO_CELSIUS;

    flt_10003360 = f32(m3, 6);
    flt_1000335C = f32(m3, 10);
    flt_1000339C = f32(m3, 14);
    flt_10003398 = f32(m3, 18);
    flt_10003394 = f32(m3, 22);
    readParaFromDevFlag = True
    if readParaFromDevFlag:
        if debug > 0: print('m3:', m3[127*2:127*2+30])
        Fix_ = f32(m3,127*2);
        refltmp_ = f32(m3,127*2 + 4);
        airtmp_ = f32(m3,127*2 + 8);
        Humi_ = f32(m3,127*2 + 12);
        Emiss_ = f32(m3,127*2 + 16);
        Distance_ = u16(m3,127*2 + 20);
        #readParaFromDevFlag = 0;

    if debug > 0:
        print('Fix_',Fix_)
        print('refltmp_',refltmp_)
        print('airtmp_',airtmp_)
        print('Humi_',Humi_)
        print('Emiss_',Emiss_)
        print('Distance_',Distance_)

        print()
        print('v5',v5)
        print('coretmp_',coretmp_, meta3[1])
        print()
        print('flt_10003360',flt_10003360)
        print('flt_1000335C',flt_1000335C)
        print('flt_1000339C',flt_1000339C)
        print('flt_10003398',flt_10003398)
        print('flt_10003394',flt_10003394)

    if abs(Emiss_) < 0.0001 or abs(flt_10003360) < 0.0001:
        ##bugfix??
        return np.arange(16384.0)
    return sub_10001180(fpatmp_, coretmp_, v5); #//bug in IDA


def info(meta, device_strings, width, height):

    meta0, meta3 = meta[0], meta[3]

    Tfpa_raw = meta0[1]
    fpatmp_ = 20.0 - (float(Tfpa_raw) - 7800.0) / 36.0;

    temperature_LUT_C = temperatureLut(fpatmp_, meta3)

    fpaavg_  = meta0[0]
#   Tfpa_raw = meta0[1]
    Tmax_x   = meta0[2]
    Tmax_y   = meta0[3]
    Tmax_raw = meta0[4]
    Tmin_x   = meta0[5]
    Tmin_y   = meta0[6]
    Tmin_raw = meta0[7]

    orgavg_  = meta0[8]

    Tcenter_raw = meta0[12]
    Tarr0_raw = meta0[13]
    Tarr1_raw = meta0[14]
    Tarr2_raw = meta0[15]

    r_info = {
        'Tmin_C': temperature_LUT_C[Tmin_raw],
        'Tmin_raw': Tmin_raw,
        'Tmin_point': (Tmin_x, Tmin_y),
        'Tmax_C': temperature_LUT_C[Tmax_raw],
        'Tmax_raw': Tmax_raw,
        'Tmax_point': (Tmax_x, Tmax_y),
        'Tcenter_C': temperature_LUT_C[Tcenter_raw],
        'Tcenter_raw': Tcenter_raw,
        'Tcenter_point': (int(width/2), int(height/2)),
        'device_strings': device_strings,
        'device_type': device_strings[3]
    }

    if debug > 1:
        print('meta0 :',meta0.tolist())
        if debug > 2:  print('meta12:',meta[1:2].tolist())
        print('meta3 :',meta3.tolist())

    if debug > 0:
        print('fpatmp_:',fpatmp_,Tfpa_raw)
        print('fpaavg_:',fpaavg_)
        print('orgavg_:',orgavg_)
        print('TarrX_raw:',Tarr0_raw, Tarr1_raw, Tarr2_raw)

        for k in r_info:
            print(k+':',r_info[k])

    return r_info, temperature_LUT_C

def findString(m3chr, idx):
    try:
        ends = m3chr.index(0, idx)
    except ValueError:
        ends = idx
    return ends+1, ''.join(chr(x) for x in m3chr[idx:ends])

def device_info(meta):
    meta3 = meta[3]
    m3chr = list(meta3.view(dtype=np.dtype(np.uint8)))
    idx = 48
    device_strings = []
    for i in range(6):
        idx, s = findString(m3chr, idx)
        device_strings.append(s)
    if debug > 0: print('device_info:', device_strings)
    return device_strings



class HT301:
    FRAME_WIDTH = 384
    FRAME_HEIGHT = 292

    def __init__(self, video_dev = None):

        if video_dev == None:
            video_dev = self.find_device()

        self.cap = cv2.VideoCapture(video_dev)
        if not self.isHt301(self.cap):
            Exception('device ' + str(video_dev) + ": HT301 not found!")

        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        # Use raw mode
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8004)
        # Calibrate
        self.calibrate()
        #? enable thermal data - not needed
        #self.cap.set(cv2.CAP_PROP_ZOOM, 0x8020)

    def isHt301(self, cap):
        if not cap.isOpened():
            if debug > 0: print('open failed!')
            return False
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if debug > 0: print('width:', w, 'height:', h)
        if w == self.FRAME_WIDTH and h == self.FRAME_HEIGHT: return True
        return False

    def find_device(self):
        for i in range(10):
            if debug > 0: print('testing device nr:',i)
            cap = cv2.VideoCapture(i)
            ok = self.isHt301(cap)
            cap.release()
            if ok: return i
        raise Exception("HT301 device not found!")

    def read_(self):
        ret, frame = self.cap.read()
        dt = np.dtype('<u2')
        frame = frame.view(dtype=dt)
        frame = frame.reshape(self.FRAME_HEIGHT, self.FRAME_WIDTH)
        frame_raw = frame
        f_visible = frame_raw[:frame_raw.shape[0] - 4,...]
        meta      = frame_raw[frame_raw.shape[0] - 4:,...]
        return ret, frame_raw, f_visible, meta

    def read(self):
        frame_ok = False
        while not frame_ok:
            ret, frame_raw, frame, meta = self.read_()
            device_strings = device_info(meta)
            if device_strings[3] == 'T3-317-13': frame_ok = True
            else:
                if debug > 0: print('frame meta no match:', device_strings)

        self.frame_raw = frame_raw
        self.frame = frame
        self.meta  = meta
        self.device_strings  = device_strings
        return ret, self.frame

    def info(self):
        width, height = self.frame.shape
        return info(self.meta, self.device_strings, height, width)

    def calibrate(self):
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)

    def release(self):
        return self.cap.release()