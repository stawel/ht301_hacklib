#!/usr/bin/python3
import numpy as np
import math
import cv2
from datetime import datetime
from sys import platform
from dataclasses import dataclass
debug = 0


def f32(m3, idx):
    v = m3[idx : idx + 4].view(dtype=np.dtype(np.float32))
    return float(v[0])


def u16(m3, idx):
    v = m3[idx : idx + 4].view(dtype=np.dtype(np.uint16))
    return int(v[0])


Fix_, Distance_, refltmp_, airtmp_, Humi_, Emiss_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
fpatmp_, fpaavg_, orgavg_, coretmp_ = 0.0, 0.0, 0.0, 0.0

part_emi_t_1, part_Tatm_Trefl = 0.0, 0.0
flt_10003360, flt_1000335C, flt_1000339C, flt_100033A4, flt_10003398 = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
flt_10003394 = 0.0, 0.0, 0.0

ABSOLUTE_ZERO_CELSIUS = -273.15
# based on https://gitlab.com/netman69/inficam/-/blame/master/libinficam/src/main/jni/InfiCam/InfiFrame.cpp#L39


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

    w = math.exp(h3 * airtmp_**3 + h2 * airtmp_**2 + h1 * airtmp_ + h0) * Humi_

    # K_atm - scaling factor for the atmosphere damping
    # a1,a2 - attenuation for atmosphere without water vapor
    # b1,b2 - attenuation for water vapor

    K_atm = 1.9
    a1, a2 = 0.0066, 0.0126
    b1, b2 = -0.0023, -0.0067

    # t - transmittance of the atmosphere
    d_ = -(Distance_**0.5)
    w_ = w**0.5
    t = K_atm * math.exp(d_ * (a1 + b1 * w_)) + (1.0 - K_atm) * math.exp(
        d_ * (a2 + b2 * w_)
    )

    if debug > 0:
        print("water vapour content coefficient:", w)
        print("transmittance of atmosphere:     ", t)

    part_emi_t_1 = 1.0 / (Emiss_ * t)
    part_Tatm_Trefl = (1.0 - Emiss_) * t * (refltmp_ - ABSOLUTE_ZERO_CELSIUS) ** 4 + (
        1.0 - t
    ) * (airtmp_ - ABSOLUTE_ZERO_CELSIUS) ** 4

    # ---------------------
    # a1 = coretmp_
    # flt_100033A4 = fpatmp_

    l_flt_1000337C = flt_1000335C / (2.0 * flt_10003360)
    l_flt_1000337C_2 = l_flt_1000337C**2

    v23 = flt_10003360 * coretmp_**2 + flt_1000335C * coretmp_
    v22 = flt_1000339C * fpatmp_**2 + flt_10003398 * fpatmp_ + flt_10003394

    type_ = 0
    if type_ != 0:
        v2 = 0
    else:
        v2 = int(390.0 - fpatmp_ * 7.05)
    v4 = cx - v2
    v5 = -v4
    p = []
    if Distance_ >= 20:
        distance_c = (20 * 0.85 - 1.125) / 100.0
    else:
        distance_c = (Distance_ * 0.85 - 1.125) / 100.0

    np_v5 = np.arange(16384.0) - v4
    np_v8 = (np_v5 * v22 + v23) / flt_10003360 + l_flt_1000337C_2
    np_Ttot = np_v8**0.5 - l_flt_1000337C - ABSOLUTE_ZERO_CELSIUS
    np_Tobj_C = (
        (np_Ttot**4 - part_Tatm_Trefl) * part_emi_t_1
    ) ** 0.25 + ABSOLUTE_ZERO_CELSIUS
    np_result = np_Tobj_C + distance_c * (np_Tobj_C - airtmp_)

    if debug > 1:
        v = np_result.tolist()
        print("cx:", cx, "v2:", v2)
        print("v5:", v5)
        print(
            "flt_1000339C",
            flt_1000339C,
            "flt_10003398",
            flt_10003398,
            "flt_10003394",
            flt_10003394,
            "fpatmp_",
            fpatmp_,
        )
        print("v22:", v22)
        print("v23:", v23)
        print("np1:", v[:10])
        print("np2:", v[-10:])
    return np_result


def temperatureLut(fpatmp_, meta3):
    global Fix_, Distance_, refltmp_, airtmp_, Humi_, Emiss_
    global fpaavg_, orgavg_, coretmp_

    global part_emi_t_1, part_Tatm_Trefl
    global flt_10003360, flt_1000335C, flt_1000339C, flt_100033A4, flt_10003398
    global flt_10003394

    m3 = meta3.view(dtype=np.dtype(np.uint8))

    v5 = meta3[0]
    coretmp_ = float(meta3[1]) / 10.0 + ABSOLUTE_ZERO_CELSIUS

    flt_10003360 = f32(m3, 6)
    flt_1000335C = f32(m3, 10)
    flt_1000339C = f32(m3, 14)
    flt_10003398 = f32(m3, 18)
    flt_10003394 = f32(m3, 22)
    readParaFromDevFlag = True
    if readParaFromDevFlag:
        if debug > 0:
            print("m3:", m3[127 * 2 : 127 * 2 + 30])
        Fix_ = f32(m3, 127 * 2)
        refltmp_ = f32(m3, 127 * 2 + 4)
        airtmp_ = f32(m3, 127 * 2 + 8)
        Humi_ = f32(m3, 127 * 2 + 12)
        Emiss_ = f32(m3, 127 * 2 + 16)
        Distance_ = u16(m3, 127 * 2 + 20)
        # readParaFromDevFlag = 0;

    if debug > 0:
        print("Fix_", Fix_)
        print("refltmp_", refltmp_)
        print("airtmp_", airtmp_)
        print("Humi_", Humi_)
        print("Emiss_", Emiss_)
        print("Distance_", Distance_)

        print()
        print("v5", v5)
        print("coretmp_", coretmp_, meta3[1])
        print()
        print("flt_10003360", flt_10003360)
        print("flt_1000335C", flt_1000335C)
        print("flt_1000339C", flt_1000339C)
        print("flt_10003398", flt_10003398)
        print("flt_10003394", flt_10003394)

    if abs(Emiss_) < 0.0001 or abs(flt_10003360) < 0.0001:
        ##bugfix??
        return np.arange(16384.0)
    return sub_10001180(fpatmp_, coretmp_, v5)
    # //bug in IDA





def info(meta, device_strings, width, height,  meta_mapping=[0, 3]):
    meta0, meta3 = meta[meta_mapping[0]], meta[meta_mapping[1]]

    Tfpa_raw = meta0[1]
    fpatmp_ = 20.0 - (float(Tfpa_raw) - 7800.0) / 36.0

    temperature_LUT_C = temperatureLut(fpatmp_, meta3)

    fpaavg_ = meta0[0]
    #   Tfpa_raw = meta0[1]
    Tmax_x = meta0[2]
    Tmax_y = meta0[3]
    Tmax_raw = meta0[4]
    Tmin_x = meta0[5]
    Tmin_y = meta0[6]
    Tmin_raw = meta0[7]

    orgavg_ = meta0[8]

    Tcenter_raw = meta0[12]
    Tarr0_raw = meta0[13]
    Tarr1_raw = meta0[14]
    Tarr2_raw = meta0[15]

    r_info = {
        "Tmin_C": temperature_LUT_C[Tmin_raw],
        "Tmin_raw": Tmin_raw,
        "Tmin_point": (Tmin_x, Tmin_y),
        "Tmax_C": temperature_LUT_C[Tmax_raw],
        "Tmax_raw": Tmax_raw,
        "Tmax_point": (Tmax_x, Tmax_y),
        "Tcenter_C": temperature_LUT_C[Tcenter_raw],
        "Tcenter_raw": Tcenter_raw,
        "Tcenter_point": (int(width / 2), int(height / 2)),
        "device_strings": device_strings,
        "device_type": device_strings[3],
        "date": datetime.now(),
        "meta": meta,
    }

    if debug > 1:
        print("meta0 :", meta0.tolist())
        if debug > 2:
            print("meta12:", meta[1:2].tolist())
        print("meta3 :", meta3.tolist())

    if debug > 0:
        print("fpatmp_:", fpatmp_, Tfpa_raw)
        print("fpaavg_:", fpaavg_)
        print("orgavg_:", orgavg_)
        print("TarrX_raw:", Tarr0_raw, Tarr1_raw, Tarr2_raw)

        for k in r_info:
            print(k + ":", r_info[k])

    return r_info, temperature_LUT_C


def findString(m3chr, idx):
    try:
        ends = m3chr.index(0, idx)
    except ValueError:
        ends = idx
    return ends + 1, "".join(chr(x) for x in m3chr[idx:ends])


def device_info(meta, meta_mapping=3, idx=48):
    meta3 = meta[meta_mapping]
    m3chr = list(meta3.view(dtype=np.dtype(np.uint8)))
    device_strings = []
    for i in range(6):
        idx, s = findString(m3chr, idx)
        device_strings.append(s)
    if debug > 0:
        print("device_info:", device_strings)
    return device_strings


class HT301:
    FRAME_RAW_WIDTH = 384
    FRAME_RAW_HEIGHT = 292
    FRAME_WIDTH = FRAME_RAW_WIDTH
    FRAME_HEIGHT = FRAME_RAW_HEIGHT - 4

    def __init__(self, video_dev=None):
        if video_dev == None:
            video_dev = self.find_device()

        # loosely taken from https://framagit.org/ericb/ir_thermography/-/blob/master/ht301_hacklib/ht301_hacklib.py
        if platform.startswith("linux"):
            # ensure v4l2 is used on Linux as gstreamer is broken with OpenCV
            # see : https://github.com/opencv/opencv/issues/10324
            self.cap = cv2.VideoCapture(video_dev, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(video_dev)

        if not self.isHt301(self.cap):
            Exception("device " + str(video_dev) + ": HT301 or T3S not found!")

        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        # Use raw mode
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8004)
        # Calibrate
        self.calibrate()
        # ? enable thermal data - not needed
        # self.cap.set(cv2.CAP_PROP_ZOOM, 0x8020)
        self.frame_raw = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def isHt301(self, cap):
        if not cap.isOpened():
            if debug > 0:
                print("open failed!")
            return False
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if debug > 0:
            print("width:", w, "height:", h)
        if w == self.FRAME_RAW_WIDTH and h == self.FRAME_RAW_HEIGHT:
            return True
        return False

    def find_device(self):
        for i in range(10):
            if debug > 0:
                print("testing device nr:", i)
            cap = cv2.VideoCapture(i)
            ok = self.isHt301(cap)
            cap.release()
            if ok:
                return i
        raise Exception("HT301 or T3S device not found!")

    def read_(self):
        ret, frame = self.cap.read()
        dt = np.dtype("<u2")
        frame = frame.view(dtype=dt)
        frame = frame.reshape(self.FRAME_RAW_HEIGHT, self.FRAME_RAW_WIDTH)
        frame_raw = frame
        f_visible = frame_raw[: frame_raw.shape[0] - 4, ...]
        meta = frame_raw[frame_raw.shape[0] - 4 :, ...]
        return ret, frame_raw, f_visible, meta

    def read(self):
        frame_ok = False
        while not frame_ok:
            ret, frame_raw, frame, meta = self.read_()
            device_strings = device_info(meta)
            if device_strings[3] == "T3-317-13":
                frame_ok = True
            elif device_strings[4] == "T3-317-13":
                frame_ok = True
            elif device_strings[5] == "T3S-A13":
                frame_ok = True
            else:
                if debug > 0:
                    print("frame meta no match:", device_strings)
                if self.frame_raw != None:
                    return False, self.frame

        self.frame_raw = frame_raw
        self.frame = frame
        self.meta = meta
        self.device_strings = device_strings
        return ret, self.frame

    def info(self):
        width, height = self.frame.shape
        return info(self.meta, self.device_strings, height, width)

    def calibrate(self):
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)

    def release(self):
        return self.cap.release()


class T2SPLUS(HT301):
    FRAME_RAW_WIDTH = 256
    FRAME_RAW_HEIGHT = 196
    FRAME_WIDTH = FRAME_RAW_WIDTH
    FRAME_HEIGHT = FRAME_RAW_HEIGHT - 4

    def read(self):
        frame_ok = False
        while not frame_ok:
            ret, frame_raw, frame, meta = self.read_()
            device_strings = device_info(meta, meta_mapping=2, idx=0)
            if device_strings[1] == "T2S+":
                frame_ok = True
            else:
                if debug > 0:
                    print("frame meta no match:", device_strings)
                if self.frame_raw != None:
                    return False, self.frame

        self.frame_raw = frame_raw
        self.frame = frame
        self.meta = meta
        self.device_strings = device_strings
        return ret, self.frame

    def info(self):
        width, height = self.frame.shape
        return info(self.meta, self.device_strings, height, width, meta_mapping=[0, 1])

    def temperature_range_normal(self):
        """Switch camera to the normal temperature range (-20°C to 120°C)"""
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8020)

    def temperature_range_high(self):
        """Switch camera to the high temperature range (-20°C to 450°C)"""
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8021)


import numpy as np
import math
import cv2
from sys import platform
from typing import Tuple

SET_CORRECTION  = 0 * 4
SET_REFLECTION  = 1 * 4
SET_AMB         = 2 * 4
SET_HUMIDITY    = 3 * 4
SET_EMISSIVITY  = 4 * 4
SET_DISTANCE    = 5 * 4

def read_u16(arr_u16, offset):
    ''' arr: np.uint16 '''
    return arr_u16[offset]

def read_f32(arr_u16, offset, step=2):
    ''' arr: np.uint16 '''
    return arr_u16[offset:offset+step].view(np.float32)[0]

def read_u8(arr_u16, offset, step):
    return arr_u16[offset:offset + step].view(np.uint8)

class InfiFrame:
    supported_widths = {240, 256, 392, 640}
    ZEROC = 273.15
    distance_multiplier = 1.0
    offset_temp_shutter = 0.0
    offset_temp_fpa = 0.0

    range = 120
    cal_00_offset = 390.0
    cal_00_fpamul = 7.05
    
    correction_coefficient_m = 1
    correction_coefficient_b = 0
    
    height:int
    width:int
    frame:np.ndarray
    frame_raw:np.ndarray
    meta:np.ndarray
    device_strings:Tuple[str, str, str, str, str, str]
    userArea:int
    amountPixels:int
    fourLinePara:int
    cap:cv2.VideoCapture
    frame_raw_u16:np.ndarray
    def __init__(self, video_dev:cv2.VideoCapture|None=None) -> None:
        if video_dev == None:
            video_dev = self.find_device()
        if not video_dev:
            raise Exception("No video device found!")
        self.cap = video_dev
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.fourLinePara = self.width * (self.height - 4)
        self.init_parameters()
        self.userArea = self.amountPixels + 127
        

        # Decide whether or not convert data to RGB
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        
        # using Raw mode 16 bit data
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8004)
        
        self.calibrate()

 
    def find_device(cls, width=None, height=None) -> cv2.VideoCapture:
        """Find a supported thermal camera
         
          Optionally narrow down the resultion of the camera too look for."""
        for i in range(10):
            try:
                if platform.startswith('linux'):
                    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                else:
                    cap = cv2.VideoCapture(i)
                   

                if cap.get(cv2.CAP_PROP_FRAME_WIDTH) in cls.supported_widths: 
                    if width is not None and cap.get(cv2.CAP_PROP_FRAME_WIDTH) != width:
                        continue
                    if height is not None and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) != height:
                        continue
                    return cap

            except: pass
        
        raise ValueError(f"Cannot find camera with a width of one of {cls.supported_widths} that also matches: {width=} and {height=}")

    def update(self) -> Tuple[dict, np.ndarray]:
        shutTemper = read_u16(self.frame_raw_u16, self.fourLinePara + self.amountPixels + 1)
        floatShutTemper = shutTemper / 10.0 - self.ZEROC
        
        coreTemper = read_u16(self.frame_raw_u16, self.fourLinePara + self.amountPixels + 2)
        floatCoreTemper = coreTemper / 10.0 - self.ZEROC
        
        cal_00 = float(read_u16(self.frame_raw_u16, self.fourLinePara + self.amountPixels))
        self.cal_01 = read_f32(self.frame_raw_u16, self.fourLinePara + self.amountPixels + 3)
        cal_02 = read_f32(self.frame_raw_u16, self.fourLinePara + self.amountPixels + 5)
        cal_03 = read_f32(self.frame_raw_u16, self.fourLinePara + self.amountPixels + 7)
        cal_04 = read_f32(self.frame_raw_u16, self.fourLinePara + self.amountPixels + 9)
        cal_05 = read_f32(self.frame_raw_u16, self.fourLinePara + self.amountPixels + 11)
        
        cameraSoftVersion: np.ndarray = read_u8(self.frame_raw_u16, self.fourLinePara + self.amountPixels + 24, step=8)
        cameraSoftVersion = cameraSoftVersion.tobytes().decode("ascii").rstrip("\x00")
        
        sn: np.ndarray = read_u8(self.frame_raw_u16, self.fourLinePara + self.amountPixels + 32, step=3) 
        sn = sn.tobytes().decode("ascii").rstrip("\x00")
        
        correction = read_f32(self.frame_raw_u16, self.fourLinePara + self.userArea)
        Refltmp = read_f32(self.frame_raw_u16, self.fourLinePara + self.userArea + 2)
        Airtmp = read_f32(self.frame_raw_u16, self.fourLinePara + self.userArea + 4)
        humi = read_f32(self.frame_raw_u16, self.fourLinePara + self.userArea + 6)
        emiss = read_f32(self.frame_raw_u16, self.fourLinePara + self.userArea + 8) 
        distance = read_u16(self.frame_raw_u16, self.fourLinePara + self.userArea + 10)
        
        fpa_avg = read_u16(self.frame_raw_u16, self.fourLinePara)
        fpaTmp = read_u16(self.frame_raw_u16, self.fourLinePara + 1)
        maxx1 = read_u16(self.frame_raw_u16, self.fourLinePara + 2)
        maxy1 = read_u16(self.frame_raw_u16, self.fourLinePara + 3)
        self.max_raw = read_u16(self.frame_raw_u16, self.fourLinePara + 4)
        minx1 = read_u16(self.frame_raw_u16, self.fourLinePara + 5)
        miny1 = read_u16(self.frame_raw_u16, self.fourLinePara + 6)
        self.min_raw = read_u16(self.frame_raw_u16, self.fourLinePara + 7)
        self.avg_raw = read_u16(self.frame_raw_u16, self.fourLinePara + 8)
        
        fpatmp_ = 20.0 - (float(fpaTmp) - self.fpa_off) / self.fpa_div
        
        center_raw = read_u16(self.frame_raw_u16, self.fourLinePara + 12)
        user_raw00 = read_u16(self.frame_raw_u16, self.fourLinePara + 13)
        user_raw01 = read_u16(self.frame_raw_u16, self.fourLinePara + 14)
        user_raw02 = read_u16(self.frame_raw_u16, self.fourLinePara + 15)
        
        distance_adjusted = (20.0 if distance >= 20.0 else distance) * self.distance_multiplier
        atm = self.atmt(humi, Airtmp, distance_adjusted)
        self.numerator_sub = (1.0 - emiss) * atm * math.pow(Refltmp + self.ZEROC, 4) + (1.0 - atm) * math.pow(Airtmp + self.ZEROC, 4)
        self.denominator = emiss * atm
        
        ts = floatShutTemper + self.offset_temp_shutter
        tfpa = fpatmp_ + self.offset_temp_fpa
        
        self.cal_a = cal_02 / (self.cal_01 + self.cal_01)
        self.cal_b = cal_02 * cal_02 / (self.cal_01 * self.cal_01 * 4.0)
        self.cal_c = self.cal_01 * math.pow(ts, 2) + ts * cal_02
        self.cal_d = cal_03 * math.pow(tfpa, 2) + cal_04 * tfpa + cal_05

        cal_00_corr = 0
        
        if self.range == 120:
            cal_00_corr = int(self.cal_00_offset - tfpa * self.cal_00_fpamul)
        
        table_offset = cal_00 - (cal_00_corr if cal_00_corr > 0 else 0)
        
        temperatureTable = self.get_temp_table(correction, Airtmp, table_offset, distance_adjusted)
        
        ''' build infomation '''
        info = {
            "temp_shutter": floatShutTemper,
            "temp_core": floatCoreTemper,
            "cameraSoftVersion": cameraSoftVersion,
            "sn": sn,
            "correction": correction,
            "temp_reflected": Refltmp,
            "temp_air": Airtmp,
            "humidity": humi,
            "emissivity": emiss,
            "distance": distance,
            "fpa_average": fpa_avg,
            "temp_fpa": fpatmp_,
            "temp_max_x": maxx1,
            "temp_max_y": maxy1,
            "temp_max_raw": self.max_raw,
            "temp_max": temperatureTable[self.max_raw],
            "temp_min_x": minx1,
            "temp_min_y": miny1,
            "temp_min_raw": self.min_raw,
            "temp_min": temperatureTable[self.min_raw],
            "temp_average_raw": self.avg_raw,
            "temp_average": temperatureTable[self.avg_raw],
            "temp_center_raw": center_raw,
            "temp_center": temperatureTable[center_raw],
            "temp_user_00": temperatureTable[user_raw00],
            "temp_user_01": temperatureTable[user_raw01],
            "temp_user_02": temperatureTable[user_raw02],
            "Tmin_point": (minx1, miny1),
            "Tmax_point": (maxx1, maxy1),	
            "Tcenter_point": (self.width // 2, self.height // 2),
            "Tmin_C": temperatureTable[self.min_raw],
            "Tmax_C": temperatureTable[self.max_raw],
            "Tcenter_C": temperatureTable[center_raw],
        }
        
        return info, temperatureTable
    
    def info(self) -> Tuple[dict, np.ndarray]:
        return self.update()

    # read raw data from cam, seperate visible frame from metadata
    def read_data(self) -> Tuple[bool, np.ndarray]:
        ret, frame_raw = self.cap.read()
        self.frame_raw_u16: np.ndarray = frame_raw.view(np.uint16).ravel()
        frame_visible = self.frame_raw_u16[:self.fourLinePara].copy().reshape(self.height - 4, self.width)
        return ret, frame_visible

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.read_data()

    def set_correction(self, correction: float) -> None:
        self.sendFloatCommand(position=SET_CORRECTION, value=correction)
    
    def set_reflection(self, reflection: float) -> None:
        self.sendFloatCommand(position=SET_REFLECTION, value=reflection)

    def set_amb(self, amb: float) -> None:
        self.sendFloatCommand(position=SET_AMB, value=amb)
    
    def set_humidity(self, humidity: float) -> None:
        self.sendFloatCommand(position=SET_HUMIDITY, value=humidity)
    
    def set_emissivity(self, emiss: float) -> None:
        self.sendFloatCommand(position=SET_EMISSIVITY, value=emiss)
    
    def set_distance(self, distance: int) -> None:
        self.sendUshortCommand(position=SET_DISTANCE, value=distance)

    ''' Control methods'''
    # send command and a float value to camera
    def sendFloatCommand(self, position: int, value: float) -> None:
        # Split float to 4 bytes
        b0, b1, b2, b3 = np.array([value], dtype=np.float32).view(np.uint8)
        
        positionAndValue0 = (position << 8) | (0x000000ff & b0)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue0):
            print("Control fail {}".format(positionAndValue0))
        
        positionAndValue1 = ((position + 1) << 8) | (0x000000ff & b1)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue1):
            print("Control fail {}".format(positionAndValue1))
            
        positionAndValue2 = ((position + 2) << 8) | (0x000000ff & b2)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue2):
            print("Control fail {}".format(positionAndValue2))

        positionAndValue3 = ((position + 3) << 8) | (0x000000ff & b3)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue3):
            print("Control fail {}".format(positionAndValue3))

    # send command and 16 bit value to camera
    def sendUshortCommand(self, position: int, value: int) -> None:
        value0, value1 = np.array([value], dtype=np.uint16).view(np.uint8)
        positionAndValue0 = (position << 8) | (0x000000ff & value0)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue0):
            print("Control fail {}".format(positionAndValue0))
        
        positionAndValue1 = ((position + 1) << 8) | (0x000000ff & value1)
        if not self.cap.set(cv2.CAP_PROP_ZOOM, positionAndValue1):
            print("Control fail {}".format(positionAndValue1))

    # send command and byte value to camera
    def sendByteCommand(self, position: int, value: int) -> None:
        value0 = np.array([value], dtype=np.uint8)[0]
        psitionAndValue0 = (position << 8) | (0x000000ff & value0)
        self.cap.set(cv2.CAP_PROP_ZOOM, psitionAndValue0)

    # save set parameters
    def save_parameters(self) -> None:
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x80ff)

    # set custom point to measure temperature
    def set_point(self, x: int, y: int, index: int) -> None:
        match index:
            case 0:
                x1 = 0xf000 + x
                y1 = 0xf200 + y
            case 1:
                x1 = 0xf400 + x
                y1 = 0xf600 + y
            case 2:
                x1 = 0xf800 + x
                y1 = 0xfa00 + y
            case _:
                raise ValueError("Invalid index: {}.\nCan only set 3 custom points to measure temperature at indexes: {}, {}, {}".format(index, 0, 1, 2))
        
        self.cap.set(cv2.CAP_PROP_ZOOM, x1)
        self.cap.set(cv2.CAP_PROP_ZOOM, y1)

    def calibrate(self) -> None:
        '''camera calibration'''
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8000)

    def release(self) -> None:
        ''' Release cap opencv '''
        self.cap.release()

    def init_parameters(self) -> None:
        ''' Initalize parameters based on thermal camera resolution '''
        match self.width:
            case 640:
                self.fpa_off = 6867
                self.fpa_div = 33.8
                self.amountPixels = self.width * 3
            case 384:
                self.fpa_off = 7800
                self.fpa_div = 36.0
                self.amountPixels = self.width * 3 
            case 256:
                self.fpa_off = 8617
                self.fpa_div = 37.682
                self.amountPixels = self.width
                self.cal_00_offset = 170.0
                self.cal_00_fpamul = 0.0
            case 240:
                self.fpa_off = 7800
                self.fpa_div = 36.0
                self.amountPixels = self.width
            case _: raise ValueError("{} does not match supported device".format(self.width))
    

    
    ''' Temperature calculation '''        
    # Water vapor coefficient from humidity and ambient temperature
    def wvc(self, h: float, t_atm: float):
        h1 = 1.5587
        h2 = 0.06939
        h3 = -2.7816e-4
        h4 = 6.8455e-7
        return h * math.exp(h1 + h2 * t_atm + h3 * math.pow(t_atm, 2) + h4 * math.pow(t_atm, 3))

    # Transmittance of the atmosphere from humitity, ambient temperature and distance.
    def atmt(self, h: float, t_atm: float, d: float):
        k_atm = 1.9
        nsqd = -math.sqrt(d)
        sqw = math.sqrt(self.wvc(h, t_atm))
        
        '''Athmospheric attenuation without water vapor'''
        a1 = 0.006569
        a2 = 0.01262
        
        '''Attenuation for water vapor.'''
        b1 = -0.002276
        b2 = -0.00667
        return k_atm * math.exp(nsqd * (a1 + b1 * sqw)) + (1.0 - k_atm) * math.exp(nsqd * (a2 + b2 * sqw))

    # calculate temperature table 
    # for each 16 bit value from frame data will return correspond temperture value
    def get_temp_table(self, correction, Airtmp, table_offset, distance_adjusted):
        ''' x: uint16 '''
        n = np.sqrt(np.abs(((np.arange(16384, dtype=np.float32) - table_offset) * self.cal_d + self.cal_c) / self.cal_01 + self.cal_b))
        n[np.isnan(n)] = 0.0        
        wtot = np.power(n - self.cal_a + self.ZEROC, 4)
        ttot = np.power((wtot - self.numerator_sub) / self.denominator, 0.25) - self.ZEROC
        temperatureTable = ttot + (distance_adjusted * 0.85 - 1.125) * (ttot - Airtmp) / 100.0 + correction
        return self.correction_coefficient_m * temperatureTable + self.correction_coefficient_b

    def temperature_range_normal(self):
        """Switch camera to the normal temperature range (-20°C to 120°C)"""
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8020)
        self.correction_coefficient_m = 1
        self.correction_coefficient_b = 0

    def temperature_range_high(self):
        """Switch camera to the high temperature range (-20°C to 450°C)"""
        self.cap.set(cv2.CAP_PROP_ZOOM, 0x8021)
        self.correction_coefficient_m = 1.17
        self.correction_coefficient_b = -40.9