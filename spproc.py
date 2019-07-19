import os
import os.path as osp
import re
from collections import namedtuple
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

#================================================
#by Anton
#================================================
def replace_comma(lines):
    for line in lines:
        yield re.sub(r'(\d),(\d)', r'\1.\2', line)
        
def parse_metadata_line(line):
    k, v = line.split(':', 1)
    return k, v.strip()

Spectrum = namedtuple('Spectrum', ['info', 'data'])

def read_spectrum(filepath, is_relative, wlrange=(350, 800)):
    with open(filepath, 'r') as f:
        lines = list(replace_comma(l.strip() for l in f if l.strip()))
    data_marker = '>>>>>Begin Spectral Data<<<<<'
    index = lines.index(data_marker)
    header = lines[0]
    metadata = dict(parse_metadata_line(line) for line in lines[1:index])
    metadata['header'] = header
    metadata['relative'] = is_relative
    if is_relative:
        assert 'Relative' in header
    data = pd.DataFrame.from_records((map(float, line.split()) for line in lines[index + 1 :]),
                                    columns=('wl', 'ratio' if is_relative else 'raw_val'))
    data = data[(data.wl > wlrange[0]) & (data.wl < wlrange[1])]
    return Spectrum(metadata, data)

REF_lamp_ref = pd.read_csv('../source_calibration/calibration_source_lamp.lmp',
                           header=None, sep='\t', names=('wl', 'val'))
REF_lamp = read_spectrum('../source_calibration/Calibration_ON_FLMS061131_20-15-34-986.txt', False)
REF_lamp_dark = read_spectrum('../source_calibration/Calibration_OFF_FLMS061131_20-16-10-985.txt', False)
REF_room_dark = read_spectrum('../source_spectra/FLMS061131_16-12-55-435.dark.txt', False)

REF_radiometric_factors = np.interp(REF_lamp.data.wl, REF_lamp_ref.wl, REF_lamp_ref.val) / (REF_lamp.data.raw_val - REF_lamp_dark.data.raw_val)


def calibrate(s):
    new_data = s.data.copy()
    new_data['cal_val'] = (new_data.raw_val - REF_room_dark.data.raw_val) * REF_radiometric_factors
    new_info = s.info.copy()
    new_info['calibrated'] = True
    return Spectrum(info=new_info, data=new_data)

def read_source_spectrum(path):
    return calibrate(read_spectrum(path, False))

def plot_spectrum(spectrum, **kwargs):
    value = spectrum.data.ratio if spectrum.info['relative'] else spectrum.data.cal_val
    plt.plot(spectrum.data.wl, value, **kwargs)

#================================================
#! CIEXYZ matching functions
CIEXYZ_json = open('../camera.kinect1.json')
CIEXYZ = json.load(CIEXYZ_json)
xY = (np.asarray(CIEXYZ['sensitivities']['green']).T)[0]
yY = (np.asarray(CIEXYZ['sensitivities']['green']).T)[1]
xX = (np.asarray(CIEXYZ['sensitivities']['red']).T)[0]
yX = (np.asarray(CIEXYZ['sensitivities']['red']).T)[1]
xZ = (np.asarray(CIEXYZ['sensitivities']['blue']).T)[0]
yZ = (np.asarray(CIEXYZ['sensitivities']['blue']).T)[1]


def getXYZ(reflect_spec, illuminance_spec=None):
    ILLUM = np.array([])

    wl_interp = reflect_spec.data.wl 

    if illuminance_spec != None:
        illum_val = illuminance_spec.data.cal_val
        illum_wl = illuminance_spec.data.wl
        ILLUM = np.diag(np.interp(wl_interp, illum_wl, illum_val))
    else:
        ILLUM = np.eye(wl_interp.shape[0])

    c_x = np.interp(wl_interp, xX, yX)
    c_y = np.interp(wl_interp, xY, yY)
    c_z = np.interp(wl_interp, xZ, yZ)

    C_XYZ = np.array([c_x, c_y, c_z])

    return np.dot(np.dot(C_XYZ, ILLUM), reflect_spec.data.ratio)

def XYZ_to_RGB(XYZ):
    XYZ_to_RGB = np.array([[0.41847, -0.15866, -0.082835],
                            [-0.091169, 0.25243, 0.015708],
                            [0.000920, -0.002549, 0.17860]])
    return np.dot(XYZ_to_RGB, XYZ)

def C_sRGB(C_lin):
    if C_lin <= 0.0031308:
        return 12.92 * C_lin
    else: return 1.055 * C_lin ** (1 / 2.4) - 0.055

def RGB_to_sRGB(RGB):
    return np.array([C_sRGB(RGB[0]), C_sRGB(RGB[1]), C_sRGB(RGB[2])])

def XYZ_to_LMS(XYZ):
    XYZ_to_LMS = np.array([[0.38971, 0.68898, -0.07868],
                          [-0.22981, 1.18340, 0.04641],
                          [0.0, 0.0, 1.0]])
    return np.dot(XYZ_to_LMS, XYZ)

def calcA(SRCS_SPECTR, REFCOLORS_SPECTR, imgpatches, wlrange=(350, 800)):
    c_lms = []
    for CLR in REFCOLORS_SPECTR:
        clr = np.array([0.0, 0.0, 0.0])
        for SRC in SRCS_SPECTR:
            clr += getXYZ(CLR, SRC)
        #clr /= np.amax(clr)
        c_lms.append(clr)

    c = np.concatenate(c_lms)

    z = np.zeros((3))

    C_patches = []

    for patch in imgpatches:
        C_patches.append(np.array( [np.concatenate((patch,z,z)), np.concatenate((z,patch,z)), np.concatenate((z,z,patch))] ))

    C = np.concatenate(C_patches)

    a = np.dot(np.dot(np.linalg.inv(np.dot(C.T, C)), C.T), c)
    
    A = np.array([a[0:3], a[3:6], a[6:9]])
    return A


