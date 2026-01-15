"""
Wavelet-based Background Subtraction in 3D Fluorescence Microscopy
-Source Code-
Author: Manuel Hüpfel, Institute of Apllied Physics, KIT, Karlsruhe, Germany
Mail: manuel.huepfel@kit.edu
see README.txt for instructions
"""


#---IMPORT PACKAGES-----------------------------------------------------------

import os
from skimage import io
import numpy as np
from pywt import wavedecn, waverecn 
import tifffile as tif  
from scipy.ndimage import gaussian_filter  
from joblib import Parallel, delayed 
import multiprocessing
import matplotlib.pyplot as plt

#---DEFINE FUNCTIONS----------------------------------------------------------


def wavelet_based_BG_subtraction(image,num_levels,noise_lvl):
    coeffs = wavedecn(image, 'db1', level=None) #decomposition
    coeffs2 = coeffs.copy()
    coeffs_copy = coeffs.copy()

    for BGlvl in range(1, num_levels):
        coeffs[-BGlvl] = {k: np.zeros_like(v) for k, v in coeffs_copy[-BGlvl].items()} #set lvl 1 details  to zero

    Background = waverecn(coeffs, 'db1') #reconstruction
    del coeffs
    BG_unfiltered = Background
    Background = gaussian_filter(Background, sigma=2**num_levels) #gaussian filter sigma = 2^#lvls 

    coeffs2[0] = np.ones_like(coeffs2[0]) #set approx to one (constant)
    for lvl in range(1, len(coeffs2)-noise_lvl):
        coeffs2[lvl] = {k: np.zeros_like(v) for k, v in coeffs2[lvl].items()} #keep first detail lvl only
    
    Noise = waverecn(coeffs2, 'db1') #reconstruction
    del coeffs2

    return Background, Noise, BG_unfiltered

#---RUN WBNS; PLOT AND SAVE RESULTS-------------------------------------------

def WBNS_image(image, resolution_px, noise_lvl=1):

    #resolution_px : resolution in units of pixels (FWHM of the PSF)
    #noise_lvl: the noise level. If resolution_px > 6 then noise_lvl = 2 may be better 


    #number of levels for background estimate
    num_levels = np.uint16(np.ceil(np.log2(resolution_px)))

    #take image adjust shape if neccessary (padding) 
    img_type = image.dtype
    image = np.array(image,dtype = 'float32')

    if np.ndim(image) == 2:
        shape = np.shape(image)
        image = np.reshape(image, [1, shape[0], shape[1]])
    shape = np.shape(image)
    if shape[1] % 2 != 0:
        image = np.pad(image,((0,0), (0,1), (0, 0)), 'edge')
        pad_1 = True
    else:
        pad_1 = False
    if shape[2] % 2 != 0:
         image = np.pad(image,((0,0), (0,0), (0, 1)), 'edge')
         pad_2 = True
    else:
         pad_2 = False


    #extract background and noise
    num_cores = multiprocessing.cpu_count() #number of cores on your CPU
    res = Parallel(n_jobs=num_cores,max_nbytes=None)(delayed(wavelet_based_BG_subtraction)(image[slice],num_levels, noise_lvl) for slice in range(np.size(image,0)))
    Background, Noise, BG_unfiltered = zip(*res)

    Background, Noise, BG_unfiltered = zip(*res)

    #convert to float64 numpy array
    Noise = np.asarray(Noise,dtype = 'float32')
    Background = np.asarray(Background,dtype = 'float32')

    #undo padding
    if pad_1:
        image = image[:,:-1,:]
        Noise = Noise[:,:-1,:]
        Background = Background[:,:-1,:]
    if pad_2:
        image = image[:,:,:-1]
        Noise = Noise[:,:,:-1]
        Background = Background[:,:,:-1]



    #correct noise
    Noise[Noise<0] = 0 #positivity constraint
    noise_threshold = np.mean(Noise)+2*np.std(Noise)
    Noise[Noise>noise_threshold] = noise_threshold #2 sigma threshold reduces artifacts

    #subtract Noise
    result = image - Background
    result = result - Noise
    result[result<0] = 0 #positivity constraint

    #save result
    result = np.asarray(result,dtype=img_type.name)

    return result
