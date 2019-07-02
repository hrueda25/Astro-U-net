import os, time, scipy.io
import numpy as np
import glob
from astropy.io import fits

#create and save fits image with noise and shorter exposure time (depending on ratio)

def fits_read(name):
    f = fits.open(name)
    if "EXPTIME" in f[0].header:
        e = f[0].header["EXPTIME"]
    else: 
        e = 0
    return e

def poisson_noise(name, exp_time  = None, ratio = None, ron = 3, dk = 7, sv_name = None, path = None, ret = True, save = False):
    """name - name of fits long-exposure image (real image from Hubble space telescope)
       exp_time - exposure time of real fits image
       ratio =  long_exposure/short_exposure
       ron - read out noise
       dk - dark current"""
    img_data, h = fits.getdata(name, 0, header=True) #e/sec
    width, height = img_data.shape[0:2]
    if  exp_time  is None:
        exp_time = fits_read(img_data) 
    img = img_data * exp_time # e
    DN = np.random.poisson(np.sqrt(dk*exp_time/ratio),(width,height))
    RON = np.random.poisson(ron,(width,height))
    SN = np.random.poisson(np.sqrt(img/ratio))  
    noise_img = (img/ratio + SN + RON + DN)/(exp_time/ratio)
    noise_img = np.where(img_data == 0.00000000e+00, 0.00000000e+00, noise_img)
    if save:
        fits.writeto(path + sv_name, noise_img, h)
    if ret:    
        return noise_img


