from astropy import units as u
from astropy.io import  ascii
from astropy.table import Table
from astropy.io import fits
import numpy as np
import glob
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize
from photutils import CircularAperture, SkyCircularAperture
from photutils import DAOStarFinder
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import os                        

def create_png(image, label):
    """image - list of fits file name, image[0] - original image, image[1] - image with noise
       d - list of cat tables from sextractor
       arc - error from sextractor
       label - list of names [original, noise, ...]"""
    sv = "/home/avojtekova/Desktop/final_results/star_det/generated_images/" 
     
    for i in range(len(image)):
        data =  fits.getdata(image[i][0], ext = 0)
        norm = ImageNormalize(data,interval = ZScaleInterval(), stretch = LinearStretch())
        
        print(image[i][0])
        plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm)#[1250:1750, 2000:2500]  add this when you want just part of image           
        plt.title(label[i])
        plt.axis('off')
        plt.tight_layout()
        plt.legend
        if i<2:
            if not os.path.isdir(sv + image[i][0][-33:-25] + "/") :
                os.makedirs(sv + image[i][0][-33:-25] + "/")
            plt.savefig(sv + image[i][0][-33:-25] + "/" + label[i]+ "_" + image[i][0][-33:-25] + "_big.png", dpi = 1000)#,bbox_inches='tight', pad_inches = 0)            
        else:
            if not os.path.isdir(sv + image[i][0][-40:-32] + "/") :
                os.makedirs(sv + image[i][0][-40:-32] + "/")
            plt.savefig(sv + image[i][0][-40:-32]  + "/" + label[i]+image[i][0][-40:-32] + "_big.png", dpi = 1000)#,bbox_inches='tight', pad_inches = 0)
        plt.close()  

names = ['11700_2t', '11700_93', '11730_04']
label1 = ['3 4000','13 4000','26 5000','27 5000','29 5000', '31 3000']
label = ['Original', 'Noise', '3 4000','13 4000','26 5000','27 5000','29 5000', '31 3000']

for n in names:
    d = []
    image = []
    image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/validation/*"+ n + "*.fits" ))))
    image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/validation/noise0/*"+ n + "*.fits" ))))
    
    for i in range(len(label1)):
        if i == 0:
            image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/final_result" +label1[i][0] + "/" + label1[i][2:] +"/noise0/*"+ n + "*.fits" ))))

        else:
            image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/final_result" +label1[i][:2] + "/" + label1[i][3:] +"/noise0/*"+ n + "*.fits" ))))
    create_png(image, label)
