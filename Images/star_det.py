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


#example of loading images
names = ['11700_2t', '11700_93', '11730_04']
label1 = ['3 4000','13 4000','26 5000','27 5000','29 5000', '31 3000']
label = ['Original', 'Noise', '3 4000','13 4000','26 5000','27 5000','29 5000', '31 3000']

for n in names:
    d = []
    image = []
    d.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/validation/cat_table2/*"+ n + "*.cat"))))
    image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/validation/*"+ n + "*.fits" ))))
    d.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/validation/noise0/cat_table2/*"+ n + "*.cat" ))))
    image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/validation/noise0/*"+ n + "*.fits" ))))
    
    for i in range(len(label1)):
        if i == 0:
            d.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/final_result" +label1[i][0] + "/" + label1[i][2:] +"/cat_table2/*"+ n + "*.cat" ))))
            image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/final_result" +label1[i][0] + "/" + label1[i][2:] +"/noise0/*"+ n + "*.fits" ))))

        else:
            image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/final_result" +label1[i][:2] + "/" + label1[i][3:] +"/noise0/*"+ n + "*.fits" ))))
            d.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/final_result" +label1[i][:2] + "/" + label1[i][3:] +"/cat_table2/*"+ n + "*.cat"))))                            




def detection(image, d, arc, label):
    """image - list of fits file name, image[0] - original image, image[1] - image with noise
       d - list of cat tables from sextractor
       arc - error from sextractor
       label - list of names [original, noise, ...]"""
    sv = "/home/avojtekova/Desktop/final_results/star_det/big/"# + str(arc) 
    table = Table.read(d[0][0])
    hdul = fits.open(image[1][0])
    h = hdul[0].header
    p = SkyCoord(table['ALPHA_J2000'], table['DELTA_J2000'], unit='deg', frame = 'fk5')
    a = SkyCircularAperture(p, r= 0.2 * u.arcsec)
    wcs = WCS(h)
    pix_aperture_orig = a.to_pixel(wcs)
    
    table = Table.read(d[1][0])
    p = SkyCoord(table['ALPHA_J2000'], table['DELTA_J2000'], unit='deg', frame = 'fk5')
    a = SkyCircularAperture(p, r= 0.4 * u.arcsec)
    wcs = WCS(h)
    pix_aperture_noise = a.to_pixel(wcs)
    
    
    
    for i in range(len(image)):
        print(image[i][0])
        data =  fits.getdata(image[i][0], ext = 0)
        table = Table.read(d[i][0])
        if i == 0:
            hdul = fits.open(image[i+1][0])
        else:
            hdul = fits.open(image[i][0])
        h = hdul[0].header
        #plt.subplot(4,2,i+1)
        positions = SkyCoord(table['ALPHA_J2000'], table['DELTA_J2000'], unit='deg', frame = 'fk5')
        apertures = SkyCircularAperture(positions, r= 0.3 * u.arcsec)
        norm = ImageNormalize(data,interval = ZScaleInterval(), stretch = LinearStretch())
        wcs = WCS(h)
        pix_aperture = apertures.to_pixel(wcs)
        plt.imshow(data, cmap='Greys_r', origin='lower', norm=norm)#[1250:1750, 2000:2500]  add this when you want just part of image
        
        if i !=1 and i != 0:
            pix_aperture.plot(color='red', lw=0.05,label = "Original") #origin=(2000, 1250) add this when you want just part of image, it is beginning of image
            pix_aperture_orig.plot(color='coral', lw=0.06, label = label[i], fill = False)
            pix_aperture_noise.plot(color='cyan', lw=0.03, alpha = 0.9 ,label = "Noise")  

        else:
            pix_aperture_orig.plot(color='coral', lw=0.06, label = label[i], fill = False)
            pix_aperture_noise.plot(color='cyan', lw=0.03, alpha = 0.9 ,label = "Noise")
 
            
        plt.title(label[i])
        plt.axis('off')
        plt.tight_layout()
        plt.legend
        if i<2:
            plt.savefig(sv + image[i][0][-33:-25] + "/" + label[i]+ "_" + image[i][0][-33:-25] + "_big.png", dpi = 1200)#,bbox_inches='tight', pad_inches = 0)
            
        else:
            plt.savefig(sv + image[i][0][-40:-32]  + "/" + label[i]+image[i][0][-40:-32] + "_big.png", dpi = 1200)#,bbox_inches='tight', pad_inches = 0)
        plt.close() 



names = ['11700_2t', '11700_93', '11730_04']
label1 = ['3 4000','13 4000','26 5000','27 5000','29 5000', '31 3000']
label = ['Original', 'Noise', '3 4000','13 4000','26 5000','27 5000','29 5000', '31 3000']

for n in names:
    d = []
    image = []
    d.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/validation/cat_table2/*"+ n + "*.cat"))))
    image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/validation/*"+ n + "*.fits" ))))
    d.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/validation/noise0/cat_table2/*"+ n + "*.cat" ))))
    image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/validation/noise0/*"+ n + "*.fits" ))))
    
    for i in range(len(label1)):
        if i == 0:
            d.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/final_result" +label1[i][0] + "/" + label1[i][2:] +"/cat_table2/*"+ n + "*.cat" ))))
            image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/final_result" +label1[i][0] + "/" + label1[i][2:] +"/noise0/*"+ n + "*.fits" ))))

        else:
            image.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/final_result" +label1[i][:2] + "/" + label1[i][3:] +"/noise0/*"+ n + "*.fits" ))))
            d.append(list(np.sort(glob.glob("/home/avojtekova/Desktop/final_results/final_result" +label1[i][:2] + "/" + label1[i][3:] +"/cat_table2/*"+ n + "*.cat"))))     
    detection(image, d, '01', label)
