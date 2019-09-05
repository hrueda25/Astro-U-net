from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from astroquery.mast import Observations
import urllib.request
import os


filter_wide = ["F555W","F606W"]
for f in filter_wide:
    if not os.path.isdir("data/" + f + "/"):
        os.makedirs("data/" + f + "/")
    obsTable = Observations.query_criteria(calib_level= 3, dataproduct_type = 'image',
                                               obs_collection = ["HLA"],
                                        instrument_name = ["WFC3/UVIS"], filters = [f])

    for url in obsTable['dataURL']:
        if isinstance(url, type(obsTable[7]['dataURL'])):
            if url[-5:] == str(".fits"):
                file = "data/" + f + "/" + str(url[50:])
                urllib.request.urlretrieve(url,file)
 
#This code will download all fits images with filters F555W and F606W. 
#For our network were select 174 images from all images
#Training dataset available on https://drive.google.com/open?id=1STjsuQ2_OSGrueLFwVTNp-oO94DAPqe2
#Validation dataset available on https://drive.google.com/open?id=1o2lI_O2e4H8tYUCNUlZOUWNnV8r6chiD
