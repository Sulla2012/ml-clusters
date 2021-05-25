"""

Class that can figure out which DES tile given RA, dec coords are in.

"""

import os
import sys
import numpy as np
import astropy.table as atpy
import astropy.io.fits as pyfits
from astLib import *
import urllib
import time
import IPython

class DESTiler:
    """A class for relating RA, dec coords to DES tiled survey geometry.
    
    """
    
    def __init__(self, tileInfoCSVPath = "DES_DR1_TILE_INFO.csv"):
        
        self.WCSTabPath=tileInfoCSVPath
        t0=time.time()
        self.setUpWCSDict()
        t1=time.time()
        print("... WCS set-up took %.3f sec ..." % (t1-t0))
    
    
    def setUpWCSDict(self):
        """Sets-up WCS info, needed for fetching images. This is slow (~30 sec) if the survey is large,
        so don't do this lightly.
        
        """
        
        # Add some extra columns to speed up searching
        self.tileTab=atpy.Table().read(self.WCSTabPath)        
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'RAMin'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'RAMax'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'decMin'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileTab)), 'decMax'))
        self.WCSDict={}
        keyWordsToGet=['NAXIS', 'NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 
                        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2']
        for row in self.tileTab:
            newHead=pyfits.Header()
            for key in keyWordsToGet:
                if key in self.tileTab.keys():
                    newHead[key]=row[key]
            # Defaults if missing (needed for e.g. DES)
            if 'NAXIS' not in newHead.keys():
                newHead['NAXIS']=2
            if 'CUNIT1' not in newHead.keys():
                newHead['CUNIT1']='deg'
            if 'CUNIT2' not in newHead.keys():
                newHead['CUNIT2']='deg'
            self.WCSDict[row['TILENAME']]=astWCS.WCS(newHead.copy(), mode = 'pyfits')  
            ra0, dec0=self.WCSDict[row['TILENAME']].pix2wcs(0, 0)
            ra1, dec1=self.WCSDict[row['TILENAME']].pix2wcs(row['NAXIS1'], row['NAXIS2'])
            if ra1 > ra0:
                ra1=-(360-ra1)
            row['RAMin']=min([ra0, ra1])
            row['RAMax']=max([ra0, ra1])
            row['decMin']=min([dec0, dec1])
            row['decMax']=max([dec0, dec1])
    
    
    def getTileName(self, RADeg, decDeg):
        """Returns the DES TILENAME in which the given coordinates are found. Returns None if the coords
        are not in the DES footprint.
        
        """
        raMask=np.logical_and(np.greater_equal(RADeg, self.tileTab['RAMin']), 
                              np.less(RADeg, self.tileTab['RAMax']))
        decMask=np.logical_and(np.greater_equal(decDeg, self.tileTab['decMin']), 
                               np.less(decDeg, self.tileTab['decMax']))
        tileMask=np.logical_and(raMask, decMask)
        if tileMask.sum() == 0:
            return None
        else:
            return self.tileTab[tileMask]['TILENAME'][0] 


    def fetchTileImages(self, RADeg, decDeg, outDir, bands = ['g', 'r', 'i', 'z', 'y'], refetch = False):
        """Fetches DES FITS images for the tile in which the given coords are found. 
        Output is stored under outDir.
                
        """
        
        # Inside footprint check
        raMask=np.logical_and(np.greater_equal(RADeg, self.tileTab['RAMin']), 
                              np.less(RADeg, self.tileTab['RAMax']))
        decMask=np.logical_and(np.greater_equal(decDeg, self.tileTab['decMin']), 
                               np.less(decDeg, self.tileTab['decMax']))
        tileMask=np.logical_and(raMask, decMask)
        if tileMask.sum() == 0:
            return None
        
        os.makedirs(outDir, exist_ok = True)
        for row in self.tileTab[tileMask]:
            for band in bands:
                url=row['FITS_IMAGE_%s' % (band.upper())]
                fileName=outDir+os.path.sep+os.path.split(url)[-1]
                print("... fetching %s-band image from %s" % (band, url))
                if os.path.exists(fileName) == False or refetch == True:
                    urllib.request.urlretrieve(url, filename = fileName)
