import os, sys
import pickle as pk
import yaml
import random

import numpy as np

from scipy import interpolate, ndimage

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from astropy import wcs

from astropy.nddata import Cutout2D
from astropy.nddata import block_reduce, block_replicate

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord

from astLib import astWCS, astImages

import os, sys
import pickle as pk
import yaml
import random

from pixell import enmap,utils, reproject, enplot
from pixell.enmap import sky2pix

from PIL import Image

import glob

def autotiler(surveyMask, wcs, targetTileWidth, targetTileHeight):
    """Given a survey mask (where values > 0 indicate valid area, and 0 indicates area to be ignored), 
    figure out an optimal tiling strategy to accommodate tiles of the given dimensions. The survey mask need
    not be contiguous (e.g., AdvACT and SO maps, using the default pixelization, can be segmented into three
    or more different regions).
    
    Args:
        surveyMask (numpy.ndarray): Survey mask image (2d array). Values > 0 will be taken to define valid 
            area.
        wcs (astWCS.WCS): WCS associated with survey mask image.
        targetTileWidth (float): Desired tile width, in degrees (RA direction for CAR).
        targetTileHeight (float): Desired tile height, in degrees (dec direction for CAR).
    
    Returns:
        Dictionary list defining tiles in same format as config file.
    
    Note:
        While this routine will try to match the target file sizes, it may not match exactly. Also,
        :meth:`startUp.NemoConfig.getTileCoordsDict` will expand tiles by a user-specified amount such that
        they overlap.
    
    """
    
    # This deals with identifying boss vs. full AdvACT footprint maps 
    mapCentreRA, mapCentreDec=wcs.getCentreWCSCoords()    
    skyWidth, skyHeight=wcs.getFullSizeSkyDeg()
    if mapCentreRA < 0.1 and skyWidth < 0.1 or skyWidth > 359.9:
        handle180Wrap=True
    else:
        handle180Wrap=False
    
    segMap=surveyMask
    try:
        numObjects=ndimage.label(segMap, output = segMap)
    except:
        raise Exception("surveyMask given for autotiler is probably too complicated (breaks into > 256 regions) - check your mask and/or config file.")

    # More memory efficient than previous version
    fieldIDs=np.arange(1, numObjects+1, dtype = segMap.dtype)
    maskSections=ndimage.find_objects(segMap)
    tileList=[]
    for maskSection, f in zip(maskSections, fieldIDs):
        yMin=maskSection[0].start
        yMax=maskSection[0].stop-1
        if yMax-yMin < 1000:  # In case of stray individual pixels (e.g., combined with extended sources mask)
            continue
        xc=int((maskSection[1].start+(maskSection[1].stop-1))/2)

        # Some people want to run on full sky CAR ... so we have to avoid that blowing up at the poles
        decMin, decMax=np.nan, np.nan
        deltaY=0
        while np.isnan(decMin) and np.isnan(decMax):
            RAc, decMin=wcs.pix2wcs(xc, yMin+deltaY)
            RAc, decMax=wcs.pix2wcs(xc, yMax-deltaY)
            deltaY=deltaY+0.01

        numRows=int((decMax-decMin)/targetTileHeight)
        if numRows == 0:
            raise Exception("targetTileHeight is larger than the height of the map - edit your config file accordingly.")
        tileHeight=np.ceil(((decMax-decMin)/numRows)*100)/100

        for i in range(numRows):
            decBottom=decMin+i*tileHeight
            decTop=decMin+(i+1)*tileHeight
            xc, yBottom=wcs.wcs2pix(RAc, decBottom)
            xc, yTop=wcs.wcs2pix(RAc, decTop)
            yBottom=int(yBottom)
            yTop=int(yTop)
            yc=int((yTop+yBottom)/2)

            strip=segMap[yBottom:yTop]
            ys, xs=np.where(strip == f)
            xMin=xs.min()
            xMax=xs.max()
            del ys, xs, strip
            stripWidthDeg=(xMax-xMin)*wcs.getXPixelSizeDeg()
            RAMax, decc=wcs.pix2wcs(xMin, yc)
            RAMin, decc=wcs.pix2wcs(xMax, yc)
            numCols=int(stripWidthDeg/targetTileWidth)
            tileWidth=np.ceil((stripWidthDeg/numCols)*100)/100
            #assert(tileWidth < targetTileWidth*1.1)

            stretchFactor=1/np.cos(np.radians(decTop))
            numCols=int(stripWidthDeg/(targetTileWidth*stretchFactor))
            for j in range(numCols):
                tileWidth=np.ceil((stripWidthDeg/numCols)*100)/100
                RALeft=RAMax-j*tileWidth
                RARight=RAMax-(j+1)*tileWidth
                if RALeft < 0:
                    RALeft=RALeft+360
                if RARight < 0:
                    RARight=RARight+360
                # HACK: Edge-of-map handling
                if handle180Wrap == True:
                    if RARight < 180.01 and RALeft < 180+tileWidth and RALeft > 180.01:
                        RARight=180.01
                # NOTE: floats here to make tileDefinitions.yml readable
                tileList.append({'tileName': '%d_%d_%d' % (f, i, j),
                                 'RADecSection': [float(RARight), float(RALeft), float(decBottom), float(decTop)]})

    return tileList

def getTileCoordsDict(tileList,wcs_mask,tileOverlapDeg):
    """Construct a dictionary that describes how a large map is broken up into smaller tiles
    (see :ref:`Tiling` for information on the relevant configuration file parameters).
    Returns:
        A dictionary indexed by tile name, where each entry is a dictionary containing information
        on pixel coordinates of each tile within the larger map, and the WCS of each tile.
    """
    # Spin through a map, figuring out the actual coords to clip based on the tile definitions
    clipCoordsDict={}

    # We can take any map, because we earlier verified they have consistent WCS and size
    wcs=wcs_mask
    # Tiled - this takes about 4 sec
    tileNames=[]
    coordsList=[]
    for tileDict in tileList:
            ra0, ra1, dec0, dec1=tileDict['RADecSection']
            x0, y0=wcs.wcs2pix(ra0, dec0)
            x1, y1=wcs.wcs2pix(ra1, dec1)
            xMin=min([x0, x1])
            xMax=max([x0, x1])
            yMin=min([y0, y1])
            yMax=max([y0, y1])
            coordsList.append([xMin, xMax, yMin, yMax])
            tileNames.append(tileDict['tileName'])
            
    # Define clip regions in terms of pixels, adding overlap region
    tileOverlapDeg=tileOverlapDeg
    mapData=np.ones([wcs.header['NAXIS2'], wcs.header['NAXIS1']], dtype = np.uint8)
    for c, name, tileDict in zip(coordsList, tileNames, tileList):
        y0=c[2]
        y1=c[3]
        x0=c[0]
        x1=c[1]
        ra0, dec0=wcs.pix2wcs(x0, y0)
        ra1, dec1=wcs.pix2wcs(x1, y1)
        # Be careful with signs here... and we're assuming approx pixel size is ok
        if x0-tileOverlapDeg/wcs.getPixelSizeDeg() > 0:
            ra0=ra0+tileOverlapDeg
        if x1+tileOverlapDeg/wcs.getPixelSizeDeg() < mapData.shape[1]:
            ra1=ra1-tileOverlapDeg
        if y0-tileOverlapDeg/wcs.getPixelSizeDeg() > 0:
            dec0=dec0-tileOverlapDeg
        if y1+tileOverlapDeg/wcs.getPixelSizeDeg() < mapData.shape[0]:
            dec1=dec1+tileOverlapDeg
        if ra1 > ra0:
            ra1=-(360-ra1)
        clip=astImages.clipUsingRADecCoords(mapData, wcs, ra1, ra0, dec0, dec1)

        # This bit is necessary to avoid Q -> 0.2 ish problem with Fourier filter
        # (which happens if image dimensions are both odd)
        # I _think_ this is related to the interpolation done in signals.fitQ
        if (clip['data'].shape[0] % 2 != 0 and clip['data'].shape[1] % 2 != 0) == True:
            newArr=np.zeros([clip['data'].shape[0]+1, clip['data'].shape[1]])
            newArr[:clip['data'].shape[0], :]=clip['data']
            newWCS=clip['wcs'].copy()
            newWCS.header['NAXIS1']=newWCS.header['NAXIS1']+1
            newWCS.updateFromHeader()
            testClip=astImages.clipUsingRADecCoords(newArr, newWCS, ra1, ra0, dec0, dec1)
            # Check if we see the same sky, if not and we trip this, we need to think about this more
            assert((testClip['data']-clip['data']).sum() == 0)
            clip['data']=newArr
            clip['wcs']=newWCS

        # Storing clip coords etc. so can stitch together later
        # areaMaskSection here is used to define the region that would be kept (takes out overlap)
        ra0, dec0=wcs.pix2wcs(x0, y0)
        ra1, dec1=wcs.pix2wcs(x1, y1)
        clip_x0, clip_y0=clip['wcs'].wcs2pix(ra0, dec0)
        clip_x1, clip_y1=clip['wcs'].wcs2pix(ra1, dec1)
        clip_x0=int(round(clip_x0))
        clip_x1=int(round(clip_x1))
        clip_y0=int(round(clip_y0))
        clip_y1=int(round(clip_y1))
        if name not in clipCoordsDict:
            clipCoordsDict[name]={'clippedSection': clip['clippedSection'], 'header': clip['wcs'].header,
                                  'areaMaskInClipSection': [clip_x0, clip_x1, clip_y0, clip_y1]}
    return clipCoordsDict


def _make_jpg(path, box, freqs, normalize = True):
    to_return = np.empty(len(freqs), dtype=object)
    freqs = sorted(freqs)
    files = glob.glob(path)
    for i, freq in enumerate(freqs):
        path = [path for path in files if freq in path]
        if len(path) > 1:
            raise PathError("Err: multiple paths match freq {}".format(freq))

        cur_map = enmap.read_map(path[0], box = box)
        wcs = cur_map.wcs
        cur_map = cur_map[0] #select temperature channel
        if normalize:
            cur_map = normalize_map(cur_map)
        if type(cur_map) == int: #error handling from normalize_map
            return -1
        to_return[i] = cur_map
    to_return = np.stack(to_return, axis = 0)

    to_return = np.ascontiguousarray(to_return.transpose(1,2,0))

    return to_return, wcs

def make_stamp(path, box, freqs, normalize = True):
    freqs = sorted(freqs)
    files = glob.glob(path)

    path = [path for path in files if freqs[0] in path]
    if len(path) > 1:
           raise PathError("Err: multiple paths match freq {}".format(freq))

    base_map = enmap.read_map(path[0], box = box)
    wcs = base_map.wcs

    if "websky" in path[0]:
        cur_map = base_map
    else:
        cur_map = base_map[0]
    if normalize:
        cur_map = normalize_map(cur_map)

    freq_maps = np.empty([len(freqs), cur_map.shape[0], cur_map.shape[1]]) #Gonna have to roll this in the loader
    freq_maps[0] = cur_map

    for i, freq in enumerate(freqs):
        if i == 0: continue 
        path = [path for path in files if freq in path]
        if len(path) > 1:
            raise PathError("Err: multiple paths match freq {}".format(freq))
        
        cur_map = enmap.read_map(path[0], box = box)
        wcs = cur_map.wcs
        cur_map = cur_map[0] #select temperature channel
        if normalize:
            cur_map = normalize_map(cur_map)
        if type(cur_map) == int: #error handling from normalize_map
            return -1
        freq_maps[i] = cur_map
    to_ret = enmap.enmap(freq_maps, wcs=base_map.wcs) 

    return to_ret, wcs

def make_jpg(path, box):
    '''
    Function which makes a np array which can be interpreted as a jpg
    
    Input: 
        path: str, path to where the maps we're making these jpgs from lives
        box: array([[ra_min, dec_min],[ra_max, dec_max]]) the box defining the edges of the jpg. Can be in either
             degrees or radians, but if in degrees must have attached astropy units i.e. u.deg
    Output:
        to_return: array([px, py, 3]) an array of stamps in the three frequencies, all at location box. 
        cur_map.wcs: for performance reasons we just open and close the maps here, later functions require the wcs info so it gets returned as well
    '''
    
    #Just lists the frequencies of the ACT maps. Note the ACT convention of 090 for 90GHz
    #freqs = ['090', '150', '220']
    freqs = ['090', '150'] #Not sure what's up here but there doesn't seem to be a f220 map
    to_return = []
    
    for i, freq in enumerate(freqs):
        #For each frequeny, stamp out the box we're interested in. enmap.read_map reads in the map, reading only from inside
        #box, and returns a pixell map. [0] just selects only the data
        cur_map = enmap.read_map(path + 'act_planck_s08_s22_f{}_daynight_map.fits'.format(freqs[i]), box=box)
        if i == 1: cur_wcs = cur_map.wcs      
        cur_map = normalize_map(cur_map[0])
        if type(cur_map) == int:
            return -1
        else:
            to_return.append(cur_map)
    to_return = np.stack(to_return, axis=0)
    
    #Just some dimension rearanging 
    to_return = np.ascontiguousarray(to_return.transpose(1,2,0))
    return to_return, cur_wcs

def normalize_map(imap):
    '''
    Normalizes a map in some way. Right now computes the average of the map and the STD, and then
    subtracts off the mean and divides by the std, returning a map with 0 mean and 1 std. We want to consider other
    methods for normalizing. Each frequency is normalized independantly (may want to reconsider this)
    
    Input:
        imap: array, the input map as an np array
    Ouptut:
        temp_map: array(imap.shap), the normalized input map
    
    '''
    temp_map = np.zeros(imap.shape)
    for j in range(imap.shape[-1]):
        #For each frequency, normalize the map
        std = np.std(imap[...,j])
        if std ==0: 
            return -1
        else:
            temp = (imap[...,j]-np.mean(imap[...,j]))/np.std(imap[...,j])
            temp_map[...,j] = temp 

    return temp_map


def freq_cutout(ra, dec, freq_map, scale=399, width = 3.5*utils.arcmin):

    #A function for upscaling act maps, can ignore
    stamp = freq_stamp(ra, dec, freq_map, width = width)
    
    highres = block_replicate(stamp, scale)
    if stamp is None: return None
    mymin,mymax = 0,stamp.shape[1]-1
    X = np.linspace(mymin,mymax,stamp.shape[1])
    Y = np.linspace(mymin,mymax,stamp.shape[1])

    x,y = np.meshgrid(X,Y)

    f = interpolate.interp2d(x,y,stamp[0],kind='linear')

    Xnew, Ynew = np.linspace(0, stamp.shape[1], scale), np.linspace(0, stamp.shape[1], scale)

    #print(Xnew)

    highres = f(Xnew, Ynew)
    return highres

def freq_stamp(ra, dec, freq_map, width = 3.5*utils.arcmin):
    ra, dec = ra*u.deg, dec*u.deg
    stamp = reproject.thumbnails(freq_map, ( dec.to(u.radian).value,ra.to(u.radian).value), r=width)
    
    return stamp

def cutout(ras,decs, freq_map_090, freq_map_150, freq_map_220, scale = 10):

    #Makes combined ACT+DES images, can ignore

    names = os.listdir('/project/r/rbond/jorlo/datasets/DESTileImages/')
    to_return = []
    for i in range(len(ras)):
        print(i, end = '\r')
        temp = np.zeros((380,380,8))
        ra, dec= ras[i], decs[i]

        tileName=tiler.getTileName(ra, dec)
        if tileName == None: continue
        for name in names:
            if tileName in name:
                fileName = name[:21]
                break
        bands = ['g', 'r', 'i', 'z', 'Y']
        for j, band in enumerate(bands):
            print('J: ', j)
            hi_data = fits.open('/project/r/rbond/jorlo/datasets/DESTileImages/{}_{}.fits.fz'.format(fileName,band))
            header = hi_data[1].header
            w = wcs.WCS(header)
            hdata = hi_data[1].data
            c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            px, py = wcs.utils.skycoord_to_pixel(c, w)
            size = u.Quantity([scale, scale], u.arcmin)
            print(size)
            cutout = Cutout2D(hdata, (px,py), size,  wcs = w).data
            print(cutout.shape)
            if cutout.shape != (2281, 2281): continue
            temp[...,j] = block_reduce(cutout, 6, func = np.mean)
            

        freqs = ['090', '150', '220']
        
        for i, freq in enumerate(freqs):
            if freqs[i] == '090': freq_cut = freq_cutout(ra, dec, freq_map_090, scale = 380, width = scale/2*utils.arcmin)
            elif freqs[i] == '150': freq_cut = freq_cutout(ra, dec, freq_map_150, scale = 380, width = scale/2*utils.arcmin)
            elif freqs[i] == '220': freq_cut = freq_cutout(ra, dec, freq_map_220, scale = 380, width = scale/2*utils.arcmin)
            if freq_cut is None: continue
            temp[...,5+i] = freq_cut
        temp = normalize_map(temp)
        """
        if np.all(to_return) == 1:
            to_return = temp
        elif len(to_return.shape) == 3:
            print('here')
            to_return = np.stack((to_return, temp), axis = -1)
        else:
            to_return = np.append(to_return, temp, axis = -1)
        """
        to_return.append(temp)
    to_return = np.stack(to_return, axis=0)
    return to_return

def make_mask(image, ras, decs, box, cur_wcs, size = 2.4, jpg=False):
    #Function which makes masks corresponding to clusters in a image. 
    if jpg: 
        mask = np.zeros(image[...,0].shape)

    else:
        mask = np.zeros(image[0].shape)
    box /= utils.degree
    min_ra, max_ra, min_dec, max_dec = box[0][1], box[1][1], box[0][0], box[1][0] 
   
    in_image = np.where((min_ra < ras) & (ras < max_ra) & (min_dec < decs) & (decs < max_dec))[0]

    if len(in_image) == 0:
        return mask
 
    for i in range(len(in_image)):
        cur_cluster = in_image[i]
        cur_center = SkyCoord(ras[cur_cluster], decs[cur_cluster], unit = "deg")
        x,y = wcs.utils.skycoord_to_pixel(cur_center, cur_wcs)
        
        x,y = np.round(x), np.round(y)

        pix_size = wcs.utils.proj_plane_pixel_scales(cur_wcs)[0] * 60

        r = size/2/pix_size
        
        xx, yy = np.meshgrid(np.linspace(0, mask.shape[1]-1, mask.shape[1]), np.linspace(0, mask.shape[0]-1, mask.shape[0]))    
        r_mask = (xx-x)**2 + (yy-y)**2 < r**2

        mask += r_mask*(i+1)
        
        doubled_mask = mask > i+1 #Un-double counts areas where clusters overlap
        mask -= doubled_mask*(i+1)
    
    return mask

def make_mask_wise(image, loc, res, size = 2.4):
    #Function which makes masks corresponding to clusters in a image. 
    mask = np.zeros(image.shape)
    
    x,y = loc
    x,y = np.round(x), np.round(y)

    r = size/2/res

    xx, yy = np.meshgrid(np.linspace(0, mask.shape[1]-1, mask.shape[1]), np.linspace(0, mask.shape[0]-1, mask.shape[0]))
    r_mask = (xx-x)**2 + (yy-y)**2 < r**2

    mask += r_mask
    return(mask)

