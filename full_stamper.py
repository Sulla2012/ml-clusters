import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D
import os, sys
import pickle as pk
from astropy.convolution import Gaussian2DKernel, convolve
import DESTiler
from astropy.coordinates import SkyCoord
import yaml
from pixell import enmap,utils, reproject, enplot
from scipy import interpolate
import random
from astropy.nddata import block_reduce

with open('/project/r/rbond/jorlo/S18d_202006/selFn/tileDefinitions.yml') as f:
    
    s18d = yaml.load(f)

def normalize_map(imap):
    temp_map = np.zeros(imap.shape)
    for j in range(imap.shape[-1]):
        temp = (imap[...,j]-np.mean(imap[...,j]))/np.std(imap[...,j])
        temp_map[...,j] = temp
                                                        
    return temp_map

def tileFinder(ra, dec, data):
    #Given an RA and Dec in deg, find the S18d tile containing that RA and Dec
    for i, tile in enumerate(data):
        box = tile['RADecSection']
        if box[0] >= box[1]:
            if (360 >= ra >= box[0] or 0 <= ra <box[1]) and box[2]<=dec<= box[3]:
                return tile['tileName']
        if box[0]<=ra<=box[1] and box[2]<=dec<= box[3]:
            return tile['tileName']
    return None

def s18dStamp(ra, dec, data, name, width = 0.5, write = False):
    #Find tile corresponding to RA, Dec
    path = '/project/r/rbond/jorlo/S18d_202006/filteredMaps/'
    tileName = tileFinder(ra, dec, data)
    if tileName == None: return None
    tile = enmap.read_map(path+tileName+'/Arnaud_M2e14_z0p4#'+tileName+'_filteredMap.fits')

    stamp = reproject.postage_stamp(tile, ra, dec, width*60, 0.5)
    #print(stamp)
    if write:
        temp = np.ndarray((2,), buffer=np.array([ra, dec]))    
        stamp.wcs.wcs.crval = temp
        enmap.write_map('./for_tony/mustang2/for_charles/y0_{}.fits'.format(name), stamp)
    return stamp
def y_cutout(ra, dec, scale):
    y_stamp = s18dStamp(ra, dec, s18d, dir, width = 3.5/60)
    if y_stamp is None: return None
    mymin,mymax = 0,y_stamp.shape[1]-1
    X = np.linspace(mymin,mymax,y_stamp.shape[1])
    Y = np.linspace(mymin,mymax,y_stamp.shape[1])

    x,y = np.meshgrid(X,Y)

    f = interpolate.interp2d(x,y,y_stamp[0],kind='quintic')

    Xnew, Ynew = np.linspace(0, y_stamp.shape[1], scale), np.linspace(0, y_stamp.shape[1], scale)
    
    #print(Xnew)
    
    highres = f(Xnew, Ynew)
    return highres

def cutout(ras,decs, scale = 7):
    names = os.listdir('/project/r/rbond/jorlo/datasets/DESTileImages/')
    to_return = []
    for i in range(len(ras)):
        print(i, end = '\r')
        temp = np.zeros((399, 399,6))
        ra, dec= ras[i], decs[i]

        tileName=tiler.getTileName(ra, dec)
        if tileName == None: continue
        for name in names:
            if tileName in name:
                fileName = name[:21]
                break
        bands = ['g', 'r', 'i', 'z', 'Y']
        for j, band in enumerate(bands):
            hi_data = fits.open('/project/r/rbond/jorlo/datasets/DESTileImages/{}_{}.fits.fz'.format(fileName,band))
            header = hi_data[1].header
            w = wcs.WCS(header)
            hdata = hi_data[1].data
            c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            px, py = wcs.utils.skycoord_to_pixel(c, w)

            size = u.Quantity([scale, scale], u.arcmin)
            cutout = Cutout2D(hdata, (px,py), size,  wcs = w).data
            if cutout.shape != (1597, 1597): continue
            temp[...,j] = block_reduce(cutout, 4, func = np.mean)
        y_cut = y_cutout(ra, dec, 399)
        if y_cut is None: continue
        temp[...,5] = y_cut
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

temp_name = str(sys.argv[1])

tiler=DESTiler.DESTiler("DES_DR1_TILE_INFO.csv")

if temp_name.lower() == 'act':
	act_catalog = fits.open('/gpfs/fs0/project/r/rbond/jorlo/cluster_catalogs/DR5_cluster-catalog_v1.0b2.fits')
	ras, decs = act_catalog[1].data['RADeg'], act_catalog[1].data['decDeg']
elif temp_name.lower() == 'des':
	des_catalog = fits.open('/home/r/rbond/jorlo/dev/ML-clusters/data/des/redmapper_sva1_public_v6.3_catalog.fits')
	ras, decs = des_catalog[1].data['RA'], des_catalog[1].data['dec']

elif temp_name.lower() == 'random':

	act_catalog = fits.open('/gpfs/fs0/project/r/rbond/jorlo/cluster_catalogs/DR5_cluster-catalog_v1.0b2.fits')
	des_catalog = fits.open('/home/r/rbond/jorlo/dev/ML-clusters/data/des/redmapper_sva1_public_v6.3_catalog.fits')
	act_ra = act_catalog[1].data['RADeg']
	act_names = act_catalog[1].data['name']
	act_dec = act_catalog[1].data['decDeg']

	des_ra = des_catalog[1].data['RA']
	des_names = des_catalog[1].data['name']
	des_dec = des_catalog[1].data['dec']


	ras, decs = [], []
	i = 0
	names = os.listdir('/project/r/rbond/jorlo/datasets/DESTileImages/')
	while i < 4000:
		tileName = random.choice(names)[:23]
		#print(tileName)
		hdu = fits.open('/project/r/rbond/jorlo/datasets/DESTileImages/{}.fits.fz'.format(tileName))
		header = hdu[1].header
		w = wcs.WCS(header)
		ra_px, dec_px = np.random.uniform(1600, 8000), np.random.uniform(1600,8000)
		coord = wcs.utils.pixel_to_skycoord(ra_px, dec_px, w)
		ra, dec = coord.ra.degree, coord.dec.degree
		#print(ra,dec)
		for j in range(len(act_ra)):
			if np.sqrt((ra/60-act_ra[j]/60)**2+(dec/60-act_dec[j]/60)**2) < 5:
				#print('too close to act')
				continue
		for j in range(len(des_ra)):
			if np.sqrt((ra/60-des_ra[j]/60)**2+(dec/60-des_dec[j]/60)**2) < 5:
				#print('too close to des')
				continue 
		#tileName=tiler.getTileName(ra, dec)
		#print(tileName)
		if tileName == None: 
			print('no tile')
			continue
		try:
			hi_data = open('/project/r/rbond/jorlo/datasets/DESTileImages/{}.fits.fz'.format(tileName))
		except: 
			print('tile not downloaded')
			continue
		ras.append(ra), decs.append(dec)
		i += 1
		print(i,end='\r')
	print(ras, decs)
big_cut = cutout(ras, decs)

np.savez_compressed('/gpfs/fs0/project/r/rbond/jorlo/datasets/cluster-test/{}/full_{}_w_y.npz'.format(temp_name, temp_name), big_cut)
#pk.dump(big_cut, open('/gpfs/fs0/project/r/rbond/jorlo/datasets/cluster-test/{}/full_{}_w_y.pk'.format(temp_name, temp_name), 'wb'))

