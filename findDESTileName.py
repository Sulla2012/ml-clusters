"""

Example of how to figure out which DES tile given RA, dec coords are in.


"""

import DESTiler
from astropy.io import fits
# Set-up of this is slow, so do only once...
tiler=DESTiler.DESTiler("DES_DR1_TILE_INFO.csv")

# How to get tile name

act_catalog = fits.open('/gpfs/fs0/project/r/rbond/jorlo/cluster_catalogs/DR5_cluster-catalog_v1.0b2.fits')
des_catalog = fits.open('/home/r/rbond/jorlo/dev/ML-clusters/data/des/redmapper_y1a1_public_v6.4_catalog.fits')
act_ra = act_catalog[1].data['RADeg']
act_names = act_catalog[1].data['name']
act_dec = act_catalog[1].data['decDeg']

des_ra = des_catalog[1].data['RA']
des_names = des_catalog[1].data['name']
des_dec = des_catalog[1].data['dec']


for i in range(len(act_ra)):
	RADeg, decDeg= act_ra[i], act_dec[i]
	name = act_names[i]
	name = name.replace(' ', '_')
	print(name)
	tileName=tiler.getTileName(RADeg, decDeg)

	# How to fetch all images for tile which contains given coords
	tiler.fetchTileImages(RADeg, decDeg, '../DESTileImages')

#for i in range(len(des_ra)):
#	RADeg, decDeg= des_ra[i], des_dec[i]
#	name = des_names[i]
#	print(name)
#	tileName=tiler.getTileName(RADeg, decDeg)

        # How to fetch all images for tile which contains given coords
	
#	tiler.fetchTileImages(RADeg, decDeg, '../DESTileImages')

	


