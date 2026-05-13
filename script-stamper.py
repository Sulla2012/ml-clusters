# This script generates small stamps slightly offset from cluster positions
# These should be used to train the backbone, not the fcrnn

import numpy as np
import astropy.units as u
from astropy.io import fits

from pixell import enmap,utils, reproject

import healpy as hp
import constants as c
import argparse as argp

def make_stamps_rands(freq, ras, decs, map_path):

    width = 5.0*utils.arcmin
    cur_map = enmap.read_map(map_path.format(freq)) #TODO: handle freq
    coords = [(float(dec.to(u.radian).value), float(ra.to(u.radian).value)) for dec, ra in zip(decs, ras)]
    stamps = reproject.thumbnails(cur_map, coords, r=width,  apod = 2*utils.arcmin)

    
    theta, dr = np.random.uniform(0, 2*np.pi, len(ras)), np.random.uniform(5,10, len(ras))
    rand_ras = ras + np.cos(theta)*dr*u.arcmin
    rand_decs = decs + np.sin(theta)*dr*u.arcmin

    rand_coords = [(float(dec.to(u.radian).value), float(ra.to(u.radian).value)) for dec, ra in zip(rand_decs, rand_ras)]
    
    rand_stamps = reproject.thumbnails(cur_map, rand_coords, r=width, apod = 2*utils.arcmin)
    
    return stamps, rand_stamps

def _make_parser() -> argp.ArgumentParser:
    parser = argp.ArgumentParser(
        description="Make offset cluster stamps for backbone training."
    )
    parser.add_argument(
        "cat_path", type=str, help="Path to cluster catalog."
    )
    parser.add_argument(
        "map_path", type=str, help="Path to maps."
    )
    parser.add_argument(
        "out_root",
        type=str,
        help="Path to save the output files."
    )
    parser.add_argument(
        "--offset",
        "-o",
        type=float,
        default=2.0,
        help="Offset scale, in arcmin"
    )
    parser.add_argument(
        "--cut",
        "-c",
        type=int,
        default=None,
    )

    return parser

def main():
    parser = _make_parser()
    args = parser.parse_args()
    cat_path = args.cat_path

    cat_type = cat_path.split(".")[-1]

    if cat_type == "fits":
        cat = fits.open(cat_path)
        ras, decs = cat[1].data['RADeg'], cat[1].data['decDeg']


    elif cat_type == "pksc":
        cluster_catalog=open(cat_path)
        N = np.fromfile(cluster_catalog,count=3,dtype=np.int32)[0]
        cat=np.fromfile(cluster_catalog,count=int(N)*10,dtype=np.float32)
        cat=np.reshape(cat,(N,10))

        x  = cat[:,0];  y = cat[:,1];  z = cat[:,2] # Mpc (comoving)
        R  = cat[:,6] # Mpc

        # Constants
        rho       = 2.775e11*c.omegam*c.h**2       # Msun/Mpc^3
        M         = 4*np.pi/3.*rho*R**3        # this is M200m (mean density 200 times mean) in Msun
        chi       = np.sqrt(x**2+y**2+z**2)    # Mpc
        redshift  = c.zofchi(chi)
        theta,phi = hp.vec2ang(np.column_stack((x,y,z))) # in (not with utils.degree) radians
        ras        = phi
        decs       = np.pi/2. - theta
        z_cut = 0.05
        M_cut = 2.0e14 #TODO: add this to arg parse
        cluster_flags = np.where((redshift > z_cut) & (M > M_cut))[0]
        #Limit on parameters of clusters
        M            = M[cluster_flags]
        redshift     = redshift[cluster_flags]
        decs = decs[cluster_flags] * 180/np.pi
        ras  = ras[cluster_flags] * 180/np.pi
        del cluster_catalog
        del cat
        del x
        del y
        del z
        del R

    else:
        raise ValueError("Error: unsupported catalog type {}".format(cat_type))
    
    offset = args.offset
    offsets = np.random.rand(2, len(ras))*offset*u.arcmin

    ras *= u.deg
    decs *= u.deg

    ras += offsets[0]
    decs += offsets[1]

    if args.cut is not None:
        print(args.cut)
        ras = ras[:args.cut]
        decs = decs[:args.cut]

    stamp_list = []
    rand_list = []

    map_path = args.map_path

    freqs = ["090", "150", "220"]

    for i, freq in enumerate(freqs):
        stamps, rands = make_stamps_rands(freq=freq, ras=ras, decs=decs, map_path=map_path)
        stamp_list.append(stamps)
        rand_list.append(rands)

    stamp_list = np.stack(stamp_list, axis=-1)
    rand_list = np.stack(rand_list, axis=-1)

    np.savez_compressed(args.out_root + "all_clusters_offset.npz", stamp_list) 
    np.savez_compressed(args.out_root + "randoms.npz", rand_list)

if __name__ == "__main__":
    main()