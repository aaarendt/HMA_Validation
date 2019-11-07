#!/usr/bin/env python
import os
import sys
import glob
import re
from datetime import datetime, timedelta
from importlib import reload

import pandas as pd
import numpy as np
import xarray as xr
from collections import OrderedDict
from dask.diagnostics import ProgressBar


import himatpy, himatpy.GRACE_MASCON.pygrace

# --- reload for development purpose 
reload(himatpy)
reload(himatpy.GRACE_MASCON.pygrace)
from himatpy.GRACE_MASCON.pygrace import aggregate_mascons
# --- reload for development purpose 

from himatpy.MAR.nsidc_download import cmr_search, cmr_download

__author__ = ['Anthony Arendt','Zheng Liu']


def get_xr_dataset(zstore=None,files=None,datadir=None, fname=None,multiple_nc=False, 
                        twoDcoords=False, keepVars=None, keepDims=[],**kwargs):
    """
    Reads in High Mountain Asia MAR V3.5 Regional Climate Model Output from a zarr store or nc files. 
    Returns a "cleaned" xarray dataset.  
    :param zstore: path to the store containing the data. 
    :param keepVars: list of variables to keep
     **kwargs
        Arbitrary keyword arguments related to xarray open_zarr or other zarr operation.
    :return: xarray dataset
    """
    # some reformatting is necessary since MAR output does not follow CF conventions

    # first, optional selection of user-specified variables. This has to occur before the coordinate
    # manipulations below
	
    # Density of Water
    Ro_w = 1.e3
    
    if zstore is not None:
        ds   = xr.open_zarr(zstore, **kwargs)
    elif not multiple_nc:
        try:
            ds = xr.open_dataset( fname, **kwargs)
        except:
            print("Please provide filename!")
            sys.exit("Exiting...")
    else:
        if datadir is not None:
            ds = xr.open_mfdataset(os.path.join(datadir, '*.nc'), **kwargs)
        elif files is not None:
            ds = xr.open_mfdataset(files, **kwargs)
        else:
            print('Need either datadir or files for opening multiple netCDF')
            
    # Necessary dimensions 
    needDims = ['TIME','X11_210','Y11_190']
    if 'SMB' in keepVars: needDims = needDims + ['SECTOR']
    keepDims = keepDims + [tdim for tdim in needDims if tdim not in keepDims]
    
    dzsn = ds['DZSN1']
    rosn = ds['ROSN1']
    swe  = (dzsn*rosn).sum('SNOLAY')/Ro_w
    
    tt   = ds.TIME
    t0   = tt[0]
    d_tt = ( tt - t0 ) / np.timedelta64(1,'D')
    if keepVars is not None:
        try:
            products = [x for x in ds]
            deleted_vars = [y for y in products if y not in keepVars+['LAT','LON']]
            ds = ds.drop(deleted_vars)
            try:
                dims = ds.coords
                deleted_dims = [y for y in dims if y not in keepDims]
                ds = ds.drop(deleted_dims)
            except:
                print(keepDims)
                print("List of dimensions to keep does not match variable names in the dataset.")
                sys.exit("Exiting...")
        except:
            print("List of variables to keep does not match variable names in the dataset.")
            sys.exit("Exiting...")
    # add SWE to dataset
    ds   = ds.update({'SWE':swe})
    ds.update
    # rename the dimensions to be lat/long so that other himatpy utilities are consistent with this
    ds = ds.rename({'LON':'long', 'LAT':'lat'})
    ds = ds.rename({'Y11_190':'Y', 'X11_210':'X','TIME':'time'})
    ds.time.values = d_tt
    return ds


def save_agg_mascons(mar_fns,agg_dir,masked_gdf,fulldata=True):
    '''
    save MAR data aggregated to GRACE mascons
    
    Parameters
    ----------
    mar_fns:   file names including full path for the MAR files
    agg_dir:   output directory of the aggregated data
    masked_gdf: geodataframe of the info for mascons in MAR domain
    : params   fulldata=True, if the mar_fns are original full MAR dataset 
               or abbreviated dataset of key variables.

    Return:
    outfns:    output file names
    '''
    nMARs = len(mar_fns)
    if not os.path.exists(agg_dir): os.mkdir( os.path.abspath(agg_dir) )

    outfns = []
    for ifn, marfn in enumerate(mar_fns):

        sdir,sfn = os.path.split(marfn)
        agg_fn = 'agg_'+sfn
        agg_fn = os.path.join(agg_dir,agg_fn)   

        tYear  = int(sfn.split('.')[1])

        print('... aggregating '+sfn+' ...')

        if fulldata:
            ds = get_xr_dataset(fname=marfn,keepVars=[])
        else:
            ds = xr.open_dataset(marfn)
        agg_data = aggregate_mascons(ds, masked_gdf, scale_factor = 1)
        vns = agg_data['products']
        vdict = dict()
        for iv,vn in enumerate(vns):
            vdict.update( {vn: (('time','mascons'), agg_data['data'][iv].T)} )  
        tdoy  = agg_data['time']
        t_all = np.array([datetime(tYear,1,1) + timedelta(days=x) for x in tdoy]) 
        coord_dict = {'time':t_all,'mascons':agg_data['mascons']}
        dso = xr.Dataset(vdict, coords = coord_dict  )
        dso.to_netcdf(agg_fn)

        outfns.append(agg_fn)
        ds.close()
        
    return outfns


def download_MAR(short_name=None,version='1',time_start=None,time_end=None,polygon=None,filename_filter=None):
    """
    Downloads MAR data from NSIDC via NASA's Common Metadata Repository

    """ 
    
    urls = cmr_search(short_name, version, time_start, time_end,
                      polygon=polygon, filename_filter=filename_filter)

    cmr_download(urls,'Data/MAR')

    
