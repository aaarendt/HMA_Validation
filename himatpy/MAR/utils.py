#!/usr/bin/env python
import os
import sys
import glob
import re
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import xarray as xr
from PyAstronomy import pyasl

import zarr
import s3fs
    
    
import himatpy, himatpy.GRACE_MASCON.pygrace

from himatpy.GRACE_MASCON.pygrace import aggregate_mascons, trend_analysis
from himatpy.MAR.nsidc_download import cmr_search, cmr_download

__author__ = ['Anthony Arendt','Zheng Liu']



def subset_data(input_fn,output_fn,complevel=5,zlib=True,
                keepVars=None, keepDims=[],**kwargs):
    """
    Reads in High Mountain Asia MAR V3.5 Regional Climate Model Output from a zarr store or nc files. 
    Output a "cleaned" xarray dataset.  
    
    Parameters:
    -----------
    input_fn : filename of input netcdf file
    output_fn: filename of output netcdf file or the path to output zarr store
    keepVars : list of variables to keep
    keepDims : list of dimensions to keep
     **kwargs
        Arbitrary keyword arguments related to xarray open_dataset, including chunks, 
    """
	
    # Density of Water
    Ro_w = 1.e3
    
    try:
        ds = xr.open_dataset( input_fn, **kwargs)
    except:
        print("Please provide filename!")
        sys.exit("Exiting...")
            
    # Necessary dimensions 
    needDims = ['TIME','X11_210','Y11_190']
    #if 'SMB' in keepVars: needDims = needDims + ['SECTOR']
    keepDims = keepDims + [tdim for tdim in needDims if tdim not in keepDims]
    
    tt   = ds.TIME
    t0   = tt[0]
    d_tt = ( tt - t0 ) / np.timedelta64(1,'D')
    
    # --- copy data over instead of dropping unwanted data
    #     separation into ice and other sectors are not necessary
    #     using save_agg_mascons_zarr but keep it this way
    #     to work with aggregate_mascons in pygrace.py. 
    #     In future, should change to:
    #     for vn in keepVars: ds_out[vn] = ds[vn]
    ds_out = xr.Dataset()
    
    for vn in ['SMB','RU','SU']:
        ds_out[vn+'_ice']   = ds[vn][:,0]
        ds_out[vn+'_other'] = ds[vn][:,1]
    for vn in ['SF','RF']:
        ds_out[vn]          = ds[vn]
    for vn in ['SW']:
        ds_out[vn+'_ice']   = ds[vn][:,0]

    # -- rename LAT/LON from model specific name to names consistent with this package.
    #    Y/X dimension is needed for save_agg_mascons_zarr. 
    ds_out['lat'      ] = ds['LAT']
    ds_out['long'     ] = ds['LON']
    ds_out = ds_out.rename({'Y11_190':'Y', 'X11_210':'X','TIME':'time'})
    ds_out.time.values = d_tt
    
    encoding = {}
    if zlib:
        comp = dict(zlib=True,  complevel=complevel)
        encoding = {var: comp for var in ds_out.data_vars}
    ds_out.to_netcdf(output_fn,encoding=encoding)
    ds_out.close()
    return output_fn


def nc2zarr(fns,zpath,s3store=True,chunks=None,parallel=True):
    '''
    Convert netcdf files to zarr format and save to local or s3 store
    
    Parameters
    ----------
    fns     : a list of netcdf file names with full path
    zpath   : path to the local or s3 store
    s3store : flag of whether to save to s3 store, boolean
    chunks  : chunks used to read and write data
    parallel: flag to use dask to read files in parallel, boolean
    '''
    # --- remove lat/long from the list of vars to be concatenated.
    with xr.open_mfdataset(fns,parallel=True,chunks=chunks,combine='nested',concat_dim='time') as ds:
        vns = list(ds.data_vars)
    for vn in ['lat','long']:
        if vn in vns: vns.remove(vn)    
        
    with xr.open_mfdataset(fns,chunks=chunks,parallel=parallel, data_vars=vns,combine='nested',concat_dim='time') as ds:
        if s3store:
            fs = s3fs.S3FileSystem(anon=False)
            ds_store = s3fs.S3Map(root=zpath,s3=fs,check=False,create=True)
        else:
            ds_store = zpath
        if chunks is not None: 
            ds = ds.chunk(chunks=chunks) 
        else:
            ds = ds.chunk(chunks={x:ds.chunks[x][0] for x in ds.chunks})
        compressor = zarr.Blosc(cname='zstd', clevel=4)
        encoding = {vname: {'compressor': compressor} for vname in ds.data_vars}
        ds.to_zarr(store=ds_store,encoding=encoding,consolidated=True) 
        
    return 


def save_agg_mascons_zarr(zstore,agg_fn,masked_gdf,zlib=True,complevel=5):
    '''
    save MAR subset data aggregated to GRACE mascons
    
    Parameters
    ----------
    zstore    : full path to zarr store for the MAR subset 
    agg_fn    : file name of aggregated data file with full path
    masked_gdf: geodataframe of the info for mascons in MAR domain
    zlib      : the flag to use zlib to save to netcdf, boolean.
    complevel : level of compression used by zlib. 

    '''
    sdir,sfn = os.path.split(agg_fn)
    if not os.path.exists(sdir): os.mkdir( os.path.abspath(agg_dir) )
    
    geos = [x.bounds for x in masked_gdf['geometry']] 
    mascon_coords = masked_gdf['mascon']
    
        
    with xr.open_zarr(zstore) as ds:
        
        # --- temporary solution for time
        tdoy  = ds['time'].values.astype(float)
        nday  = len(tdoy)
        tyrs  = np.ones(nday).astype(int)*2000
        ii = 1
        while ii<nday:
            if tdoy[ii]-tdoy[ii-1]<-300:
                tyrs[ii:] = tyrs[ii:] + 1
                ii = ii + 364
            else:
                ii = ii + 1
        t_all = np.array([datetime(tYear,1,1) + timedelta(days=x) for tYear,x in zip(tyrs,tdoy)])
        
        # use computed lat/lon to reduce overhead for search over loop
        lat = ds.lat.data.compute()
        lon = ds.long.data.compute()
        
        # find X/Y indices for mascons
        ixys = []
        for geo in geos:
            ixy = []
            flat = np.logical_and(lat>=geo[1],lat<=geo[3])
            flon = np.logical_and(lon>=geo[0],lon<=geo[2])
            fdx  = np.logical_and(flat,flon)
            if fdx.any():
                ixy = np.where(fdx)
            ixys.append( ixy )
            
        # find mascons actually intersect the model domain
        geo_list = []
        for i in range(len(geos)):
            if len(ixys[i])>0:
                geo_list.append(i)
        dslist = []
        
        # aggregate by slicing the dataset with X/Y indice
        for i in geo_list:
            ixy = ixys[i]
            #tds = ds.isel_points(Y=ixy[0],X=ixy[1],dim='points').mean(dim='points')
            tds = ds.isel(Y=xr.DataArray(ixy[0],dims='points'),X=xr.DataArray(ixy[1],dims='points')).mean(dim='points')
            dslist.append( tds )
        nds = xr.concat(dslist,'mascon')
        nds['mascon'] = ('mascon',np.array(mascon_coords)[geo_list])
        
        nds['time'  ] = ('time',t_all)
        
    encoding = {}
    if zlib:
        comp = dict(zlib=True,  complevel=complevel)
        encoding = {var: comp for var in nds.data_vars}    
    nds.to_netcdf(agg_fn,encoding=encoding)
    nds.close()
    return 

def save_agg_mascons(mar_fns,agg_dir,masked_gdf,chunks=None):
    '''
    save MAR subset netcdf data aggregated to GRACE mascons
    
    Parameters
    ----------
    mar_fns:   file names including full path for the MAR subset files
    agg_dir:   output directory of the aggregated data
    masked_gdf: geodataframe of the info for mascons in MAR domain

    Returns
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

        ds = xr.open_dataset(marfn,chunks=chunks)
        agg_data = aggregate_mascons(ds, masked_gdf, scale_factor = 1)
        vns = agg_data['products']
        vdict = dict()
        for iv,vn in enumerate(vns):
            vdict.update( {vn: (('time','mascon'), agg_data['data'][iv].T)} )  
        tdoy  = agg_data['time'].astype(float)
        t_all = np.array([datetime(tYear,1,1) + timedelta(days=x) for x in tdoy]) 
        coord_dict = {'time':t_all,'mascon':agg_data['mascon']}
        dso = xr.Dataset(vdict, coords = coord_dict  )
        dso.to_netcdf(agg_fn)

        outfns.append(agg_fn)
        ds.close()
        
    return outfns

def MAR_trend( agg_fns,vname,t_start='2003-01-07',t_end='2015-12-31'):
    '''
    Read aggregated MAR data and apply trend analysis to a specific field
    
    Parameters
    ----------
    agg_fns:    list of file names with full path to the aggregated MAR data
    vname:      name of MAR field for trend analysis
    t_start:    starting date for trend analysis
    t_end:      end date for trend analysis
    
    Returns
    -------
    out_df:     pandas DataFrame with mascons and p0 to p7 as columns
    
    '''

    # use the following code if xarray is updated on pangeo:
    #     with xr.open_mfdataset(agg_fns,concat_dim='time',combine='nested') as ds:
    with xr.open_mfdataset(agg_fns,concat_dim='time',) as ds:

        mardf = ds.to_dataframe()
        mardf = mardf.reset_index(level='mascon')
        gpdf  = mardf.loc[t_start:t_end].groupby('mascon')

        mascons = list(gpdf.groups.keys())
        pvals = np.zeros(( 8 , len(mascons) ))

        im = 0
        for name, tgroup in gpdf:
            tmar = np.array( list( map( pyasl.decimalYear , tgroup.index.to_pydatetime() ) ) )
            mmwe = tgroup[vname].cumsum()
            pmar = trend_analysis(tmar, series=mmwe,optimization=True)
            pvals[:,im] = pmar
            im = im + 1

        col_names = ['p'+str(i) for i in range(8)]
        out_df = pd.DataFrame(data=pvals.T,columns=col_names)
        out_df['mascon'] = mascons
        out_df = out_df[ ['mascon'] + col_names ]
    
    return out_df



def download_MAR(short_name=None,version='1',time_start=None,time_end=None,polygon=None,filename_filter=None):
    """
    Downloads MAR data from NSIDC via NASA's Common Metadata Repository

    """ 
    
    urls = cmr_search(short_name, version, time_start, time_end,
                      polygon=polygon, filename_filter=filename_filter)

    cmr_download(urls,'Data/MAR')

    
