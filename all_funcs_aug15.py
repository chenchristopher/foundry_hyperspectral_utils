#################
#    imports    #
#################
import os
import sys
import scipy as sp
import scipy.misc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import h5py
import numpy as np

import hyperspy.api as hs
import hyperspy_gui_ipywidgets as hsgui


##############
#    help    #
##############

def printname(name):
    #executable for hdf5.visit
    print(name)


def sortfnames(dir='.'):
    """
    Returns str arrays of all files in the given directory.

    Updated by Chris to take a path from the keyword argument.

    Keyword Args:
        dir (str): Path to the directory of interest.

    Returns:
        tuple:  tuple containing lists of strings for each of the following
                file types
                    *sync_raster_scan.h5    sync_raster_scan_h5
                    *hyperspec_cl.h5        hyperspec_cl_h5
                    *.h5                    HDF5 files
                    *.txt                   text files
                    *.sif                   sif files
    """
    a = os.listdir(os.path.abspath(dir))
    txts = [i for i in a if i.endswith('.txt')]
    sifs = [i for i in a if i.endswith('.sif')]
    hdf5s = [i for i in a if i.endswith('.h5')]
    clmaps = [i for i in a if i.endswith('hyperspec_cl.h5')]
    clims = [i for i in a if i.endswith('sync_raster_scan.h5')]
    return (clmaps, clims, hdf5s, txts, sifs)

def sizerasterscans(path,flist):
    for fname in flist:
        file  = h5py.File(path+fname, 'r')
        print(fname,'|',
              file['measurement/sync_raster_scan/adc_map'].shape
             )
def sizespecims (path,flist):
    for fname in flist:
        file  = h5py.File(path+fname, 'r')
        print(fname,'|',
              file['measurement/hyperspec_cl/spec_map'].shape
             )

def findstrinlist(listtobesearched,tobefound):
#returns index of strings in listtobesearched containing tobefound
    ind = [n for n,l in enumerate(listtobesearched) if tobefound in l]
    return ind

def search_attrs(items_dict,word):
#returns all items in dict were word is part of the key,
#word expected to be low kays key can be capitals
    new_dict=[(key,value) for key, value in items_dict.items() if word in key.lower()]
    return new_dict

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray##############
#    read    #
##############

def x_import(res_dir='/Volumes/GoogleDrive/My Drive/CL data/calibs/',
             fname='560 150.asc'):
    name=res_dir +fname
    temp = np.loadtxt(name,delimiter=',')
    delta=temp[:,0]
    return delta

def getshortlist(path,fname,remcon_dict,daq_dict,data_dict,andor_dict,acton_dict,**kwargs) :
#add calib calc
    #calcs
    size_in_pix=[data_dict['Nv'],data_dict['Nh']]
    pix_in_V = (data_dict['dh'],data_dict['dv'])
    pix_in_nm = np.multiply(pix_in_V,4.3*2.54e7/(remcon_dict['magnification'])/20)
    size_in_nm = size_in_pix * pix_in_nm
    #make list
    #shortlist ={'dir': path, 'filename' : fname }
    remcon_keys = ['magnification','kV','WD','select_aperture']
    remcon_short = dict((k, remcon_dict[k]) for k in remcon_keys if k in remcon_dict)
    daq_keys = ['adc_chan_names','adc_oversample','adc_rate','ctr_chan_names','dac_rate',]
    daq_short = dict((k, daq_dict[k]) for k in daq_keys if k in daq_dict)
    data_keys = ['description','frame_time',
                 'line_time','n_frames','pixel_time','total_time']
    data_short = dict((k, data_dict[k]) for k in data_keys if k in data_dict)
    data_short['size_in_pix']=[data_dict['Nv'],data_dict['Nh']]
    data_short['pix_in_V'] = (data_dict['dh'],data_dict['dv'])
    data_short['pix_in_nm']= np.multiply(pix_in_V,4.3*2.54e7/(remcon_dict['magnification'])/20)
    data_short['size_in_nm'] = size_in_pix * pix_in_nm

    andor_keys = ['acc_time','em_gain','exposure_time','readout_shape',]
    andor_short = dict((k, andor_dict[k]) for k in andor_keys if k in andor_dict)
    acton_keys = ['center_wl','entrance_slit','grating_name']
    acton_short = dict((k, andor_dict[k]) for k in acton_keys if k in andor_dict)


    shortlist = { **remcon_short,**daq_short,
                 **data_short,**andor_short,**acton_short}

    if 'printing' in kwargs.keys() :
        if kwargs['printing'] == 'None' :
            return shortlist
    else:
        s="\n".join("=".join((str(k),str(v))) for k,v in shortlist.items())
        #fig=plt.figure(figsize=(5,6))
        #ax=plt.text(0 , 0, s)
        #plt.axis('off')
        return s

def load_SI(path,fname,outputmode = 'dictofdicts',**kwargs) :
#usage:
# dod=load_SI(path,fname,outputmode = 'dictofdicts') returns dict of dicts with hdf5 componnents
#
# se_im,ctr_im,spec_im,shortlist=load_SI(path,fname,outputmode = 'short')

# se_im,ctr_im,spec_im=load_SI(path,fname,outputmode = 'images')

# spec_im,shortlist=load_SI(path,fname,outputmode = 'SI')
#
#se_im,ctr_im,spec_im, remcon_dict,daq_dict,data_dict,andor_dict,acton_dict,shortlist,sum_spec,sum_im = load_SI(path,fname)
#
#
#
    #load h5 hyperspec_cl file componenets as lists
    file1  = h5py.File(os.path.join(path,fname), 'r')
    #file1.visit(printname)

    remcon=file1['hardware/sem_remcon/settings']
    remcon_dict = dict(remcon.attrs.items())
    andor=file1['hardware/andor_ccd/settings']
    andor_dict = dict(andor.attrs.items())
    acton=file1['hardware/acton_spectrometer/settings']
    acton_dict = dict(acton.attrs.items())
    data=file1['measurement/hyperspec_cl/settings']
    data_dict = dict(data.attrs.items())
    daq=file1['hardware/sync_raster_daq/settings']
    daq_dict = dict(daq.attrs.items())

    shortlist = getshortlist(path,fname,remcon_dict,daq_dict,data_dict,andor_dict,acton_dict,printing='None')

    #load h5 hyperspec_cl file componenets as lists
    se_imgs=file1['measurement/hyperspec_cl/adc_map']
    ctr_imgs=file1['measurement/hyperspec_cl/ctr_map']
    spec_imgs=file1['measurement/hyperspec_cl/spec_map']
    wavelength=file1['measurement/hyperspec_cl/wls']

    #remove singular dimetions
    se_im = np.squeeze(se_imgs)
    ctr_im = np.squeeze(ctr_imgs)
    spec_im = np.squeeze(spec_imgs)

    spec_im[0,0,:]=spec_im[1,1,:]

    SI_size = spec_im.shape

    sum_spec = np.sum(a=spec_im,axis=(0,1))
    sum_im = np.sum(a=spec_im,axis=2)

    if outputmode == 'dictofdicts':
        dict_of_dicts = {'filename':fname,
                        'path':path,
                        'remcon': remcon_dict,
                        'daq': daq_dict,
                        'data':data_dict,
                        'andor':andor_dict,
                        'acton':acton_dict,
                        'summary':shortlist,
                        'se': se_im,
                        'cntr' : ctr_im,
                        'SI': spec_im,
                        'wavelength': wavelength,
                        'sum1D': sum_spec,
                        'sum2D':sum_im}
        return dict_of_dicts
    elif outputmode== 'short':
        return se_im,ctr_im,spec_im,shortlist
    elif outputmode == 'images':
        return se_im,ctr_im,spec_im
    elif outputmode == 'SI':
        return spec_im, shortlist
    else:
        return se_im,ctr_im,spec_im, remcon_dict,daq_dict,data_dict,andor_dict,acton_dict,shortlist,sum_spec,sum_im


def load_RS(path,fname,**kwargs) :
#usage:
# dod=load_SI(path,fname,outputmode = 'dictofdicts') returns dict of dicts with hdf5 componnents
#
# se_im,ctr_im,spec_im,shortlist=load_SI(path,fname,outputmode = 'short')

# se_im,ctr_im,spec_im=load_SI(path,fname,outputmode = 'images')

# spec_im,shortlist=load_SI(path,fname,outputmode = 'SI')
#
#se_im,ctr_im,spec_im, remcon_dict,daq_dict,data_dict,andor_dict,acton_dict,shortlist,sum_spec,sum_im = load_SI(path,fname)
#
#
#
    #load h5 hyperspec_cl file componenets as lists
    file1  = h5py.File(os.path.join(path, fname), 'r')
    #file1.visit(printname)
    remcon=file1['hardware/sem_remcon/settings']
    remcon_dict = dict(remcon.attrs.items())
    andor=file1['hardware/andor_ccd/settings']
    andor_dict = dict(andor.attrs.items())
    acton=file1['hardware/acton_spectrometer/settings']
    acton_dict = dict(acton.attrs.items())
    data=file1['measurement/sync_raster_scan/settings']
    data_dict = dict(data.attrs.items())
    daq=file1['hardware/sync_raster_daq/settings']
    daq_dict = dict(daq.attrs.items())
    shortlist = getshortlist(path,fname,remcon_dict,daq_dict,data_dict,andor_dict,acton_dict,printing='None')

    #load raster scan h5 file componenets as lists
    se_imgs=file1['measurement/sync_raster_scan/adc_map']
    ctr_imgs=file1['measurement/sync_raster_scan/ctr_map']

    #remove singular dimetions
    se_im = np.squeeze(se_imgs)
    ctr_im = np.squeeze(ctr_imgs)

    if 'outputmode' in kwargs.keys() :
        if kwargs['outputmode'] == 'dictofdicts':
            dict_of_dicts = {'filename':fname,
                             'path':path,
                             'remcon': remcon_dict,
                             'daq': daq_dict,
                             'data':data_dict,
                             'andor':andor_dict,
                             'acton':acton_dict,
                             'summary':shortlist,
                              'se': se_im,
                              'cntr' : ctr_im}
            return dict_of_dicts
        elif kwargs['outputmode']== 'short':
             return se_im,ctr_im,shortlist
        elif kwargs['outputmode'] == 'images':
             return se_im,ctr_im
    else:
        return se_im,ctr_im, remcon_dict,daq_dict,data_dict,andor_dict,acton_dict,shortlist,sum_spec,sum_im


##############
#    disp    #
##############

def plot_si_bands(spec_im,cpath,fname,*args,**kwargs):
#revize
#plots image bands defined by *args,
#each arg is (low bound, high bound, colormap)
#Usage:
#       plot_si_bands(spec_im1,cpath,fname,
#              (0,1600,'plasma'),
#              (180,220,'Blues'),
#              (220,350,'Blues'),
#              (400,600,'Greens'),
#              (550,750,'plasma'),
#              (750,1000,'plasma'),
#              (500,1000,'plasma'),
#              (1100,1500,'Blues'),
#              percentile = 3,
#              no_of_cols = 4)

    f_h=plt.figure()
    # disp defaults

    if 'percentile' in kwargs.keys() :
        percentile = kwargs['percentile']
    else:
        percentile = 5

    if 'no_of_cols' in kwargs.keys() :
        no_of_cols = kwargs ['no_of_cols']
    else:
        no_of_cols = 3

    no_of_bands = len(args)
    no_of_rows = no_of_bands//no_of_cols

    gs=gspec.GridSpec(no_of_rows,no_of_cols)

    subplotindxs = np.reshape(np.arange(0,no_of_cols*no_of_rows),[no_of_rows,no_of_cols])

    i=0;
    for arg in args:
        band=np.sum(a=spec_im[:,:,arg[0]:arg[1]],axis=2)
        coord=np.where(subplotindxs==i)
        plt.subplot(gs[coord[0][0],coord[1][0]])
        plt.title(arg)
        plt.imshow(band,cmap=arg[2],
               vmin=np.percentile(band,percentile),
               vmax=np.percentile(band,100-percentile))
        plt.colorbar()
        i = i +1

    f_h.suptitle(cpath + fname)

    return f_h


def plotmosaic(data,sizex,sizey,percentile=1,printing='',display='keep',**kwargs):
#. plots mosaic of sizex by sizey spectral image slices.

    no_of_tiles=sizex*sizey
    # taking care of missing or extra tiles
    imsizex,imsizey,imsizez=data['SI'].shape
    pixperslice=imsizez//no_of_tiles
    extras=sp.remainder(imsizez,no_of_tiles)
    #print(imsizex,imsizey,imsizez,no_of_tiles,pixperslice,extras)

    #rebin the original array
    newd=bin_ndarray(data['SI'][:,:,0:imsizez-extras],[imsizex,imsizey,no_of_tiles])

    # make tiled image
    line=[]
    impo=[]
    for j in  range(sizey):
        line=newd[:,:,0+j*sizex]
        for i in range(1,sizex):
            imp=newd[:,:,i+j*sizex]
            line=np.concatenate((line,imp), axis=1)
        if j==0 :
            impo=line
        else :
            impo=np.concatenate((impo, line), axis=0)

    #plot
    plt.figure(figsize=[2.5*sizex*imsizex/imsizey,2*sizey*imsizey/imsizey])
    ax=plt.imshow(impo,
          vmin=np.percentile(impo,1),
          vmax=np.percentile(impo,99),
          cmap='gray')
    plt.colorbar()

    #make labels
    tt=np.arange(pixperslice//2,(imsizez-extras),pixperslice)
    #print(tt)
    aa=np.array(data['wavelength'])[tt]
    aas = ["%6.0f" % n for n in aa]

    for i in range(no_of_tiles):
        col=np.remainder(i,sizex)
        row=i//sizex
        plt.text(col*imsizex,(row+0.95)*imsizey, aas[i]+'nm',
                      color='orange', fontsize=12)

    plt.xticks([])
    plt.yticks([])
    plt.title(data['filename'])
    plt.axis('tight')

    if printing=='pdf':
        outname =data['path'] + data['filename'][:-5]+'mosaic.pdf'
        plt.savefig(outname,dpi=150, orientation='landscape')

    if display!='keep':
        plt.close()
        return

    return impo,aas##########################
#    new CL display    #
##########################

def new_si_summary(data,percentile=1,printing='',display='keep',**kwargs):
#plots summary of SI with its images in a fourplot
#usage:
#      plot_si_sprint(se_im,ctr_im,spec_im,cpath,fname
#                     percentile=3)


    f_h=plt.figure(figsize=(16,6))

    f_h.suptitle(data['path'] + data['filename'])

   #analogs
    ax_ad=[(3,5,2),(3,5,3),(3,5,4),(3,5,5)]
    i=0
    for nrows,ncols,plotnum in ax_ad:
        plt.subplot(nrows,ncols,plotnum),
        plt.imshow(data['se'][:,:,i],cmap='pink',
               vmin=np.percentile(data['se'][:,:,i],percentile),
               vmax=np.percentile(data['se'][:,:,i],100-percentile))
        plt.colorbar()
        i=i+1

    #ctrs
    ax_ctr=[(3,5,7), (3,5,8), (3,5,9),(3,5,10)]
    i=0
    for nrows,ncols,plotnum in ax_ctr:
        plt.subplot(nrows,ncols,plotnum),
        plt.imshow(data['cntr'][:,:,i],cmap='plasma',
               vmin=np.percentile(data['cntr'][:,:,i],percentile),
               vmax=np.percentile(data['cntr'][:,:,i],100-percentile))
        plt.colorbar()
        i=i+1

    #txt
    plt.subplot(1,6,1)
    s="\n".join(" = ".join((str(k),str(v))) for k,v in data['summary'].items())
    plt.text(-1.0 , 0, s, fontsize=8)
    plt.axis('off')

    #sumim
    plt.subplot(3,5,12),
    no_of_specs = np.sum(a=data['sum2D']!=0,axis=(0,1))
    plt.imshow(data['sum2D'],cmap='plasma',
           vmin=np.percentile(data['sum2D'],percentile),
           vmax=np.percentile(data['sum2D'],100-percentile))
    plt.colorbar()

    #sumspecs
    plt.subplot(3,5,(13,15)),
    plt.plot(data['wavelength'],data['sum1D']/no_of_specs)

    if printing=='pdf':
        outname =data['path'] + data['filename'][:-5]+'.pdf'
        plt.savefig(outname,dpi=150, orientation='landscape')

    if display!='keep':
        plt.close()
        return
    else:
            return f_h

    return f_h

def new_rs_summary(data,percentile=1,printing='',display='keep',**kwargs):
#plots summary of SI with its images in a fourplot
#usage:
#

    f_h=plt.figure(figsize=(11,6),dpi=150)

    f_h.suptitle([data['path'] + data['filename']])

    #analogs
    ax_ad=[(2,5,2),(2,5,3),(2,5,4),(2,5,5)]
    i=0
    for nrows,ncols,plotnum in ax_ad:
        plt.subplot(nrows,ncols,plotnum)
        plt.imshow(data['se'][:,:,i],cmap='pink',
               vmin=np.percentile(data['se'][:,:,i],percentile),
               vmax=np.percentile(data['se'][:,:,i],100-percentile))
        plt.colorbar()
        plt.axis('off')
        plt.title('ai'+ str(i))
        i=i+1

    #ctrs
    ax_ctr=[(2,5,7), (2,5,8), (2,5,9),(2,5,10)]
    j=0
    for nrows,ncols,plotnum in ax_ctr:
        plt.subplot(nrows,ncols,plotnum)
        plt.imshow(data['cntr'][:,:,j],cmap='plasma',
               vmin=np.percentile(data['cntr'][:,:,j],percentile),
               vmax=np.percentile(data['cntr'][:,:,j],100-percentile))
        plt.colorbar()
        plt.axis('off')
        plt.title('cntr'+ str(j))
        j=j+1

    #txt
    plt.subplot(1,6,1)
    s="\n".join(" = ".join((str(k),str(v))) for k,v in data['summary'].items())
    plt.text(-1.0 , 0, s, fontsize=8)
    plt.axis('off')

    if printing == 'pdf':
        outname = data['path'] + data['filename']+'.pdf'
        plt.savefig(outname,dpi=150, orientation='landscape')

    if display!='keep':
        plt.close()

    return f_h
##############
#    sums    #
##############
def make_image_summaries(clims,path):
    #read all clims in dir and plot summary saves to pdf and delete the figs

    for fname in clims:
        try:
            data = load_RS(path,fname,outputmode = 'dictofdicts')
            new_rs_summary(data,percentile=1,printing='pdf',display='None')
            print(fname)
        except Exception as ex:
            print('rs_summary failed with ', fname, ex)
    return clims

def make_si_summaries(clmaps,path):
#read all cmaps in dir and plot summary saves to pdf and delete the figs
#[se_im,ctr_im,spec_im,remcon_dict,daq_dict,data_dict,andor_dict,acton_dict,shortlist,sum_spec,sum_im ]

    for fname in clmaps:
        try:
            data = load_SI(path,fname,outputmode = 'dictofdicts')

            new_si_summary(data,percentile=2,printing='pdf',display='None')
            print(fname)
        except Exception as ex:
            print('si_summary failed with ', fname, ex)
    return clmaps


######################
#                    #
#   other utilities  #
#                    #
######################

def get_all_factors(n):
# returns factors of n
    factors = []
    for i in range(1,n+1):
        if n%i == 0:
            factors.append(i)
    return factors
