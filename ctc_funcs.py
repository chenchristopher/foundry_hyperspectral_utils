#################
#    imports    #
#################
import os
#import sys
import scipy as sp
#import scipy.misc
#import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import h5py as h5
import numpy as np
import pandas as pd
from ipywidgets import widgets
import IPython
import seaborn as sns
#import hyperspy.api as hs
#import hyperspy_gui_ipywidgets as hsgui
from skimage.measure import compare_ssim as ssim
from mpl_toolkits.axes_grid1 import host_subplot

from mpl_toolkits.axes_grid1 import make_axes_locatable

#####################
# IO
#####################
def load_horiba_map(fname,*args,**kwargs):
    fext = os.path.splitext(fname)[1]

    if fext == '.txt':
        df = pd.read_csv(fname,delimiter='\t')
        df = df.rename(columns = {'Unnamed: 0':'x','Unnamed: 1':'y'})

        shift = np.array(df.columns[2:]).astype(np.float)
        E = 1240/532-1240*shift*1e-7
    else:
        print('error')
        return

    # working out indexing of an image
    xvals = np.sort(df.x.unique())
    yvals = np.sort(df.y.unique())
    xs = np.array(df.x.astype(np.float))
    ys = np.array(df.y.astype(np.float))

    #calculating replacement vectors for position value in pandas array
    xind=np.searchsorted(xvals,xs)
    yind=np.searchsorted(yvals,ys)
    #print('indexing check',xind.size,yind.size)
    #print('image size check',xvals.size,yvals.size)
    #print('SI depth',shift.size)

    #wrapping data into si in wavelngth (wl), energy (e)
    data=np.zeros([xvals.size,yvals.size,shift.size],dtype='float')
    dataes=data
    shiftes=np.linspace(shift.min(),shift.max(),shift.size)
    Ees=1240/532-1240*shiftes*1e-7
    icorr=1
    w = widgets.IntProgress(value = 0,min = 0, max=xind.size,description="Progress...")
    display(w)
    for i in range(0,xind.size):
        data[xind[i],yind[i],:]=df.values[i,2:]
        dataes[xind[i],yind[i],:]=sp.interpolate.griddata(
                shift,df.as_matrix()[i,2:]*icorr,shiftes,method='cubic')
        #plt.plot(shift,df.values[i,2:])
        w.value = i+1

    #print('Complete')

    return (xvals,yvals,shiftes,Ees,dataes)


def load_uvpl_map(fname):
    """Short summary.

    Parameters
    ----------
    fname : type
        Description of parameter `fname`.

    Returns
    -------
    type
        Description of returned object.

    """
    measurement_name = 'asi_OO_hyperspec_scan'

    if measurement_name+'.h5' not in fname:
        print('Error - bad input file')
        return

    print('Loading UVPL scan: ' + fname)
    f = h5.File(fname)
    sample = str(f['app']['settings'].attrs['sample'])
    wls = np.array(f['measurement'][measurement_name]['wls'])
    spec_map = np.array(f['measurement'][measurement_name]['spec_map'])
    h_array = np.array(f['measurement'][measurement_name]['h_array'])*1e3
    v_array = np.array(f['measurement'][measurement_name]['v_array'])*1e3
    nh = int(f['measurement'][measurement_name]['settings'].attrs['Nh'])
    nv = int(f['measurement'][measurement_name]['settings'].attrs['Nv'])
    dh = float(f['measurement'][measurement_name]['settings'].attrs['dh'])*1e3
    dv = float(f['measurement'][measurement_name]['settings'].attrs['dv'])*1e3
    nf = np.size(wls)
    spec_map = np.reshape(spec_map,(nv,nh,nf))
    f.close()

    print('Sample: ' + sample)
    map_min = np.amin(spec_map)
    if map_min < 0:
        print('Correcting negative minimum value in spec map: ' + str(map_min))
        spec_map = spec_map - map_min + 1e-2

    print( str(nv) + ' x ' + str(nh) + ' spatial x ' + str(nf) + ' spectral points')

    return (sample, wls, spec_map, h_array, v_array, nh, nv, nf, dh, dv)



##############
# util
##############
def rebin_spec_map(spec_map,wls,**kwargs):
    """Short summary.

    Parameters
    ----------
    spec_map : type
        Description of parameter `spec_map`.
    wls : type
        Description of parameter `wls`.
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    type
        Description of returned object.

    """
    """
    Rebins a spectral map from nm to eV. Resamples the map to provide evenly spaced energy values.

    Also supports trimming wavelength range by array indicies (ind_min,ind_max) or spectral values (spec_min,spec_max).

    Spectral locations supercede array indicies.
    """
    [nv, nh, nf] = np.shape(spec_map)

    if 'spec_min' in kwargs:
        ind_min = np.searchsorted(wls,kwargs['spec_min'])
    elif 'ind_min' in kwargs:
        ind_min = kwargs['min']
    else:
        ind_min = 0

    if 'spec_max' in kwargs:
        ind_max = np.searchsorted(wls,kwargs['spec_max'])
    elif 'ind_max' in kwargs:
        ind_max = kwargs['max']
    else:
        ind_max = nf-1

    if ind_max < ind_min:
        tmp = ind_max
        ind_max = ind_min
        ind_min = tmp

    print('Interpolating spectral data from nm to eV')
    print(str(wls[ind_min]) + ' to ' + str(wls[ind_max]) + ' nm')

    ne = ind_max-ind_min
    En_wls = 1240/wls[ind_min:ind_max]
    icorr = wls[ind_min:ind_max]**2

    En = np.linspace(En_wls[-1],En_wls[0],ne)
    En_spec_map = np.zeros((nv,nh,ne))

    spec_map_interp = sp.interpolate.interp1d(En_wls, spec_map[:,:,ind_min:ind_max], axis=-1)
    En_spec_map = spec_map_interp(En)

    print(str(nv) + ' x ' + str(nh) + ' spatial x ' + str(ne) + ' spectral points')

    map_min = np.amin(En_spec_map)
    if map_min < 0:
        print('Correcting negative minimum value in spec map: ' + str(map_min))
        En_spec_map = En_spec_map - map_min + 1e-2

    return (En, En_spec_map, ne)


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
    return ndarray

##############
#    disp    #
##############

def nice_matplotlib():
    # inspired by http://nipunbatra.github.io/2014/08/latexify/
    params = {
    #    'text.latex.preamble': ['\\usepackage{gensymb}'],
        'image.origin': 'lower',
        'image.interpolation': 'nearest',
        'image.cmap': 'gray',
        'axes.grid': False,
        'savefig.dpi': 300,  # to adjust notebook inline plot size
        'axes.labelsize': 12, # fontsize for x and y labels (was 10)
        'axes.titlesize': 12,
        'font.size': 12, # was 10
        'legend.fontsize': 10, # was 10
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    #    'text.usetex': True,
    #    'figure.figsize': [5, 5],
    #    'font.family': 'serif',
    }
    matplotlib.rcParams.update(params)

def plot_all_data(shift,data,linlog=True,index=False):
    dshape = np.shape(data)
    spec_size = dshape[2]
    data = bin_ndarray(data,(1,1,spec_size))
    if index:
        f = plt.figure()
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        if linlog:
            ax1.plot(shift,np.reshape(data,(spec_size,)))
        else:
            ax1.semilogy(shift,np.reshape(data,(spec_size,)))
        ax1.set_xlabel('spectral unit')
        if linlog:
            ax2.plot(np.reshape(data,(spec_size,)))
        else:
            ax2.semilogy(np.reshape(data,(spec_size,)))
        ax2.set_xlabel('index')
    else:
        f = plt.figure()
        if linlog:
            plt.plot(shift,np.reshape(data,(spec_size,)))
        else:
            plt.semilogy(shift,np.reshape(data,(spec_size,)))
        plt.xlabel('spectral unit')
        plt.ylabel('a.u.')

    return f

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

    if no_of_bands > no_of_rows*no_of_cols:
        no_of_rows = no_of_rows+1

    gs=gspec.GridSpec(no_of_rows,no_of_cols)

    subplotindxs = np.reshape(np.arange(0,no_of_cols*no_of_rows),[no_of_rows,no_of_cols])

    i=0
    for arg in args:
        band=np.sum(a=spec_im[:,:,arg[0]:arg[1]],axis=2)
        coord=np.where(subplotindxs==i)
        plt.subplot(gs[coord[0][0],coord[1][0]])
        plt.title(arg)
        img = plt.imshow(band,cmap=arg[2],
            vmin=np.percentile(band,percentile),
            vmax=np.percentile(band,100-percentile))
        #plt.colorbar()
        colorbar(img)
        i = i + 1

    f_h.suptitle(cpath + fname)
    return f_h

def plot_spec_bands(spec_im,spec,xvals,yvals,cpath,fname,*args,**kwargs):
    """Short summary.

    Parameters
    ----------
    spec_im : type
        Description of parameter `spec_im`.
    spec : type
        Description of parameter `spec`.
    xvals : type
        Description of parameter `xvals`.
    yvals : type
        Description of parameter `yvals`.
    cpath : type
        Description of parameter `cpath`.
    fname : type
        Description of parameter `fname`.
    *args : type
        Description of parameter `*args`.
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    type
        Description of returned object.

    """
#revize
#plots image bands defined by *args,
#each arg is (low bound, high bound, colormap, descriptor)
#Usage:
#       plot_si_bands(spec_im1,spec,cpath,fname,
#              (0,1600,'plasma','1'),
#              (180,220,'Blues','2'),
#              (220,350,'Blues','3'),
#              (400,600,'Greens','4'),
#              (550,750,'plasma','5'),
#              (750,1000,'plasma','6'),
#              (500,1000,'plasma','7'),
#              (1100,1500,'Blues','8'),
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

    if no_of_bands > no_of_rows*no_of_cols:
        no_of_rows = no_of_rows+1

    gs=gspec.GridSpec(no_of_rows,no_of_cols)

    subplotindxs = np.reshape(np.arange(0,no_of_cols*no_of_rows),[no_of_rows,no_of_cols])

    if spec[0] > spec[1]:
            print('Reversed')

    X,Y = np.meshgrid(yvals,xvals)

    i=0
    for arg in args:
        band_min = np.searchsorted(spec,float(arg[0]))
        band_max = np.searchsorted(spec,float(arg[1]))
        print('Desired spectral range: ' + str(arg[0]) + ' to ' + str(arg[1]))
        print('Integrating indicies from ' + str(band_min) + ' to ' + str(band_max))
        band=np.sum(a=spec_im[:,:,band_min:band_max],axis=2)
        coord=np.where(subplotindxs==i)
        plt.subplot(gs[coord[0][0],coord[1][0]])
        plt.title(arg[3] + ' ' + str(arg[0:2]))
        #img = plt.imshow(band,cmap=arg[2],
        #    vmin=np.percentile(band,percentile),
        #    vmax=np.percentile(band,100-percentile))
        img = plt.pcolormesh(X,Y,band,cmap=arg[2],
            vmin=np.percentile(band,percentile),
            vmax=np.percentile(band,100-percentile),
            shading='flat')
        plt.axis('equal')
        plt.xlabel('$\mu$m')
        plt.ylabel('$\mu$m')
        #plt.colorbar()
        colorbar(img,orientation='horizontal',position='bottom')
        i = i + 1

    f_h.suptitle(cpath + fname)
    return f_h

def colorbar(mappable,orientation='vertical',position='right'):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size="5%", pad=0.5)
    return fig.colorbar(mappable, cax=cax,orientation=orientation)
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #return fig.colorbar(mappable,cax=cax,orientation='vertical')


def plot_bss_results(s,no_of_cols=3,no_of_bands=6,cmap='plasma'):
    # grab the blind source separation loadings and factors
    loading_list = s.get_bss_loadings().split()
    factor_list = s.get_bss_factors().split()
    no_of_loadings = len(loading_list)

    (xvals,yvals,spec)=get_hs_axes(s)

    # some quick math to calculate the number of rows per figure and number of figures
    #no_of_bands = 6
    #no_of_cols = 3
    no_of_rows = no_of_bands//no_of_cols
    if no_of_bands > no_of_rows*no_of_cols:
        no_of_rows = no_of_rows+1

    no_of_figs= no_of_loadings//no_of_bands
    if no_of_loadings > no_of_figs*no_of_bands:
        no_of_figs = no_of_figs+1

    fig_list = list()

    # loop over all figures
    for jj in range(0,no_of_figs):
        print('fig ' + str(jj+1) + ' of ' + str(no_of_figs))
        f = plt.figure()

        # set up grid for figure
        gs=gspec.GridSpec(2*no_of_rows,no_of_cols)

        subplotindxs = []
        for kk in range(0,no_of_rows):
            start = 2*no_of_cols*kk
            end = start + no_of_cols*2
            tmp = np.arange(start,end)
            tmp = np.transpose(np.reshape(tmp,(no_of_cols,no_of_rows)))

            if kk == 0:
                subplotindxs = tmp
            else:
                subplotindxs = np.concatenate((subplotindxs,tmp),axis=0)

        # start of list for this figure
        l0 = jj*no_of_rows*no_of_cols

        X,Y = np.meshgrid(yvals,xvals)
        i=0
        for ll in range(0,no_of_bands):
            lx = l0 + ll
            if lx >= no_of_loadings:
                break
            print('component ' + str(lx+1) + ' of ' + str(no_of_loadings))

            coord=np.where(subplotindxs==i)
            plt.subplot(gs[coord[0][0],coord[1][0]])
            img = plt.pcolormesh(X,Y,loading_list[lx].data,cmap=cmap,
                shading='flat')
            plt.title(lx)
            plt.xlabel('$\mu$m')
            plt.ylabel('$\mu$m')
            plt.axis('equal')
            colorbar(img)
            i = i + 1

            coord=np.where(subplotindxs==i)
            plt.subplot(gs[coord[0][0],coord[1][0]])
            plt.plot(spec,factor_list[lx].data)
            plt.gca().ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            plt.xlabel('$E$ [eV]')
            i = i + 1
        plt.axis('tight')
        plt.suptitle('BSS pg ' + str(jj+1) + ' of ' + str(no_of_figs))
        f.set_size_inches(10,10)
        plt.tight_layout(h_pad=1)
        fig_list.append(f)

    return fig_list

def plot_decomp_results(s,no_of_cols=3,no_of_bands=6,cmap='plasma'):
    # grab the blind source separation loadings and factors
    loading_list = s.get_decomposition_loadings().split()
    factor_list = s.get_decomposition_factors().split()
    no_of_loadings = len(loading_list)

    (xvals,yvals,spec)=get_hs_axes(s)

    # some quick math to calculate the number of rows per figure and number of figures
    #no_of_bands = 6
    #no_of_cols = 3
    no_of_rows = no_of_bands//no_of_cols
    if no_of_bands > no_of_rows*no_of_cols:
        no_of_rows = no_of_rows+1

    no_of_figs= no_of_loadings//no_of_bands
    if no_of_loadings > no_of_figs*no_of_bands:
        no_of_figs = no_of_figs+1

    fig_list = list()

    # loop over all figures
    for jj in range(0,no_of_figs):
        print('fig ' + str(jj+1) + ' of ' + str(no_of_figs))
        f = plt.figure()

        # set up grid for figure
        gs=gspec.GridSpec(2*no_of_rows,no_of_cols)

        subplotindxs = []
        for kk in range(0,no_of_rows):
            start = 2*no_of_cols*kk
            end = start + no_of_cols*2
            tmp = np.arange(start,end)
            tmp = np.transpose(np.reshape(tmp,(no_of_cols,no_of_rows)))

            if kk == 0:
                subplotindxs = tmp
            else:
                subplotindxs = np.concatenate((subplotindxs,tmp),axis=0)

        # start of list for this figure
        l0 = jj*no_of_rows*no_of_cols

        X,Y = np.meshgrid(yvals,xvals)
        i=0
        for ll in range(0,no_of_bands):
            lx = l0 + ll
            print('component ' + str(lx+1) + ' of ' + str(no_of_loadings))
            if lx >= no_of_loadings:
                break
            coord=np.where(subplotindxs==i)
            plt.subplot(gs[coord[0][0],coord[1][0]])
            img = plt.pcolormesh(X,Y,loading_list[lx].data,cmap=cmap)
            plt.title(lx)
            plt.xlabel('$\mu$m')
            plt.ylabel('$\mu$m')
            plt.axis('equal')
            colorbar(img)
            i = i + 1

            coord=np.where(subplotindxs==i)
            plt.subplot(gs[coord[0][0],coord[1][0]])
            plt.plot(spec,factor_list[lx].data)
            plt.gca().ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            plt.xlabel('$E$ [eV]')
            i = i + 1
        plt.axis('tight')
        plt.suptitle('NMF pg ' + str(jj+1) + ' of ' + str(no_of_figs))
        f.set_size_inches(10,10)
        plt.tight_layout(h_pad=1)
        fig_list.append(f)

    return fig_list

def plot_loading_correlations(s,mask=None,alpha=0.2,cmap='plasma'):
    sns.set_context('paper')

    bss_comp_arr = s.get_bss_loadings().split()
    nmf_comp_arr = s.get_decomposition_loadings().split()

    no_of_rows = len(bss_comp_arr)
    no_of_cols = len(nmf_comp_arr)
    f_h=plt.figure()

    gs=gspec.GridSpec(no_of_rows,no_of_cols)

    subplotindxs = np.reshape(np.arange(0,no_of_cols*no_of_rows),[no_of_rows,no_of_cols])
    #i=0
    for jj in range(0, len(bss_comp_arr)):
        for kk in range(0,len(nmf_comp_arr)):
            i = jj*no_of_cols + kk
            coord=np.where(subplotindxs==i)
            x_data = nmf_comp_arr[kk].data.flatten()
            y_data = bss_comp_arr[jj].data.flatten()
            ax = plt.subplot(gs[coord[0][0],coord[1][0]])
            if mask is None:
                #sns.scatterplot(y=y_data,x=x_data)
                plt.scatter(x_data,y_data,alpha=alpha,cmap=cmap,edgecolor='white')
            else:
                plt.scatter(x_data,y_data,c=mask.flatten(),alpha=alpha,cmap=cmap,edgecolor='white')
            ax.set_aspect((x_data.max()-x_data.min())/(y_data.max()-y_data.min()))
            (rsq,pval) = sp.stats.spearmanr(x_data,b=y_data)
            #mssim = ssim(nmf_comp_arr[kk].data,bss_comp_arr[jj].data,gaussian_weights=True)
            #plt.title('BSS ' + str(jj) + ' vs NMF ' + str(kk) + ' R$^2$= %0.2f SSIM=%0.2f' % (rsq,mssim))
            plt.title('BSS ' + str(jj) + ' vs NMF ' + str(kk) + ' R$^2$= %0.2f' % rsq)

    sns.set_context('notebook')




def plot_relevant_correlations(s,threshold=0.5,mask=None,alpha=0.2,cmap='plasma'):
    sns.set_context('paper')

    bss_load_arr = s.get_bss_loadings().split()
    nmf_load_arr = s.get_decomposition_loadings().split()
    bss_factor_arr = s.get_bss_factors().split()
    nmf_factor_arr = s.get_decomposition_factors().split()

    no_of_nmf = len(nmf_load_arr)
    no_of_bss = len(bss_load_arr)

    (xvals,yvals,En) = get_hs_axes(s)
    X,Y = np.meshgrid(xvals,yvals)

    for jj in range(0,no_of_bss):
        y_factor = bss_factor_arr[jj]
        y_load = bss_load_arr[jj]
        for kk in range(0, no_of_nmf):
            x_factor = nmf_factor_arr[kk]
            x_load = nmf_load_arr[kk]
            x_data = x_load.data.flatten()
            y_data = y_load.data.flatten()
            (rsq,pval) = sp.stats.spearmanr(y_data,b=x_data)

            if rsq >= threshold:
                print('BSS %d vs NMF %d R^2 = %0.2f' % (jj, kk, rsq))

                f_h=plt.figure()

                axScatter = plt.subplot(231)
                if mask is None:
                    plt.scatter(x_data,y_data,cmap=cmap,alpha=alpha,edgecolor='white')
                else:
                    plt.scatter(x_data,y_data,c=mask.flatten(),alpha=alpha,cmap=cmap,edgecolor='white')
                axScatter.set_aspect((x_data.max()-x_data.min())/(y_data.max()-y_data.min()))
                axScatter.set_title('R$^2$= %0.2f' % rsq)
                axScatter.set_xlabel('NMF ' + str(kk))
                axScatter.set_ylabel('BSS ' + str(jj))

                axNMFmesh = plt.subplot(232)
                img = axNMFmesh.pcolormesh(X,Y,x_load.data,cmap='plasma')
                axNMFmesh.set_title('NMF ' + str(kk))
                axNMFmesh.set_xlabel('$\mu$m')
                axNMFmesh.set_ylabel('$\mu$m')
                axNMFmesh.axis('equal')
                colorbar(img,orientation='horizontal',position='bottom')

                axBSSmesh = plt.subplot(233)
                img = axBSSmesh.pcolormesh(X,Y,y_load.data,cmap='plasma')
                axBSSmesh.set_title('BSS ' + str(jj))
                axBSSmesh.set_xlabel('$\mu$m')
                axBSSmesh.set_ylabel('$\mu$m')
                axBSSmesh.axis('equal')
                colorbar(img,orientation='horizontal',position='bottom')

                axplot = host_subplot(212)
                axplot.plot(En,x_factor.data,label='NMF ' + str(kk))
                axplot.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
                axplot.set_ylabel('NMF ' + str(kk))
                axplot2 = axplot.twinx()
                axplot2.plot(En,y_factor.data,label='BSS ' + str(jj))
                axplot2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
                axplot2.set_ylabel('BSS ' + str(jj))
                axplot.legend()
                axplot.set_xlabel('$E$ (eV)')

                plt.tight_layout(h_pad=1)

    sns.set_context('notebook')


def get_hs_axes(s):
    dx = s.axes_manager[0].scale
    nx = s.axes_manager[0].size
    X = np.linspace(0, dx*(nx-1), num=nx)
    dy = s.axes_manager[1].scale
    ny = s.axes_manager[1].size
    Y = np.linspace(0, dy*(ny-1), num=ny)
    dE = s.axes_manager[2].scale
    nE = s.axes_manager[2].size
    E = s.axes_manager[2].offset + np.linspace(0,dE*(nE-1),num=nE)
    return (X,Y,E)


def plot_hs_results(loadings,factors,hs_axes,no_of_bands=4,title='',cmap='plasma'):
    # grab the blind source separation loadings and factors
    loading_list = loadings.split()
    factor_list = factors.split()
    no_of_cols = 3
    no_of_loadings = len(loading_list)
    (xvals,yvals,spec) = hs_axes
    #(xvals,yvals,spec)=get_hs_axes(s)

    # some quick math to calculate the number of rows per figure and number of figures
    no_of_figs = no_of_loadings//no_of_bands
    no_of_rows = no_of_bands
    if no_of_loadings > no_of_figs*no_of_bands:
        no_of_figs = no_of_figs+1

    fig_list = list()
    print(no_of_figs)
    # loop over all figures
    for jj in range(0,no_of_figs):
        print('fig ' + str(jj+1) + ' of ' + str(no_of_figs))
        f = plt.figure()

        # set up grid for figure
        gs=gspec.GridSpec(no_of_rows,no_of_cols)

        subplotindxs = []
        # for kk in range(0,no_of_rows):
        #     start = 2*no_of_cols*kk
        #     end = start + no_of_cols*2
        #     tmp = np.arange(start,end)
        #     tmp = np.transpose(np.reshape(tmp,(no_of_cols,no_of_rows)))
        #
        #     if kk == 0:
        #         subplotindxs = tmp
        #     else:
        #         subplotindxs = np.concatenate((subplotindxs,tmp),axis=0)

        tmp = np.arange(0,2*no_of_bands)
        tmp = np.reshape(tmp,(no_of_bands,2))
        subplotindxs = np.zeros((tmp.shape[0],tmp.shape[1]+1))
        subplotindxs[:,0:2] = tmp
        subplotindxs[:,2] = tmp[:,1]

        # start of list for this figure
        l0 = jj*no_of_bands

        X,Y = np.meshgrid(xvals,yvals)
        i=0
        for ll in range(0,no_of_bands):
            lx = l0 + ll
            if lx >= no_of_loadings:
                break
            print('component ' + str(lx+1) + ' of ' + str(no_of_loadings))

            coord=np.where(subplotindxs==i)
            coord_shape = np.shape(coord)
            plt.subplot(gs[coord[0][0],coord[1][0]:(coord_shape[1])])
            img = plt.pcolormesh(X,Y,loading_list[lx].data,cmap=cmap,
                shading='flat')
            plt.title(lx)
            plt.xlabel('$\mu$m')
            plt.ylabel('$\mu$m')
            plt.axis('equal')
            colorbar(img)
            i = i + 1

            coord=np.where(subplotindxs==i)
            coord_shape = np.shape(coord)
            plt.subplot(gs[coord[0][0],coord[1][0]:(coord_shape[1]+1)])
            plt.plot(spec,factor_list[lx].data)
            plt.gca().ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            plt.xlabel('$E$ [eV]')
            i = i + 1
        plt.axis('tight')
        plt.suptitle(title + ' pg ' + str(jj+1) + ' of ' + str(no_of_figs))
        f.set_size_inches(10,10)
        plt.tight_layout(h_pad=1)
        fig_list.append(f)

    return fig_list
