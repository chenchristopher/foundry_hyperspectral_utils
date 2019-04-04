import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gspec
from qtpy import QtWidgets
import spec_im as si


def plot_si_bands(spec_im, *args, **kwargs):
    """
    revize
    plots image bands defined by *args,
    each arg is (low bound, high bound, colormap, descriptor)
    Usage:
          plot_si_bands(spec_im1,spec,cpath,fname,
                 (0,1600,'plasma','1'),
                 (180,220,'Blues','2'),
                 (220,350,'Blues','3'),
                 (400,600,'Greens','4'),
                 (550,750,'plasma','5'),
                 (750,1000,'plasma','6'),
                 (500,1000,'plasma','7'),
                 (1100,1500,'Blues','8'),
                 percentile = 3,
                 no_of_cols = 4)
    """
    assert isinstance(spec_im, si.SpectralImage)
    f_h = plt.figure()

    if 'percentile' in kwargs.keys():
        percentile = kwargs['percentile']
    else:
        percentile = 5

    if 'no_of_cols' in kwargs.keys():
        no_of_cols = kwargs['no_of_cols']
    else:
        no_of_cols = 3

    no_of_bands = len(args)
    no_of_rows = no_of_bands//no_of_cols

    if no_of_bands > no_of_rows*no_of_cols:
        no_of_rows = no_of_rows + 1

    gs = gspec.GridSpec(no_of_rows, no_of_cols)

    subplotindxs = np.reshape(np.arange(0, no_of_cols*no_of_rows),
                              [no_of_rows, no_of_cols])

    i = 0
    units = spec_im.spec_units
    for arg in args:
        coord = np.where(subplotindxs == i)
        plt.subplot(gs[coord[0][0], coord[1][0]])
        ax = spec_im[arg[0]:arg[1]].plot(percentile=percentile)
        if units == 'eV':
            ax.set_title('%s from %0.1f to %0.1f %s' % (arg[3], arg[0], arg[1],
                                                        units))
        else:
            ax.set_title('%s from %d to %d %s' % (arg[3], arg[0], arg[1],
                                                  units))
        i = i + 1

    # f_h.suptitle(cpath+fname)
    return f_h


def gui_fname(dir=None):
    """Select a file via a dialog and return the file name"""
    if dir is None: dir = '../'
    fname = QtWidgets.QFileDialog.getOpenFileName(None, "select data file...", dir, filter="h5 files (*hyperspec_cl.h5)")
    return fname[0]

def plot_cl_summary(spec_im):
    assert isinstance(spec_im, si.CLSpectralImage)
    f0, ax0 = plt.subplots(3, 3, subplot_kw=dict(polar=True))

    plt.subplot(3,3,1)
    se2 = spec_im.plot_adc('SE2', cmap='gray', cbar=False)
    se2.set_title('SE2')

    plt.subplot(3,3,2)
    il = spec_im.plot_adc('InLens',cmap='gray', cbar=False)
    il.set_title('InLens')

    plt.subplot(3,3,3)
    T = spec_im.plot_adc('ai3',cmap='gray')
    T.set_title('ai3')


    plt.subplot(3,3,4)
    c0 = spec_im.plot_ctr('ctr0')
    c0.set_title('ctr0')

    plt.subplot(3,3,5)
    c1 = spec_im.plot_ctr('ctr1')
    c1.set_title('ctr1')

    plt.subplot(3,3,6)
    c2 = spec_im.plot_ctr('ctr2')
    c2.set_title('ctr2')

    plt.subplot(3,3,7)
    sisum = spec_im.plot()
    sisum.set_title('SI')

    plt.subplot(3,3,(8,9))
    spec_im.plot_spec()
    plt.text(spec_im.spec_x_array.min(), spec_im.get_spec(sum=True).min(),
             spec_im.dat['summary']['description'])

    f0.set_size_inches(10,10)
    f0.tight_layout(h_pad=1)

    return f0


def plot_hs_results(loadings, factors, spec_im, no_of_bands=4, title='',
                    cmap='plasma'):
    # grab the blind source separation loadings and factors
    loading_list = loadings.split()
    factor_list = factors.split()
    no_of_loadings = len(loading_list)

    # some quick math to calculate the number of rows per figure and number of figures
    no_of_figs = no_of_loadings//no_of_bands
    no_of_rows = no_of_bands
    if no_of_loadings > no_of_figs*no_of_bands:
        no_of_figs = no_of_figs+1

    fig_list = list()
    #print(no_of_figs)
    # loop over all figures
    for jj in range(0, no_of_figs):
        #print('fig ' + str(jj+1) + ' of ' + str(no_of_figs))
        f = plt.figure()

        # start of list for this figure
        l0 = jj*no_of_bands

        for ll in range(0, no_of_bands):
            lx = l0 + ll
            if lx >= no_of_loadings:
                break
            #print('component ' + str(lx+1) + ' of ' + str(no_of_loadings))
            plt.subplot(no_of_rows, 3, 3*ll+1)
            ax = spec_im._plot(loading_list[lx].data, cmap=cmap,
                               cbar_orientation='vertical',
                               cbar_position='right')
            ax.set_title(lx)
            ax.set_xlabel('m')
            ax.set_ylabel('m')
            ax.axis('equal')

            plt.subplot(no_of_rows, 3, (3*ll+2, 3*ll+3))
            plt.plot(spec_im.spec_x_array, factor_list[lx].data)
            plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.xlabel(spec_im.spec_units)

        plt.axis('tight')
        plt.suptitle(title + ' pg ' + str(jj+1) + ' of ' + str(no_of_figs))
        f.set_size_inches(10,10)
        plt.tight_layout(h_pad=1)
        fig_list.append(f)

    return fig_list


def plot_bss_results(s, spec_im, title='', cmap='gray', fig_rows=4,
                     components=8, max_iter=1000):
    s.blind_source_separation(number_of_components=components,
                              on_loadings=True, max_iter=max_iter)
    return plot_hs_results(s.get_bss_loadings(), s.get_bss_factors(),
                           spec_im, title=title, cmap=cmap,
                           no_of_bands=fig_rows)


def plot_decomp_results(s, spec_im, title='', cmap='gray', fig_rows=4,
                        algorithm='nmf', output_dimension=8):
    s.decomposition(algorithm=algorithm, output_dimension=output_dimension)
    return plot_hs_results(s.get_decomposition_loadings(),
                           s.get_decomposition_factors(),
                           spec_im, title=title, cmap=cmap,
                           no_of_bands=fig_rows)
