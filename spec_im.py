from collections.abc import Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from all_funcs_aug15 import load_SI, load_RS
from matplotlib_scalebar.scalebar import ScaleBar
from ctc_funcs import load_uvpl_map
from hyperspy.signals import Signal1D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy as sp


def colorbar(mappable, orientation='vertical', position='right'):
    if orientation == 'vertical':
        axis = 'y'
    else:
        axis = 'x'
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size="5%", pad=0.5)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation)
    cax.ticklabel_format(axis=axis, style='sci', scilimits=(0, 0))
    return cb


def rebin_spec_map(spec_map, wls, **kwargs):
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
        ind_min = np.searchsorted(wls, kwargs['spec_min'])
    elif 'ind_min' in kwargs:
        ind_min = kwargs['min']
    else:
        ind_min = 0

    if 'spec_max' in kwargs:
        ind_max = np.searchsorted(wls, kwargs['spec_max'])
        if ind_max == np.size(wls):
            ind_max = ind_max - 1
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

    En = np.linspace(En_wls[-1], En_wls[0], ne)
    En_spec_map = np.zeros((nv, nh, ne))

    spec_map_interp = sp.interpolate.interp1d(
        En_wls, spec_map[:, :, ind_min:ind_max]*icorr, axis=-1)
    En_spec_map = spec_map_interp(En)

    print(str(nv) + ' x ' + str(nh) + ' spatial x '
          + str(ne) + ' spectral points')

    map_min = np.amin(En_spec_map)
    if map_min < 0:
        print('Correcting negative minimum value in spec map: ' + str(map_min))
        En_spec_map = En_spec_map - map_min + 1e-2

    return (En, En_spec_map, ne)


class SpectralImage(Sequence):
    file_types = []
    _item_list = ['spec_im', 'spec_x_array', 'x_array', 'y_array', 'units',
                  'spec_units']

    def __init__(self, fname='', spec_im=None, spec_x_array=None, x_array=None,
                 y_array=None, units='px', spec_units='index', **kwargs):
        if np.all([i is None for i in [spec_im, spec_x_array]]):
            if self.is_supported(fname):
                self.load_h5_data(fname)
            else:
                raise Exception('%s is not a valid file' % fname)
                return
        elif spec_im is None:
            raise Exception('No spectral image or filename supplied.')
            return
        else:
            self.spec_im = spec_im
            self.spec_units = spec_units
            self.units = units

            if spec_x_array is None:
                self.spec_x_array = np.arange(0, np.size(spec_im[0, 0, :]))
            else:
                self.spec_x_array = spec_x_array

            if x_array is None:
                self.x_array = np.arange(0, np.size(spec_im[:, 0, 0]))
            else:
                self.x_array = x_array

            if y_array is None and x_array is not None:
                self.y_array = x_array
            elif y_array is None:
                self.y_array = np.arange(0, np.size(spec_im[0, :, 0]))
            else:
                self.y_array = y_array

            for kwarg in kwargs.keys():
                setattr(self, kwarg, kwargs[kwarg])
                self._item_list.append(kwarg)

        Sequence.__init__(self)

    def __len__(self):
        return len(self.spec_x_array)

    def _getitem_helper(self, i):
        if isinstance(i, slice):
            ind_min = np.searchsorted(self.spec_x_array, i.start)
            ind_max = np.searchsorted(self.spec_x_array, i.stop)
            if ind_min < 0:
                ind_min = 0
            elif ind_min >= len(self.spec_x_array):
                ind_min = len(self.spec_x_array) - 1

            if ind_max < 0:
                ind_max = 0
            elif ind_max >= len(self.spec_x_array):
                ind_max = len(self.spec_x_array) - 1

            spec_im = self.spec_im[:, :, ind_min:ind_max]
            spec_x_array = self.spec_x_array[ind_min:ind_max]
        else:
            ind = np.searchsorted(self.spec_x_array, i)
            if ind < 0:
                ind = 0
            elif ind >= len(self.spec_x_array):
                ind = len(self.spec_x_array) - 1

            spec_im = self.spec_im[:, :, ind]
            spec_x_array = self.spec_x_array[ind]

        return spec_im, spec_x_array

    def __getitem__(self, i):
        spec_im, spec_x_array = self._getitem_helper(i)
        item_item_list = self._item_list.copy()
        item_item_list.remove('spec_im')
        item_item_list.remove('spec_x_array')
        return type(self)(spec_im=spec_im, spec_x_array=spec_x_array,
                          **{arg: getattr(self, arg) for arg in item_item_list})

    def load_h5_data(self, fname):
        pass

    def plot(self, percentile=5, cmap='viridis'):
        spec_map = self.spec_im.sum(axis=-1)
        return self._plot(spec_map, percentile=5, cmap='viridis')

    def _plot(self, map, title='', percentile=5, cmap='viridis', cbar=True,
              cbar_orientation='horizontal', cbar_position='bottom',):
        ax = plt.gca()
        X, Y = np.meshgrid(self.y_array, self.x_array)
        img = plt.imshow(map, cmap=cmap,
                         vmin=np.percentile(map, percentile),
                         vmax=np.percentile(map, 100-percentile))
        plt.axis('equal')
        plt.axis('off')
        plt.title(title)
        scalebar = ScaleBar(self.x_array[1]-self.x_array[0])
        plt.gca().add_artist(scalebar)
        if cbar:
            colorbar(img, orientation=cbar_orientation, position=cbar_position)
        return ax

    def is_supported(self, fname):
        for ftype in self.file_types:
            if ftype in fname:
                return True
        return False

    def to_energy(self):
        if self.spec_units == 'nm':
            (En, En_spec_im, ne) = rebin_spec_map(self.spec_im,
                                                  self.spec_x_array)
            En_item_list = self._item_list.copy()
            En_item_list.remove('spec_im')
            En_item_list.remove('spec_x_array')
            En_item_list.remove('spec_units')
            return type(self)(spec_im=En_spec_im, spec_x_array=En,
                              spec_units='eV',
                              **{arg: getattr(self, arg) for arg in En_item_list})
        elif self.spec_units == 'eV':
            return self
        else:
            raise Exception('Cannot convert from %s to eV' % self.spec_units)
            return None

    def to_index(self):
        ind_item_list = self._item_list.copy()
        ind_item_list.remove('spec_x_array')
        return type(self)(**{arg: getattr(self, arg) for arg in ind_item_list})

    def to_signal(self):
        s = Signal1D(self.spec_im)
        s.change_dtype('float64')
        s.axes_manager[2].scale = self.spec_x_array[1] - self.spec_x_array[0]
        s.axes_manager[2].offset = self.spec_x_array.min()
        s.axes_manager[2].units = self.spec_units
        if self.spec_units == 'nm':
            s.axes_manager[2].name = '$\lambda$'
        elif self.spec_units == 'eV':
            s.axes_manager[2].name = '$E$'
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].units = 'm'
        s.axes_manager[0].scale = self.x_array[1] - self.x_array[0]
        s.axes_manager[1].name = 'Y'
        s.axes_manager[1].units = 'm'
        s.axes_manager[1].scale = self.y_array[1] - self.y_array[0]
        return s

    def plot_spec(self):
        plt.plot(self.spec_x_array, self.get_spec(sum=True))
        plt.xlabel(self.spec_units)

    def get_spec(self, loc=(0, 0), sum=False):
        if sum:
            return self.spec_im.sum(axis=(0, 1)) / (np.size(self.x_array) * np.size(self.y_array))
        else:
            return self.spec_im[loc[0], loc[1], :]


class CLImage:
    adc_names=['ai0', 'SE2', 'InLens', 'ai3']
    ctr_names=['ctr0', 'ctr1', 'ctr2', 'ctr3']
    file_types=['sync_raster_scan.h5', 'hyperspec_cl.h5']

    def __init__(self, fname = '', **kwargs):
        self.load_h5_data(fname)

    def load_h5_data(self, fname):
        abspath=os.path.abspath(fname)
        self.dat=load_RS(os.path.dirname(abspath),
                           os.path.basename(abspath),)
        span=self.dat['summary']['size_in_nm']
        self.x_array=np.linspace(0.0, span[0]*1e-9,
                                   num = np.size(self.spec_im[:, 0, 0]))
        self.y_array=np.linspace(0.0, span[1]*1e-9,
                                   num = np.size(self.spec_im[0, :, 0]))
        self.units='m'
        self.spec_units='nm'

    def _plot(self, map, percentile = 5, cmap = 'viridis', **kwargs):
        ax=plt.gca()
        X, Y=np.meshgrid(self.y_array, self.x_array)
        img=plt.imshow(map, cmap = cmap,
                         vmin=np.percentile(map, percentile),
                         vmax=np.percentile(map, 100-percentile))
        plt.axis('equal')
        plt.axis('off')

        scalebar = ScaleBar(self.x_array[1]-self.x_array[0])
        plt.gca().add_artist(scalebar)
        return gca

    def get_adc(self, name):
        assert name in self.adc_names
        return self.dat['se'][:, :, self.adc_names.index(name)]

    def get_ctr(self, name):
        assert name in self.ctr_names
        return self.dat['cntr'][:, :, self.ctr_names.index(name)]

    def plot_adc(self, name, percentile=5, cmap='viridis', **kwargs):
        adc_map = self.get_adc(name)
        return self._plot(adc_map, percentile=percentile, cmap=cmap, **kwargs)

    def plot_ctr(self, name, percentile=5, cmap='viridis', **kwargs):
        ctr_map = self.get_ctr(name)
        return self._plot(ctr_map, percentile=percentile, cmap=cmap, **kwargs)

    def is_supported(self, fname):
        for ftype in self.file_types:
            if ftype in fname:
                return True
        return False


class CLSpectralImage(SpectralImage, CLImage):
    file_types = ["hyperspec_cl.h5", ]

    def __init__(self, fname='', spec_im=None, spec_x_array=None, x_array=None,
                 y_array=None, units='px', spec_units='index', dat=None):
        SpectralImage.__init__(self, fname=fname, spec_im=spec_im,
                               spec_x_array=spec_x_array, x_array=x_array,
                               y_array=y_array, units=units,
                               spec_units=spec_units, dat=dat)

    def load_h5_data(self, fname):
        abspath = os.path.abspath(fname)
        self.dat = load_SI(os.path.dirname(abspath),
                           os.path.basename(abspath),)
        self._item_list.append('dat')
        self.spec_x_array = np.array(self.dat['wavelength'])
        self.spec_im = self.dat['SI']
        span = self.dat['summary']['size_in_nm']
        self.x_array = np.linspace(0.0, span[0]*1e-9,
                                   num=np.size(self.spec_im[:, 0, 0]))
        self.y_array = np.linspace(0.0, span[1]*1e-9,
                                   num=np.size(self.spec_im[0, :, 0]))
        self.units = 'm'
        self.spec_units = 'nm'


class PLSpectralImage(SpectralImage):
    file_types = ['asi_OO_hyperspec_scan.h5', ]

    def load_h5_data(self, fname):
        (sample, self.spec_x_array, self.spec_im, h_array, v_array, nh, nv, nf,
         dh, dv) = load_uvpl_map(fname)
        self.x_array = h_array * 1e-3
        self.y_array = v_array * 1e-3
        self.spec_units = 'nm'
        self.units = 'm'
