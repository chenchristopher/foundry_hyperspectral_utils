from collections.abc import Sequence
from collections import OrderedDict
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from hyperspy.signals import Signal1D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py as h5
import scipy as sp
from hyperspy.misc.utils import DictionaryTreeBrowser
import re
from copy import copy
import math
from sklearn.decomposition import PCA


def h5_to_dictionary(fh, tree=None, level='', debug=False):
    if tree is None:
        if debug:
            print('Creating DictionaryTreeBrowser')
        tree = DictionaryTreeBrowser()
    elif level == '' and debug:
        print('At root level')
    else:
        if debug:
            print('function called at level', level)
        level += '.'

    for key in list(fh.attrs.keys()):
        val = fh.attrs[key]
        tree.set_item(level + key, copy(val))
        if debug:
            print('level %s, %s: %s' % (level, key, val))

    for (key, desc) in list(fh.items()):
        if debug:
            print(key, desc)
        key = str(key)
        desc = str(desc)
        if '(0 members)' in desc:
            if debug:
                print('No members of level %s, %s' % (level, key))
            continue
        elif 'HDF5 dataset' in desc:
            tree.set_item(level + key, np.squeeze(fh[key]).copy())
        elif 'HDF5 group' in desc:
            if debug:
                print('Recursive call at level %s, %s' % (level, key))
            tree = h5_to_dictionary(fh[key], tree=tree, level=level + key)
    return tree


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
    Rebins a spectral map from nm to eV. Resamples the map to provide evenly
    spaced energy values.

    Also supports trimming wavelength range by array indicies (ind_min,ind_max)
    or spectral values (spec_min,spec_max).

    Spectral locations supercede array indicies.
    """
    if len(spec_map.shape) == 3:
        [nv, nh, nf] = np.shape(spec_map)
    elif len(spec_map.shape) == 4:
        [nz, nv, nh, nf] = np.shape(spec_map)

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

    if len(spec_map.shape) == 3:
        En_spec_map = np.zeros((nv, nh, ne))
        spec_map_interp = sp.interpolate.interp1d(
            En_wls, spec_map[:, :, ind_min:ind_max]*icorr, axis=-1)
    elif len(spec_map.shape) == 4:
        En_spec_map = np.zeros((nz, nv, nh, ne))
        spec_map_interp = sp.interpolate.interp1d(
            En_wls, spec_map[:, :, :, ind_min:ind_max]*icorr, axis=-1)

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

    def __init__(self, fname='', dat=None):
        if dat is not None:
            assert isinstance(dat, DictionaryTreeBrowser)
            self.dat = dat.copy()
        elif self.is_supported(fname):
            self.load_h5_data(fname)
        else:
            raise Exception('No valid dictionary or filename supplied.')
            return

        self.load_from_metadata()
        Sequence.__init__(self)

        if self.is_supported(fname):
            print('Load from %s complete.' % fname)
            if hasattr(self, 'z_array'):
                print('%d x %d x %d spatial x %d spectral points' % (
                    len(self.z_array), len(self.x_array), len(self.y_array),
                    len(self.spec_x_array)))
            else:
                print('%d x %d spatial x %d spectral points' % (
                    len(self.x_array), len(self.y_array),
                    len(self.spec_x_array)))

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

            if len(self.spec_im.shape) == 3:
                spec_im = self.spec_im[:, :, ind_min:ind_max]
            elif len(self.spec_im.shape) == 4:
                spec_im = self.spec_im[:, :, :, ind_min:ind_max]
            spec_x_array = self.spec_x_array[ind_min:ind_max]
        else:
            ind = np.searchsorted(self.spec_x_array, i)
            if ind < 0:
                ind = 0
            elif ind >= len(self.spec_x_array):
                ind = len(self.spec_x_array) - 1

            if len(self.spec_im.shape) == 3:
                spec_im = self.spec_im[:, :, ind]
            elif len(self.spec_im.shape) == 4:
                spec_im = self.spec_im[:, :, :, ind]
            spec_x_array = self.spec_x_array[ind]

        return np.array(spec_im), np.array(spec_x_array)

    def __getitem__(self, i):
        spec_im, spec_x_array = self._getitem_helper(i)

        clone = self.copy()
        clone.spec_x_array = spec_x_array
        clone.spec_im = spec_im
        return clone

    def copy(self):
        clone = type(self)(dat=self.dat)
        clone.spec_im = np.array(self.spec_im)
        clone.spec_x_array = np.array(self.spec_x_array)
        clone.x_array = np.array(self.x_array)
        clone.y_array = np.array(self.y_array)
        clone.spec_units = 'nm'
        clone.units = 'mm'
        return clone

    def load_h5_data(self, fname):
        try:
            f = h5.File(fname)
            self.dat = h5_to_dictionary(f)
        except Exception as ex:
            print('Could not load file', fname, ex)
        finally:
            f.close()

    def load_from_metadata(self):
        pass

    def plot(self, percentile=5, cmap='viridis', **kwargs):
        spec_map = self.spec_im.sum(axis=-1)
        return self._plot(spec_map, percentile=5, cmap='viridis',
                          **kwargs)

    def _plot(self, map, title='', percentile=5, cmap='viridis', cbar=True,
              cbar_orientation='horizontal', cbar_position='bottom',
              show_scalebar=True):
        ax = plt.gca()
        X, Y = np.meshgrid(self.x_array, self.y_array)
        img = plt.pcolormesh(X, Y, map, cmap=cmap,
                             vmin=np.nanpercentile(map, percentile),
                             vmax=np.nanpercentile(map, 100-percentile))
        # img = plt.imshow(map, cmap=cmap,
        #                  vmin=np.percentile(map, percentile),
        #                  vmax=np.percentile(map, 100-percentile))
        plt.axis('equal')
        plt.axis('off')
        plt.title(title)
        if show_scalebar:
            scalebar = ScaleBar(self.x_array[1]-self.x_array[0], units='m')
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
            clone = self.copy()
            clone.spec_im = En_spec_im
            clone.spec_x_array = En
            clone.spec_units = 'eV'
            return clone
        elif self.spec_units == 'eV':
            return self
        else:
            raise Exception('Cannot convert from %s to eV' % self.spec_units)
            return None

    def to_index(self):
        clone = self.copy()
        clone.spec_x_array = np.linspace(len(self.spec_x_array))
        clone.spec_units = 'index'
        return clone

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

    def get_spec(self, loc=None, sum=False):
        if sum:
            return self.spec_im.sum(
                axis=tuple(range(len(self.spec_im.shape)))[:-1])
        else:
            assert isinstance(loc, tuple)
            if len(self.spec_im.shape) == 3:
                return self.spec_im[loc[0], loc[1], :]
            elif len(self.spec_im.shape) == 4:
                return self.spec_im[loc[0], loc[1], loc[2], :]


class CLImage:
    file_types = ['sync_raster_scan.h5', 'hyperspec_cl.h5']

    def __init__(self, fname='', **kwargs):
        self.load_h5_data(fname)
        self.load_from_metadata()

    def load_h5_data(self, fname):
        abspath = os.path.abspath(fname)
        try:
            f = h5.File(abspath)
            self.dat = h5_to_dictionary(f)
        except Exception as ex:
            print('Error loading from', fname, ex)
        finally:
            f.close()

    def load_from_metadata(self):
        dat = self.dat
        for meas in self.file_types:
            meas = meas[:-3]
            if meas in list(dat['measurement'].keys()):
                M = dat['measurement'][meas]
                self.description = M['settings']['description']
                hspan = M['settings']['h_span']
                vspan = M['settings']['v_span']
                self.adc_map = np.squeeze(M['adc_map']).copy()
                self.ctr_map = np.squeeze(M['ctr_map']).copy()
                ny, nx, nm = self.adc_map.shape
                H = dat['hardware']
                mag = H['sem_remcon']['settings']['magnification']
                srd = H['sync_raster_daq']['settings']
                whitelist = r'[^a-zA-Z0-9 ]+'
                self.adc_names = re.sub(whitelist, '',
                                        srd['adc_chan_names']).split()
                self.ctr_names = re.sub(whitelist, '',
                                        srd['ctr_chan_names']).split()
                frame_size = 114e-3/mag
                hspan = hspan/20.0*frame_size
                vspan = vspan/20.0*frame_size
                self.x_array = np.linspace(0.0, hspan, num=nx)
                self.y_array = np.linspace(0.0, vspan, num=ny)
                self.units = 'm'
                self.spec_units = 'nm'

    def _plot(self, map, percentile=5, cmap='viridis', **kwargs):
        ax = plt.gca()
        X, Y = np.meshgrid(self.y_array, self.x_array)
        plt.imshow(map, cmap=cmap,
                   vmin=np.percentile(map, percentile),
                   vmax=np.percentile(map, 100-percentile))
        plt.axis('equal')
        plt.axis('off')

        scalebar = ScaleBar(self.x_array[1]-self.x_array[0])
        plt.gca().add_artist(scalebar)
        return ax

    def get_adc(self, name):
        assert name in self.adc_names
        return self.adc_map[:, :, self.adc_names.index(name)]

    def get_ctr(self, name):
        assert name in self.ctr_names
        return self.ctr_map[:, :, self.ctr_names.index(name)]

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

    def __init__(self, fname='', dat=None):
        SpectralImage.__init__(self, fname=fname, dat=dat)

    def load_from_metadata(self):
        CLImage.load_from_metadata(self)
        M = self.dat['measurement'][self.file_types[0][:-3]]
        self.spec_im = np.squeeze(M['spec_map']).copy()
        self.spec_x_array = np.array(M['wls'])


class PLSpectralImage(SpectralImage):
    file_types = ['oo_asi_hyperspec_scan.h5', 'asi_OO_hyperspec_scan.h5',
                  'andor_asi_hyperspec_scan.h5']

    def load_from_metadata(self):
        try:
            dat = self.dat
            for meas in self.file_types:
                meas = meas[:-3]
                if meas in list(dat['measurement'].keys()):
                    self.name = str(dat['app']['settings']['sample'])
                    # print(self.name)
                    M = dat['measurement'][meas]
                    self.spec_x_array = np.array(M['wls'])
                    self.spec_im = np.squeeze(M['spec_map']).copy()
                    self.x_array = np.array(M['h_array'])
                    self.y_array = np.array(M['v_array'])
                    self.spec_units = 'nm'
                    self.units = 'mm'

                    # print('Sample: ' + self.name)
                    map_min = np.amin(self.spec_im)
                    if map_min < 0:
                        # print('Correcting negative minimum value in spec map:',
                        #       map_min)
                        self.spec_im += 1e-2 - map_min
                    # print('%d x %d spatial x %d spectral points' % (
                    #     len(self.x_array), len(self.y_array),
                    #     len(self.spec_x_array)))
        except Exception as ex:
            print('Error loading from dictionary', ex)


class PL3DSpectralImage(PLSpectralImage):
    # TODO - override to_signal

    file_types = ['oo_asi_hyperspec_3d_scan.h5']

    def load_from_metadata(self):
        PLSpectralImage.load_from_metadata(self)
        self.load_slices()
        # self.spec_im = self.spec_im[0, :, :, :]

    def load_slices(self):
        if hasattr(self, 'dict'):
            del self.dict
        try:
            dat = self.dat
            for meas in self.file_types:
                if meas[:-3] in list(dat['measurement'].keys()):
                    self.z_array = np.array(
                                   dat['measurement'][meas[:-3]]['z_array'])
        except Exception as ex:
            print('Error loading z_array values', ex)

        z_stack_map = np.array(self.spec_im)
        self.dict = OrderedDict()
        for kk in range(len(self.z_array)):
            key = '%0.4f' % self.z_array[kk]
            img = SpectralImage(dat=self.dat)
            img.spec_im = np.squeeze(z_stack_map[kk, :, :, :])
            img.spec_x_array = np.array(self.spec_x_array)
            img.x_array = np.array(self.x_array)
            img.y_array = np.array(self.y_array)
            img.spec_units = 'nm'
            img.units = 'mm'
            self.dict[key] = img

    def plot(self, percentile=5, cmap='viridis'):
        num_slices = np.size(self.z_array)
        slice_list = self.get_slice_list()
        num_rows = math.ceil(float(num_slices)/5)
        for kk in range(num_slices):
            plt.subplot(num_rows, 5, kk+1)
            self.get_slice(slice_list[kk]).plot(percentile=percentile,
                                                cmap=cmap)
            plt.title('z = %s mm' % slice_list[kk])
        return plt.gcf()

    def __getitem__(self, i):
        clone = PLSpectralImage.__getitem__(self, i)
        clone.load_slices()
        return clone

    def get_slice(self, key):
        assert key in list(self.dict.keys())
        return self.dict[key]

    def get_slice_list(self):
        return list(self.dict.keys())

    def to_signal(self):
        pass
