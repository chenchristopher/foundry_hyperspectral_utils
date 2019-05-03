from collections.abc import Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from hyperspy.signals import Signal1D, BaseSignal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py as h5
import scipy as sp
from hyperspy.misc.utils import DictionaryTreeBrowser
import re
from copy import copy
import math
from sklearn.decomposition import PCA, FastICA, NMF
import pyqtgraph as pg


def h5_to_dictionary(fh, tree=None, level=''):
    """
    Transforms hdf5 file tree into a DictionaryTreeBrowser

    Recursively generates a hyperspy.misc.utils.DictionaryTreeBrowser with the
    structure of the given h5py.File object.

    Parameters
    ----------
        fh : h5py.File
            h5py File handle
        tree : DictionaryTreeBrowser, optional
            A DictionaryTreeBrowser to append to recursively
        level : str, optional
            Location in hierarchy of the hdf5/DictionaryTreeBrowser

    Returns
    ----------
        DictionaryTreeBrowser
            dictionary object with hierarchy of given HDF5 file handle
    """
    assert tree is None or isinstance(tree, DictionaryTreeBrowser)

    if tree is None:
        tree = DictionaryTreeBrowser()
    else:
        level += '.'

    for key in list(fh.attrs.keys()):
        val = fh.attrs[key]
        tree.set_item(level + key, copy(val))

    for (key, desc) in list(fh.items()):
        key = str(key)
        desc = str(desc)
        if '(0 members)' in desc:
            continue
        elif 'HDF5 dataset' in desc:
            tree.set_item(level + key, np.squeeze(fh[key]).copy())
        elif 'HDF5 group' in desc:
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
    """
    Convert spectral image from wavelength in nm to energy in eV

    Given a spectral image with a spectral dimension in nm, apply the
    appropriate intensity correction and rebin the spectra into equally spaced
    energy bins with scipy.interpolate.interp1d. Also supports trimming
    wavelength range by array indicies or spectral values.

    Parameters
    ----------
        spec_map : numpy.ndarray
            Spectral map to be converted and rebinned
        wls : numpy.array
            Wavelength values
        spec_max : float, optional
            Maximum spectral value desired
        spec_min : float, optional
            Minimum spectral value desired
        ind_max : int, optional
            Maximum spectral index desired
        ind_min : int, optional
            Minimum spectral index desired

    Returns
    ----------
        En : numpy.array
            Spectral energy array in eV
        En_spec_map : numpy.ndarray
            Rebinned spectral map
        ne : int
            Number of spectral energy points
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


class Image:
    """
    Base class for an image or group of images.

    Container for loading and visualizing images. Supports visualization of
    images

    Attributes
    ----------
        file_types : list
            List of supported file file_types
        dat : DictionaryTreeBrowser
            metadata dictionary extracted from hdf tree
        x_array : numpy.array
            Array of spatial x values
        y_array : numpy.array
            Array of spatial y values
        units : str
            Spatial units

    Methods
    ----------
        load_h5_data(fname)
            Load the data from a given hdf5 file name
        load_from_metadata()
            Placeholder for subclasses to implement. Needs to assign values to
            all attributes except file_types and dat
        is_supported(fname)
            Checks to see if the string fname is in the list of supported file
            types
    """
    file_types = []

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
        pass

    def is_supported(self, fname):
        for ftype in self.file_types:
            if ftype in fname:
                return True
        return False

    def _plot(self, map, **kwargs):
        """
        Plots 2D map with spatial dimensions of the image

        Helper function which plots any map of the correct dimensions with the
        spatial dimensions of the image

        Arguments
        ----------
            map : numpy.ndarray)
                spatial map to be plotted with dimensions equal to the spatial
                dimensions of the spectral image
            title : str, optional
                title to be added to the plot
            percentile : int, optional
                percentile of data to include in colormap range
            vmin : float, optional
                minimum value of colormap range
            vmax : float, optional
                maximum value of colormap range
            cmap : str, optional
                colormap
            show_axes : bool, optional
                Show axes labels
            show_cbar : bool, optional
                Show colorbar
            cbar_orientation : {'horizontal', 'vertical'}, optional
                Colorbar orientation
            cbar_position : {'left', 'right', 'top', 'bottom'}, optional
                Colorbar position outside of axes
            show_scalebar : bool
                Show matplotlib_scalebar.ScaleBar
            scalebar_location : str or int
                position of scalebar

        Returns
        ----------
            matplotlib.axes

        """
        kwlist = list(kwargs.keys())

        pcolormesh_kwargs = {}
        if 'cmap' in kwlist:
            pcolormesh_kwargs['cmap'] = kwargs['cmap']
        else:
            pcolormesh_kwargs['cmap'] = 'viridis'

        if 'vmin' in kwlist:
            pcolormesh_kwargs['vmin'] = kwargs['vmin']
        elif 'percentile' in kwlist:
            pcolormesh_kwargs['vmin'] = np.nanpercentile(
                map, kwargs['percentile'])
        else:
            pcolormesh_kwargs['vmin'] = np.nanpercentile(map, 5)

        if 'vmax' in kwlist:
            pcolormesh_kwargs['vmax'] = kwargs['vmax']
        elif 'percentile' in kwlist:
            pcolormesh_kwargs['vmax'] = np.nanpercentile(
                map, 100-kwargs['percentile'])
        else:
            pcolormesh_kwargs['vmax'] = np.nanpercentile(map, 95)

        ax = plt.gca()
        X, Y = np.meshgrid(self.x_array, self.y_array)
        img = plt.pcolormesh(X, Y, map, **pcolormesh_kwargs)
        plt.axis('equal')

        if 'show_axes' in kwlist:
            if not kwargs['show_axes']:
                plt.axis('off')

        if 'title' in kwlist:
            plt.title(kwargs['title'])

        if 'show_scalebar' in kwlist:
            assert isinstance(kwargs['show_scalebar'], bool)
            if kwargs['show_scalebar']:
                scalebar_kwargs = {}
                if 'scalebar_position' in kwlist:
                    scalebar_kwargs['location'] = kwargs[
                        'scalebar_position']
                scalebar = ScaleBar(self.x_array[1]-self.x_array[0], units='m',
                                    **scalebar_kwargs)
                plt.gca().add_artist(scalebar)

        if 'show_cbar' in kwlist:
            assert isinstance(kwargs['show_cbar'], bool)
            if kwargs['show_cbar']:
                cbar_kwargs = {}
                if 'cbar_orientation' in kwlist:
                    cbar_kwargs['orientation'] = kwargs['cbar_orientation']
                if 'cbar_position' in kwlist:
                    cbar_kwargs['position'] = kwargs['cbar_position']
                colorbar(img, **cbar_kwargs)
        return ax

# TODO - update subclasses to only support 4D spec_im


class SpectralImage(Sequence, Image):
    """
    Base class for a spectral image.

    Container for loading and visualizing spectral images. Directly indexable
    by spectral x values. Capable of holding 2D and 3D spectral images, but
    only supports visualization of 2D spectral images.

    Attributes
    ----------
        file_types : list
            List of supported file file_types
        dat : DictionaryTreeBrowser
            metadata dictionary extracted from hdf tree
        spec_im : numpy.ndarray
            Spectral image array with shape
            (num z values, num y values, num x values, num spectral values)
        spec_x_array : numpy.array
            Spectral x array
        spec_units : str
            Spectral units
        x_array : numpy.array
            Array of spatial x values
        y_array : numpy.array
            Array of spatial y values
        z_array : numpy.array
            Array of spatial z values
        units : str
            Spatial units

    Methods
    ----------
        copy(signal=None)
            Return a copy of the SpectralImage. If signal is a
            hyperspy.Signal1D, the spectral image is overwritten with the data
            in signal.
        load_h5_data(fname)
            Load the data from a given hdf5 file name
        load_from_metadata()
            Placeholder for subclasses to implement. Needs to assign values to
            all attributes except file_types and dat
        plot()
            Plots a map of the spectral image summed along the spectral
            direction
        set_background()
        remove_background()
            Removes background correction from data
        to_energy()
            Converts and rebins from wavelength to energy. Returns a new
            SpectralImage or subclass with the same metadata.
        to_index()
            Converts spectral x array to indices
        to_signal()
            Returns a hyperspy.Signal1D with of the spectral image
        plot_spec()
            Plots a spectrum summed over all spatial axes
        get_spec()
        is_supported(fname)
            Checks to see if the string fname is in the list of supported file
            types
    """
    file_types = []

    def __init__(self, fname='', dat=None):
        """
        Initialization for SpectralImage can be from a filename or metadata
        dictionary.

        hdf5 filename is checked against list of supported file types.
        Dictionary supercedes loading from a file.

        Parameters
        ----------
            fname : str
                filename of hdf5 file to open
            dat : DictionaryTreeBrowser
                metadata dictionary

        """
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
            print('%d x %d x %d spatial x %d spectral points' % (
                len(self.z_array), len(self.x_array), len(self.y_array),
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

            spec_im = self.spec_im[:, :, :, ind_min:ind_max]
            spec_x_array = self.spec_x_array[ind_min:ind_max]
        else:
            ind = np.searchsorted(self.spec_x_array, i)
            if ind < 0:
                ind = 0
            elif ind >= len(self.spec_x_array):
                ind = len(self.spec_x_array) - 1

            spec_im = self.spec_im[:, :, :, ind]
            spec_x_array = self.spec_x_array[ind]

        return np.array(spec_im), np.array(spec_x_array)

    def __getitem__(self, i):
        spec_im, spec_x_array = self._getitem_helper(i)

        clone = self.copy()
        clone.spec_x_array = spec_x_array
        clone.spec_im = spec_im
        return clone

    def copy(self, signal=None):
        clone = type(self)(dat=self.dat)
        if signal is not None:
            assert isinstance(signal, Signal1D)
            clone.spec_im = np.empty((1,) + signal._data.shape)
            clone.spec_im[0, :, :, :] = signal._data
        else:
            clone.spec_im = np.array(self.spec_im)
        clone.spec_x_array = np.array(self.spec_x_array)
        clone.x_array = np.array(self.x_array)
        clone.y_array = np.array(self.y_array)
        clone.z_array = np.array(self.z_array)
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

    def plot(self, z_index=None, num_rows=1, fig_rows=1, **kwargs):
        """
            Visualize spectral image

            Visualizes 2D and 3D spectral images. Passes arguments to _plot

            Arguments
            ----------
                z_index : int, optional
                    Index of z value to display as a mappable
                num_rows : int, optional
                    Number of rows over which to display all 2D spectral
                    images in 3D spectral image
                **kwargs
                    Keyword arguments passed to _plot

            Raises
            ----------
                IndexError
                    Attempted to access z index larger than z array
        """
        nz = np.size(self.z_array)
        if z_index is None:
            if np.size(self.z_array) == 1:
                self.plot(z_index=0, **kwargs)
            else:
                num_cols = math.ceil(float(nz)/num_rows)
                for kk in range(nz):
                    plt.subplot(num_rows+fig_rows-1, num_cols, kk + 1)
                    self.plot(z_index=kk, **kwargs)
        else:
            assert isinstance(z_index, int)
            if z_index >= nz:
                err = 'Cannot access z_index %d in z_array size %d' % (
                    z_index, nz)
                raise IndexError(err)
            spec_map = self.spec_im[z_index, :, :, :].sum(axis=-1)
            self._plot(spec_map, **kwargs)

    def set_background(self, lims=(-1.0, -1.0), zero_negative_values=True,
                       append=False):
        """
            Background substraction of spectral images

            Fits a background over the given spectral range and/or sets the
            minimum spectra value to 0 if < 0. Can either establish a new
            background or append it to the existing background.

            Attributes
            ----------
            lims : tuple, optional
                Spectral values over which to fit a background
            zero_negative_values : bool, optional
                Set the minimum value of the spectral image to 0 if < 0
            append : bool, optional
                Whether or not to append the background parameters to the
                existing parameters or to start over
        """
        if hasattr(self, 'bkg') and not append:
            self.remove_background()

        if lims != (-1.0, -1.0):
            bkg_si = self[lims[0]:lims[1]]
            bkg = np.average(bkg_si.spec_im, axis=-1)
            self.bkg = np.empty(np.shape(self.spec_im))
            for kk in range(self.spec_im.shape[-1]):
                self.bkg[:, :, :, kk] = bkg
            self.spec_im -= self.bkg

        if zero_negative_values:
            spec_min = np.amin(self.spec_im)
            print(spec_min)
            if spec_min < 0:
                self.bkg[:, :, :, :] -= spec_min
                self.spec_im[:, :, :, :] -= spec_min

    def remove_background(self):
        """
            Removes background correction from the spectral image
        """
        assert hasattr(self, 'bkg')
        self.spec_im += self.bkg
        del self.bkg

    def is_supported(self, fname):
        """
            Checks file name against list of supported file types
        """
        for ftype in self.file_types:
            if ftype in fname:
                return True
        return False

    def to_energy(self):
        """
            Returns SpectralImage rebinned to energy units
        """
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
        """
            Returns SpectralImage with index as spectral unit
        """
        clone = self.copy()
        clone.spec_x_array = np.linspace(len(self.spec_x_array))
        clone.spec_units = 'index'
        return clone

    def to_signal(self):
        """
            Creates a hyperspy.Signal1D object with the same spectral image.

            The resulting hyperspy.Signal1D contains the full spectral image.

            Returns
            ----------
                s : hyperspy.Signal1D
                    hyperspy object
        """
        (nz, ny, nx, nf) = np.shape(self.spec_im)

        if self.units == 'mm':
            si_scale = 1e-3

        dx = self.x_array[1] - self.x_array[0]
        dy = self.y_array[1] - self.y_array[0]
        dz = self.z_array[1] - self.z_array[0]
        dx *= si_scale
        dy *= si_scale
        dz *= si_scale

        dx = pg.fn.siFormat(dx, suffix='m')
        dy = pg.fn.siFormat(dy, suffix='m')
        x_units = dx[-2:]
        y_units = dy[-2:]
        dx = float(dx[:-3])
        dy = float(dy[:-3])

        spec_name = 'index'
        if self.spec_units in ['nm', 'um']:
            spec_name = 'Wavelength'
        elif self.spec_units == 'eV':
            spec_name = 'E'

        dict_y = {'name': 'y', 'units': y_units,
                  'scale': dy,
                  'size': ny}
        dict_x = {'name': 'x', 'units': x_units,
                  'scale': dx,
                  'size': nx}
        dict_f = {'name': spec_name, 'units': self.spec_units,
                  'scale': self.spec_x_array[1] - self.spec_x_array[0],
                  'size': nf, 'offset': self.spec_x_array[0]}

        if nz == 1:
            s = Signal1D(self.spec_im, axes=[dict_y, dict_x, dict_f])
            s.change_dtype('float64')
            return s
        else:
            dz = pg.fn.siFormat(dz, suffix='m')
            z_units = dz[-2:]
            dz = float(dz[:-3])
            dict_z = {'name': 'z', 'units': z_units,
                      'scale': dz,
                      'size': nz}
            s = BaseSignal(self.spec_im, axes=[dict_z, dict_y, dict_x, dict_f])
            return s.as_signal1D(0)

    def plot_spec(self):
        self._plot_spec(self.get_spec(sum=True))

    def _plot_spec(self, spec):
        plt.plot(self.spec_x_array, spec)
        plt.xlabel(self.spec_units)

    def get_spec(self, loc=None, sum=False):
        if sum:
            return self.spec_im.sum(
                axis=tuple(range(len(self.spec_im.shape)))[:-1])
        else:
            assert isinstance(loc, tuple)
            assert len(tuple) == 4
            return self.spec_im[loc[0], loc[1], loc[2], :]


class CLImage(Image):
    file_types = ['sync_raster_scan.h5', 'hyperspec_cl.h5']

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


class CLSpectralImage(SpectralImage, CLImage):
    """
    Container for ScopeFoundry CL spectral images.

    Container for loading and visualizing CL spectral images. Directly
    indexable by spectral x values. Convenience functions for visualizing
    CL spectral images and spectra.

    Attributes
    ----------
        file_types : list
            List of supported file file_types
        dat : DictionaryTreeBrowser
            metadata dictionary extracted from hdf tree
        spec_im : numpy.ndarray
            Spectral image array with shape
            (num z values, num y values, num x values, num spectral values)
        spec_x_array : numpy.array
            Spectral x array
        spec_units : str
            Spectral units
        x_array : numpy.array
            Array of spatial x values
        y_array : numpy.array
            Array of spatial y values
        z_array : numpy.array
            Array of spatial z values
        units : str
            Spatial units

    Methods
    ----------
        copy(signal=None)
            Return a copy of the SpectralImage. If signal is a
            hyperspy.Signal1D, the spectral image is overwritten with the data
            in signal.
        load_h5_data(fname)
            Load the data from a given hdf5 file name
        load_from_metadata()
            Placeholder for subclasses to implement. Needs to assign values to
            all attributes except file_types and dat
        plot()
            Plots a map of the spectral image summed along the spectral
            direction
        set_background()
        remove_background()
            Removes background correction from data
        to_energy()
            Converts and rebins from wavelength to energy. Returns a new
            SpectralImage or subclass with the same metadata.
        to_index()
            Converts spectral x array to indices
        to_signal()
            Returns a hyperspy.Signal1D with of the spectral image
        plot_spec()
            Plots a spectrum summed over all spatial axes
        get_spec()
        is_supported(fname)
            Checks to see if the string fname is in the list of supported file
            types
    """
    file_types = ["hyperspec_cl.h5", ]

    def __init__(self, fname='', dat=None):
        SpectralImage.__init__(self, fname=fname, dat=dat)

    def load_from_metadata(self):
        CLImage.load_from_metadata(self)
        M = self.dat['measurement'][self.file_types[0][:-3]]
        self.spec_im = np.array(M['spec_map'])
        self.spec_x_array = np.array(M['wls'])
        self.z_array = [0.0]


class PLSpectralImage(SpectralImage):
    """
    Container for ScopeFoundry PL spectral images.

    Container for loading and visualizing CL spectral images. Directly
    indexable by spectral x values. Convenience functions for visualizing
    CL spectral images and spectra. Capable of handling both 2D and 3D
    spectral images.

    Attributes
    ----------
        file_types : list
            List of supported file file_types
        dat : DictionaryTreeBrowser
            metadata dictionary extracted from hdf tree
        spec_im : numpy.ndarray
            Spectral image array with shape
            (num z values, num y values, num x values, num spectral values)
        spec_x_array : numpy.array
            Spectral x array
        spec_units : str
            Spectral units
        x_array : numpy.array
            Array of spatial x values
        y_array : numpy.array
            Array of spatial y values
        z_array : numpy.array
            Array of spatial z values
        units : str
            Spatial units

    Methods
    ----------
        copy(signal=None)
            Return a copy of the SpectralImage. If signal is a
            hyperspy.Signal1D, the spectral image is overwritten with the data
            in signal.
        load_h5_data(fname)
            Load the data from a given hdf5 file name
        load_from_metadata()
            Placeholder for subclasses to implement. Needs to assign values to
            all attributes except file_types and dat
        plot()
            Plots a map of the spectral image summed along the spectral
            direction
        set_background()
        remove_background()
            Removes background correction from data
        to_energy()
            Converts and rebins from wavelength to energy. Returns a new
            SpectralImage or subclass with the same metadata.
        to_index()
            Converts spectral x array to indices
        to_signal()
            Returns a hyperspy.Signal1D with of the spectral image
        plot_spec()
            Plots a spectrum summed over all spatial axes
        get_spec()
        is_supported(fname)
            Checks to see if the string fname is in the list of supported file
            types
    """
    file_types = ['oo_asi_hyperspec_scan.h5', 'asi_OO_hyperspec_scan.h5',
                  'andor_asi_hyperspec_scan.h5', 'oo_asi_hyperspec_3d_scan.h5']

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
                    self.spec_im = np.array(M['spec_map'])
                    self.x_array = np.array(M['h_array'])
                    self.y_array = np.array(M['v_array'])
                    if 'z_array' in list(M.keys()):
                        self.z_array = np.array(M['z_array'])
                    else:
                        stage = dat['hardware']['asi_stage']['settings']
                        self.z_array = [stage['z_position']]
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
    """
    Experimental SpectralImage for native use of scikit-learn instead of using
    Hyperspy as an intermediary.
    """
    file_types = ['oo_asi_hyperspec_3d_scan.h5']

    def flatten(self, key=None):
        (nz, ny, nx, nf) = np.shape(self.spec_im)
        nf = np.size(self.spec_x_array)
        spec_im = self.spec_im
        if key is not None:
            spec_im = self.get_slice(key).spec_im
            nz = 1

        if hasattr(self, 'mask'):
            mask = self.mask['full']
            nm = np.count_nonzero(self.mask['base'].flatten())
            nonzero_inds = np.nonzero(mask.flatten())[0]
            return np.reshape(spec_im.flatten()[nonzero_inds], (nm*nz, nf))
        else:
            return np.reshape(spec_im, (nz*nx*ny, nf))

    def apply_mask(self, mask, **kwargs):
        if hasattr(self, 'mask'):
            del self.mask

        (nz, ny, nx, nf) = np.shape(self.spec_im)
        nf = np.size(self.spec_x_array)
        self.mask = {}
        self.mask['base'] = mask

        self.mask['loadings'] = np.empty((nz, ny*nx))
        for kk in range(nz):
            self.mask['loadings'][kk, :] = mask.flatten()
        self.mask['loadings'] = np.reshape(self.mask['loadings'], (nz, ny, nx))

        self.mask['full'] = np.empty((nz*ny*nx, nf))
        for ll in range(nf):
            self.mask['full'][:, ll] = self.mask['loadings'].flatten()
        self.mask['full'] = np.reshape(self.mask['full'], (nz, ny, nx, nf))
        print('spec_im', np.size(self.spec_im),
              'full mask', np.size(self.mask['full']))

    def remove_mask(self, mask):
        if hasattr(self, 'mask'):
            del self.mask

    def reshape(self, arr, key=None):
        arr = arr.T
        (nz, ny, nx, nf) = np.shape(self.spec_im)
        print('array shape', arr.shape)
        nc = arr.shape[0]  # number of components
        if key is not None:
            nz = 1

        data = arr
        if hasattr(self, 'mask'):
            mask = np.empty((nc, nz*ny*nx))
            for kk in range(nc):
                mask[kk, :] = self.mask['loadings'].flatten()

            data = np.empty(nc*nz*ny*nx)
            data[:] = np.nan
            print('data shape', data.shape)
            nonzero_inds = np.nonzero(mask.flatten())
            print(np.shape(nonzero_inds), type(nonzero_inds))
            data[nonzero_inds[0]] = arr.flatten()

        print(data.shape, (nc, nz, ny, nx))
        data = np.reshape(data, (nc, nz, ny, nx))
        return np.squeeze(data)

    def fit_pca(self, **kwargs):
        if hasattr(self, 'decomp'):
            del self.decomp
        if hasattr(self, 'decomp_comps'):
            del self.decomp_comps
        if hasattr(self, 'decomp_loadings'):
            del self.decomp_loadings
        flatten_kwargs = {}
        if 'key' in list(kwargs.keys()):
            flatten_kwargs['key'] = kwargs.pop('key')

        self.decomp = PCA(**kwargs)
        self.decomp.fit(self.flatten(**flatten_kwargs))
        self.decomp_comps = self.decomp.components_

    def plot_explained_variance_ratio(self, newfig=True):
        assert hasattr(self, 'decomp')
        assert isinstance(self.decomp, PCA)
        assert hasattr(self.decomp, 'explained_variance_ratio_')
        if newfig:
            plt.figure()
        xvals = np.array(range(len(self.decomp.explained_variance_ratio_))) + 1
        plt.semilogy(xvals, self.decomp.explained_variance_ratio_, 'b.')
        plt.xlabel('component')
        plt.ylabel('explained variance ratio')

    def transform_decomp(self, **kwargs):
        assert hasattr(self, 'decomp')
        if hasattr(self, 'decomp_result'):
            del self.decomp_result

        flatten_kwargs = {}
        if 'key' in list(kwargs.keys()):
            flatten_kwargs['key'] = kwargs.pop('key')

        self.decomp_loadings = self.decomp.transform(
            self.flatten(**flatten_kwargs))

    def blind_source_separation(self, **kwargs):
        assert hasattr(self, 'decomp')
        assert isinstance(self.decomp, PCA)
        if hasattr(self, 'ica'):
            del self.ica
            del self.ica_comps
            del self.ica_loadings

        if 'number_of_components' in list(kwargs.keys()):
            kwargs['n_components'] = kwargs.pop('number_of_components')

        if 'on_loadings' in list(kwargs.keys()):
            kwargs.pop('on_loadings')

        key_kwarg = {}
        if 'key' in list(kwargs.keys()):
            key_kwarg['key'] = kwargs.pop('key')

        self.ica = FastICA(**kwargs)
        self.ica_loadings = self.reshape(
            self.ica.fit_transform(self.decomp_loadings, **key_kwarg))
        self.ica_comps = np.dot(self.ica.mixing_.T, self.decomp.components_)

    def get_bss_loadings(self):
        assert hasattr(self, "ica_loadings")
        return self.ica_loadings

    def get_bss_factors(self):
        assert hasattr(self, 'ica_comps')
        return self.ica_comps

    def decomposition(self, **kwargs):
        assert 'algorithm' in list(kwargs.keys())
        algorithm = kwargs.pop('algorithm')

        assert 'output_dimension' in list(kwargs.keys())
        kwargs['n_components'] = kwargs.pop('output_dimension')

        key_kwarg = {}
        if 'key' in list(kwargs.keys()):
            key_kwarg['key'] = kwargs['key']

        if algorithm == 'svd':
            self.fit_pca(**kwargs)
        elif algorithm == 'nmf':
            self.fit_nmf(**kwargs)
        else:
            raise Exception('Invalid algorithm')

        self.transform_decomp(**key_kwarg)

    def get_decomposition_loadings(self):
        assert hasattr(self, 'decomp')
        assert hasattr(self.decomp_loadings)
        return self.reshape(self.decomp_loadings)

    def get_decomposition_factors(self):
        assert hasattr(self, 'decomp')
        return self.decomp_comps

    def fit_nmf(self, **kwargs):
        if hasattr(self, 'decomp'):
            del self.decomp
            del self.decomp_comps
        if hasattr(self, 'decomp_loadings'):
            del self.decomp_loadings
        flatten_kwargs = {}
        if 'key' in list(kwargs.keys()):
            flatten_kwargs['key'] = kwargs.pop('key')

        self.decomp = NMF(**kwargs)
        self.decomp_comps = self.decomp.components_
        self.decomp.fit(self.flatten(**flatten_kwargs))
