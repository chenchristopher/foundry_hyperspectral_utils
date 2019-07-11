from spec_im import PLSpectralImage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, NMF


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
