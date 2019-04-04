from all_funcs_aug15 import sortfnames
from spec_im_utils import plot_cl_summary, plot_si_bands, plot_bss_results, plot_decomp_results
from spec_im import CLSpectralImage
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import os


def summarize_all_from_directory(dir='.'):
    """
    Summarizes all of the sync_raster_scan_h5 and hyperspec_cl_h5 files in a
    given directory.

    Using a modified version of Shaul's all_funcs_aug15.py, this function
    reports the CL spectral maps and images in the directory given. All input
    is converted into absolute paths.

    Keyword Args:
        dir (str): Path to directory with files. Can be a relative path.

    """
    full_path = os.path.abspath(dir)
    (clmaps, clims, hdf5s, txts, sifs) = sortfnames(dir=full_path)
    print('CL Maps', clmaps)
    print('CL Images', clims)
    #make_image_summaries(clims, full_path)
    #make_si_summary(clmaps, full_path)


def recursive_summarize(dir='.', threshold=1e-4):
    """
    Summarizes all of the sync_raster_scan_h5 and hyperspec_cl_h5 file in a
    given directory, recursively checking all levels of subdirectories.

    Calls summarize_all_from_directory on the given directory and all levels of
    contained subdirectories within. Dependent on a modified version of Shaul's
    all_funcs_aug15. Although summarize_all_from_directory also handles
    relative and absolute paths equally well, an absolute path is used because
    it is reported to standard output.

    Keyword Args:
        dir (str): Path to directory with files. Can be a relative path.

    """
    full_path = os.path.abspath(dir)
    print('Summarizing %s' % full_path)

    with os.scandir(full_path) as it:
        for entry in it:
            if entry.is_dir():
                recursive_summarize(entry.path, threshold=threshold)
            else:
                for ftype in CLSpectralImage.file_types:
                    if ftype in entry.path:
                        try:
                            make_si_summary(
                                entry.path, save_figs=True, threshold=threshold)
                        except Exception as ex:
                            print('Error, ', ex)
                            plt.close(fig='all')
                        print()
                        break

    print('completed.')


def make_si_summary(fname, save_figs=False, threshold=1e-5):
    fname = os.path.abspath(fname)
    print(fname)
    try:
        si = CLSpectralImage(fname)
    except Exception as ex:
        print('Could not open %s' % fname, ex)
        return

    si = si.to_energy()

    f0 = plot_cl_summary(si)

    f1 = plot_si_bands(si,
                       (2.5, 2.6, 'viridis', ''),
                       (2.6, 2.7, 'viridis', ''),
                       (2.7, 2.8, 'viridis', ''),
                       (2.8, 2.9, 'viridis', ''),
                       (2.9, 3.0, 'viridis', ''),
                       (3.0, 3.1, 'viridis', ''),
                       (3.1, 3.2, 'viridis', ''),
                       (3.3, 3.4, 'viridis', ''),
                       (3.2, 3.3, 'viridis', ''),
                       (3.4, 3.5, 'viridis', ''),
                       no_of_cols=5, percentile=3)

    f1.set_size_inches(10, 7)
    f1.tight_layout(h_pad=1)

    s = si.to_signal()

    s.decomposition(algorithm='svd')
    f2 = s.plot_explained_variance_ratio(threshold=threshold)

    components = np.searchsorted(1/s.get_explained_variance_ratio().data,
                                 1/threshold)
    if components >= 15:
        print('Too many components (%s) for threshold %f.' %
              (components, threshold))
        print('15 components')
        components = 15
    else:
        print('%d components' % components)

    f3 = plot_bss_results(
        s, si, title='BSS', cmap='viridis', fig_rows=4, components=components,
        max_iter=1000)
    f4 = plot_decomp_results(
        s, si, title='NMF', cmap='viridis', fig_rows=4,
        algorithm='nmf', output_dimension=components)
    if save_figs:
        save_all_figs(fname[:-3]+'_decomp.pdf')


def save_all_figs(fname):
    pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
    for fig in range(1, plt.figure().number):
        # will open an empty extra figure :(
        pdf.savefig(fig)
    pdf.close()
    plt.close(fig='all')
