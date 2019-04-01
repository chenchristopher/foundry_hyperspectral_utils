from all_funcs_aug15 import make_image_summaries, make_si_summaries, sortfnames
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
    make_image_summaries(clims, full_path)
    make_si_summaries(clmaps, full_path)


def recursive_summarize(dir='.'):
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
    summarize_all_from_directory(dir)
    with os.scandir(full_path) as it:
        for entry in it:
            if entry.is_dir():
                recursive_summarize(entry.path)
