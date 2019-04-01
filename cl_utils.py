from all_funcs_aug15 import *
import os


def summarize_all_from_directory(dir):
    full_path = os.path.abspath(dir)
    clmaps, clims, hdf5s, txts, sifs = sortfnames(dir=full_path)
    print('CL Maps', clmaps)
    print('CL Images', clims)
    make_image_summaries(clims, full_path)
    make_si_summaries(clmaps, full_path)


def recursive_summarize(dir):
    full_path = os.path.abspath(dir)
    print('Summarizing %s' % full_path)
    summarize_all_from_directory(dir)
    with os.scandir(full_path) as it:
        for entry in it:
            if entry.is_dir():
                recursive_summarize(entry.path)
