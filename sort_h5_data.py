import os
import h5py as h5
import fnmatch

def getSample(h5_fhandle):
    return str(h5_fhandle['app']['settings'].attrs['sample'])

if __name__ == '__main__':
    import sys

    listOfFiles = os.listdir('.')
    path = os.getcwd()
    print('pwd ' + path)
    for fname in listOfFiles:
        if fnmatch.fnmatch(fname,'*.h5'):
            try:
                fh = h5.File(fname)
                sample = getSample(fh).strip()
                if sample == '':
                    sample = 'no_name'
                print(fname,sample)
                fh.close()
            except Exception as err:
                print('Could not open ' + fname)
                continue

            try:
                if not os.path.isdir(path + '\\' + sample):
                    print('create directory ' + path + '\\' + sample)
                    os.mkdir(path + '\\' + sample)
                print('moving file to ' + path + '\\' + sample + '\\' + fname )
                os.rename(path + '\\' + fname, path + '\\' + sample + '\\' + fname)
            except Exception as err:
                print('Could not move ' + fname)

            print()
