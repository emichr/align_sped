from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import sped_align as sa
import hyperspy.api as hs
from scipy.io.wavfile import write
from skimage.transform import warp_coords
from scipy.ndimage import map_coordinates


def shift(xy):
    global Popts_1, frameno
    return xy - np.array([72 - Popts_1[frameno, 1], 72 - Popts_1[frameno, 2]])[None, :]


#@profile
def align_stack(stack, popts, limit=None):
    n_cols = stack.axes_manager['x'].get('size')['size']
    n_rows = stack.axes_manager['y'].get('size')['size']

    n_total = n_cols * n_rows

    if limit is None:
        limit = n_total

    n_dcols = stack.axes_manager['dx'].get('size')['size']
    n_drows = stack.axes_manager['dy'].get('size')['size']

    new_stack = np.zeros((n_rows, n_cols, n_drows, n_dcols), dtype=np.uint8)

    colno = 0
    rowno = 0

    for frameno, frame in enumerate(s):
        global frameno
        if frameno == limit:
            break
        if colno == n_cols:
            colno = 0
            rowno += 1
        if not frameno % (int(limit / 100)):
            print(
                '{}% done (now on Frame {} of {}: row {}, col {})'.format(
                    int(frameno / limit * 100), frameno,
                    limit, rowno, colno))

        image = frame.data.copy()

        coords = warp_coords(shift, image.shape)
        new_stack[rowno, colno, :, :] = map_coordinates(image, coords)

        colno += 1
    s_shifted = hs.signals.Signal2D(new_stack)
    return s_shifted

if __name__ == '__main__':
    s = hs.load('C:\\Users\\emilc\\Desktop\\SPED_align_laptop\\2017_01_11_6060-20-1-C-4_001.blo')






    #s = hs.load('C:\\Users\\emilc\\Desktop\\SPED_align_laptop\\2017_01_11_6060-20-1-C-4_001.blo')

    #Popts_1 = np.load(
    #    'C:\\Users\\emilc\\PhD\\05-Experiments\\01-SubgrainFormationInPFZ\\04-Outputs\\02-TEM\\04-ASTAR\\2017_01_11_6060-20-1-C-4_001\\Popts_1.npy')

    #g = hs.load('C:\\Users\\emilc\\PhD\\05-Experiments\\01-SubgrainFormationInPFZ\\04-Outputs\\02-TEM\\04-ASTAR\\2017_01_11_6060-20-1-C-4_001\\G.hdf5')
    #g_aligned = align_stack(g, Popts_1)
    #g_aligned.save('C:\\Users\\emilc\\PhD\\05-Experiments\\01-SubgrainFormationInPFZ\\04-Outputs\\02-TEM\\04-ASTAR\\2017_01_11_6060-20-1-C-4_001\\2017_01_11_6060-20-1-C-4_001_g_aligned.hdf5')
    #g_aligned.plot(cmap=plt.get_cmap('RdBu'))

    #s_aligned = align_stack(s, Popts_1)
    #s_aligned.save('C:\\Users\\emilc\\PhD\\05-Experiments\\01-SubgrainFormationInPFZ\\04-Outputs\\02-TEM\\04-ASTAR\\2017_01_11_6060-20-1-C-4_001\\2017_01_11_6060-20-1-C-4_001_aligned.hdf5')
    #s_aligned = hs.load('C:\\Users\\emilc\\PhD\\05-Experiments\\01-SubgrainFormationInPFZ\\04-Outputs\\02-TEM\\04-ASTAR\\2017_01_11_6060-20-1-C-4_001\\2017_01_11_6060-20-1-C-4_001_aligned.hdf5')
    #s.plot(cmap=plt.get_cmap('RdBu'))
    #s_aligned.plot(cmap=plt.get_cmap('RdBu'))
    #g.plot(cmap=plt.get_cmap('RdBu'))
    #plt.show()
