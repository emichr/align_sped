from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import sped_align as sa
import hyperspy.api as hs
from scipy.io.wavfile import write
from skimage.transform import warp_coords
from scipy.ndimage import map_coordinates

if __name__ == '__main__':
    s = hs.load('C:\\Users\\emilc\\Desktop\\SPED_align_laptop\\2017_01_11_6060-20-1-C-4_001.blo')

    savedir = 'C:\\Users\\emilc\\PhD\\05-Experiments\\01-SubgrainFormationInPFZ\\04-Outputs\\02-TEM\\04-ASTAR\\2017_01_11_6060-20-1-C-4_001\\'
    Popts, g, s_aligned = sa.align_stack_fast(s, limit=None, save=True, savedir=savedir)

    #print(s.metadata)

    #print(g.metadata)

    #print(s_aligned.metadata)

    #g.plot()
    s.plot()
    s_aligned.plot()
    plt.show()
