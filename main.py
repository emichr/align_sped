from __future__ import division
import matplotlib.pyplot as plt
import sped_align as sa
import hyperspy.api as hs

if __name__ == '__main__':
    #Load raw data:
    s = hs.load('C:\\Users\\emilc\\Desktop\\SPED_align_laptop\\2017_01_11_6060-20-1-C-4_001.blo')
    #Should be shape (N, M, n, m), where N and M are the scan pipxels and n and m are the diffraction pattern dimensions
    #Define where outputs should be saved:
    savedir = 'C:\\Users\\emilc\\PhD\\05-Experiments\\01-SubgrainFormationInPFZ\\04-Outputs\\02-TEM\\04-ASTAR\\2017_01_11_6060-20-1-C-4_001\\'

    #Run alignment procedure. Returns a Numpy.nd array of shape (N*M, 7) that is stored in Popts (also in a .npy file), the gaussians that was used to fit each 000 reflection in g, and the aligned signal in s_aligned:
    Popts, g, s_aligned = sa.align_stack_fast(s, limit=None, save=True, savedir=savedir)

    #Plot the outputs:
    g.plot()
    s.plot()
    s_aligned.plot()
    plt.show()
