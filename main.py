from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import sped_align as sa
import hyperspy.api as hs


if __name__ == '__main__':
    s = hs.load('C:\\Users\\emilc\\Desktop\\SPED_align_laptop\\2017_01_11_6060-20-1-C-4_001.blo')

    print(type(s))
    print(s.original_metadata)
    lim = 1800-1

    Popts_1, Popts_2 = sa.align_stack_fast(s, limit=None, bounds=10, save=True)


    x_shift_1 = Popts_1[:lim+1, 1]
    y_shift_1 = Popts_1[:lim+1, 2]
    #
    x_shift_2 = Popts_2[:, :,  1]
    y_shift_2 = Popts_2[:, :,  2]

    # align_parameters = sa.align_stack_fast(s, limit=400, print_frameno=True, debug=False, subset_bounds=10)
    # print(align_parameters['Popt'])
    # x_shift = align_parameters['Popt'][:, :, 1]
    # y_shift = align_parameters['Popt'][:, :, 2]

    f, ax = plt.subplots(1, 1)
    ax.plot(x_shift_1, 'r')
    ax.plot(y_shift_1, 'b')

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(x_shift_2, interpolation='nearest', cmap=plt.get_cmap('RdBu'))
    ax[1].imshow(y_shift_2, interpolation='nearest', cmap=plt.get_cmap('RdBu'))
    #ax[0].imshow(x_shift, interpolation='nearest')
    #ax[1].imshow(y_shift, interpolation='nearest')

    plt.show()

    # N, M = 256, 256  # Define width of image
    # X, Y = np.mgrid[:256, :256]  # Represent image pixels by two NxM arrays
    # A, Vo, Ho, Vw, Hw, Rot, C = 254, 100, 180, 4, 8, 15, 0  # Define gaussian parameters amplitude, vertical origin, horizontal origin, vertical width, horizontal width, rotation and baseline, respectively.
    # G = sa.gaussian_2d((X, Y), 254, 100, 180, 4, 8, 15, 0).reshape(np.shape(X))  # Compute twodimensional gaussian
    # fitresult = sa.fit_gaussian_2d_to_imagesubset(G, subset_bounds=(50, 150, 100, 200), p0=[A, Vo, Ho, Vw, Hw, Rot, C])
    #
    # print(fitresult['parameters'])
    #
    # g_1 = sa.gaussian_2d((X, Y), *fitresult['parameters']).reshape(np.shape(X))
    #
    # g_2 = sa.gaussian_2d((fitresult['x'], fitresult['y']), *fitresult['parameters']).reshape(np.shape(fitresult['x']))
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(G, interpolation='nearest', cmap=plt.get_cmap('RdBu'))
    # sa.add_contour(X, Y, g_1, axes[0])
    #
    # axes[1].imshow(
    #     G[fitresult['x min']:fitresult['x max'] + 1, fitresult['y min']:fitresult['y max'] + 1],
    #     interpolation='nearest', cmap=plt.get_cmap('RdBu'),
    #     extent=fitresult['extent'])
    #
    # sa.add_contour(fitresult['x'], fitresult['y'], g_2, axes[1])
    # plt.show()



