from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import sped_align as sa


if __name__ == '__main__':
    N, M = 256, 256  # Define width of image
    X, Y = np.mgrid[:256, :256]  # Represent image pixels by two NxM arrays
    A, Vo, Ho, Vw, Hw, Rot, C = 254, 100, 180, 4, 8, 15, 0  # Define gaussian parameters amplitude, vertical origin, horizontal origin, vertical width, horizontal width, rotation and baseline, respectively.
    G = sa.gaussian_2d((X, Y), 254, 100, 180, 4, 8, 15, 0).reshape(np.shape(X))  # Compute twodimensional gaussian
    fitresult = sa.fit_gaussian_2d_to_imagesubset(G, subset_bounds=(50, 150, 100, 200), p0=[A, Vo, Ho, Vw, Hw, Rot, C])

    print(fitresult['parameters'])

    g_1 = sa.gaussian_2d((X, Y), *fitresult['parameters']).reshape(np.shape(X))

    g_2 = sa.gaussian_2d((fitresult['x'], fitresult['y']), *fitresult['parameters']).reshape(np.shape(fitresult['x']))
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(G, interpolation='nearest', cmap=plt.get_cmap('RdBu'))
    sa.add_contour(X, Y, g_1, axes[0])

    axes[1].imshow(
        G[fitresult['x min']:fitresult['x max'] + 1, fitresult['y min']:fitresult['y max'] + 1],
        interpolation='nearest', cmap=plt.get_cmap('RdBu'),
        extent=fitresult['extent'])

    sa.add_contour(fitresult['x'], fitresult['y'], g_2, axes[1])
    plt.show()



