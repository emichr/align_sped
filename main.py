from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import sped_align as sa

N = 50
M = 70

Xo = 20
Yo = 40

Sx = 3
Sy = 6

Rot = 27

C = 10

X, Y = np.mgrid[:N, :M]
test_data = sa.gaussian_2d((X, Y), 100.0, Xo, Yo, Sx, Sy, Rot, C).reshape(np.shape(X)) \
    #         + sa.gaussian_2d((X, Y), 100.0,
    #                                                                                                     Xo + 3, Yo + 3,
    #                                                                                                     2, 3, -Rot,
    #                                                                                                     C).reshape(
    # np.shape(X))

fit_result = sa.fit_gaussian_2d_to_imagesubset(test_data, subset_bounds=(5, 35, 20, 60))

g_1 = sa.gaussian_2d((X, Y), *fit_result['parameters']).reshape(np.shape(X))
g_2 = sa.gaussian_2d((fit_result['x'], fit_result['y']), *fit_result['parameters']).reshape(np.shape(fit_result['x']))

fig, axes = plt.subplots(1, 2)
axes[0].imshow(test_data, interpolation='nearest', cmap=plt.get_cmap('RdBu'))
sa.add_contour(X, Y, g_1, axes[0])

axes[1].imshow(
    test_data[fit_result['x min']:fit_result['x max'] + 1, fit_result['y min']:fit_result['y max'] + 1],
    interpolation='nearest', cmap=plt.get_cmap('RdBu'),
    extent=fit_result['extent'])

sa.add_contour(fit_result['x'], fit_result['y'], g_2, axes[1])
plt.show()
