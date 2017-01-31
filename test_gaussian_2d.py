from sped_align import gaussian_2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def test_gaussian_2d_size():
    n = 10
    m = 15
    x, y = np.mgrid[0:n, 0:m]
    g = gaussian_2d((x, y), 1.0, 5.0, 5.0, 1.0, 1.0, 0.0, 0.0).reshape(np.shape(x))

    assert len(g[:, 0]) == n
    assert len(g[0, :]) == m


def test_gaussian_2d_center():
    n = 20
    m = 20
    x, y = np.mgrid[0:n, 0:m]
    for xo in np.linspace(5, 6.5):
        for yo in np.linspace(5, 6.5):
            g = gaussian_2d((x, y), 1000.0, xo, yo, 100, 100, 0, 0).reshape(np.shape(x))

            assert np.unravel_index(np.argmax(g), (n, m)) == (
                int(round(xo, ndigits=0)), int(round(yo, ndigits=0))), print(
                'AssertionError on testing gaussian 2d center:\n\tmax at: {0} should be at {1}\n{2}'.format(
                    np.unravel_index(np.argmax(g), (n, m)), (xo, yo), g))

def test_gaussian_2d_fittable():
    n = 100
    m = 100
    xo = 20
    yo = 10
    x, y = np.mgrid[0:n, 0:m]
    a = 1000.0
    g = gaussian_2d((x, y), a, xo, yo, 5, 5, 0, 0).reshape(np.shape(x))
    try:
        popt, pcov = curve_fit(gaussian_2d, (x, y), g.ravel(), p0=[a, xo, yo, 100, 100, 0, 0],
                               bounds=([0.0, 0.0, 0.0, 0.0, 0.0, -45.0, 0.0],
                                       [a * 10.0, float(n), float(m), np.inf, np.inf, 45.0, np.inf]))
    except:
        assert False
