from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import warnings

import hyperspy.api as hs
from scipy.optimize import curve_fit, OptimizeWarning


def check_gaussian_2d_inputs(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    try:
        theta = float(theta * np.pi / 180.0)
    except ValueError as e:
        print('Value error in {}:\n\tcould not convert theta {} to float: {}'.format(theta, e))
        return {'ok inputs': False}

    try:
        (x_pos, y_pos) = xdata_tuple
    except ValueError as e:
        print('Value error in {}: could not extract x_pos and y_pos from x_data tuple {}: {}'.format(r'gaussian_2d()',
                                                                                                     xdata_tuple, e))
        return {'ok inputs': False}
    try:
        xo = float(xo)
    except ValueError as e:
        print('Value error in {}: xo {} is not a valid float\n\t{}'.format(r'gaussian_2d()', xo, e))
        return {'ok inputs': False}

    try:
        yo = float(yo)
    except ValueError as e:
        print('Value error in {}: yo {} is not a valid float\n\t{}'.format(r'gaussian_2d()', yo, e))
        return {'ok inputs': False}

    try:
        sigma_x = float(sigma_x)
    except ValueError as e:
        print('Value error in {}: xo {} is not a valid float\n\t{}'.format(r'gaussian_2d()', sigma_x, e))
        return {'ok inputs': False}

    try:
        sigma_y = float(sigma_y)
    except ValueError as e:
        print('Value error in {}: xo {} is not a valid float\n\t{}'.format(r'gaussian_2d()', sigma_y, e))
        return {'ok inputs': False}

    try:
        offset = float(offset)
    except ValueError as e:
        print('Value error in {}: offset {} is not a valid float\n\t{}'.format(r'gaussian_2d()', offset, e))
        return {'ok inputs': False}

    try:
        amplitude = float(amplitude)
    except ValueError as e:
        print('Value error in {}: amplitude {} is not a valid float\n\t{}'.format(r'gaussian_2d()', amplitude, e))
        return {'ok inputs': False}

    return {'vertical positions': x_pos, 'horizontal positions': y_pos, 'amplitude': amplitude, 'vertical origin': xo,
            'horizontal origin': yo, 'vertical sigma': sigma_x, 'horizontal sigma': sigma_y, 'rotation': theta,
            'offset': offset, 'ok inputs': True}


def gaussian_2d(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    When xdata_tuple is on the form (X, Y), where X, Y = np.mgrid(0:vertical_width, 0:horizontal_width), the gaussian position is at the column where X = x0 and at the row where Y = y0
    :param xdata_tuple: Tuple on the form (X, Y) where X and Y are nxm matrices. X contains array row (x) coordinates while Y contains column (y) coordinates
    :param amplitude: Amplitude factor of gaussian
    :param xo: row position of gaussian (where X = x0)
    :param yo: column position of gaussian (where Y = y0)
    :param sigma_x: Width of gaussian in row direction (x)
    :param sigma_y: Width of gaussian in column direction (y)
    :param theta: rotation of gaussian, measured clockwise from horizontal, in degrees
    :param offset: Constant offset (baseline) of gaussian
    :return: A raveled numpy array containin Z values of gaussian. To get on 2d array form call .reshape(np.shape(X)) on output
    """

    checked_inputs = check_gaussian_2d_inputs(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    try:
        assert checked_inputs['ok inputs']
        try:
            x_pos = checked_inputs['vertical positions']
            y_pos = checked_inputs['horizontal positions']
            amplitude = checked_inputs['amplitude']
            xo = checked_inputs['vertical origin']
            yo = checked_inputs['horizontal origin']
            sigma_x = checked_inputs['vertical sigma']
            sigma_y = checked_inputs['horizontal sigma']
            theta = checked_inputs['rotation']
            offset = checked_inputs['offset']
        except KeyError as e:
            print('Something went wrong when unpacking values from checked_inputs in gaussian_2d():{}\n\t{}'.format(
                checked_inputs, e))
        try:
            a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
            b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
            c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
            g = offset + amplitude * np.exp(- (a * ((x_pos - xo) ** 2) + 2 * b * (x_pos - xo) * (y_pos - yo)
                                               + c * ((y_pos - yo) ** 2)))
            return g.ravel()
        except ValueError as e:
            print('{}'.format(
                e))  # Value error in {}: Could not compute result, returning None. {}'.format(r'gaussian_2d()', e))
            return None
    except AssertionError as e:
        print('Input to gaussian_2d() did not pass check: {}'.format(checked_inputs))


def fit_gaussian_2d_to_imagesubset(image, subset_bounds=(None, None, None, None),
                                   p0=[None, None, None, None, None, None, None], retryfitting=True, debug=False):
    '''Fit a twodimensional gaussian to subset of image.

    Provide a twodimensional array and a region on which to fit a twodimensional gaussian. The gaussian has two
    independent, perpendicular, widths, and is rotated by conventional rotation transform.

    Parameters
    ----------
    image : array_like
    subset_bounds : 4-tuple of int or float, optional
        (the default is (None, None, None, None) which implies fitting on the whole image).
    p0 : 7-list of int or float, optional
        Initial guess provided to `scipy.optimize.curvefit()` (the default is [None, None, None, None, None, None, None]
        which implies using details of the image subset as a guess: [maximum, width_x, width_y, 1, 1, 0, 0]).
    retryfitting : bool, optional
        If fitting using the provided initial guess `p0` failes, retry the fitting using the default values of `p0`
        (the default is True).
    debug : bool, optional
        Provide print statements as output on **some** errors, warnings or issues.

    Returns
    -------
    dict
        Dictionary containing results from the fitting, as well as row and column position arrays, the bounds used for
        the subset, and an extent for use with `matplotlib.pyplot.imshow`. The keys are `parameters`,
        `parameter uncertainties`, `x`, `y`, `bounds`, `x min`, `x max`, `y min`, `y max`, `extent`.

    '''
    (wx, wy) = np.shape(image)

    bounds = [0, wx, 0, wy]
    for i, bound in enumerate(subset_bounds):
        if bound is not None:
            try:
                bounds[i] = int(bound)
            except ValueError as e:
                bounds[i] = bounds[i]
                if debug:
                    print(
                        'Error in fit_gaussian_2d_to_imagesubset():\n\tConverting bound number {} in {} failed - could not convert to integer. Set to default {}\n{}\n'.format(
                            i, subset_bounds, bounds[i], e))
        else:
            if debug:
                print(
                    'Warning in fit_gaussian_2d_to_imagesubset():\n\tNo subset bound for bound no {} provided, using default: {}\n'.format(
                        i, bounds[i]))

    subset = image[bounds[0]:bounds[1] + 1, bounds[2]:bounds[3] + 1].copy()
    x, y = np.mgrid[bounds[0]:bounds[1] + 1, bounds[2]:bounds[3] + 1]

    try:
        assert x.min() == bounds[0], 'minimum of "x" ({}) is not equal to "bounds[0]" ({})'.format(x.min(), bounds[0])
    except AssertionError as e:
        print(e)
    try:
        assert x.max() == bounds[1], 'maximum of "x" ({}) is not equal to "bounds[1]" ({})'.format(x.max(), bounds[1])
    except AssertionError as e:
        print(e)
    try:
        assert y.min() == bounds[2], 'minimum of "y" ({}) is not equal to "bounds[2]" ({})'.format(y.min(), bounds[2])
    except AssertionError as e:
        print(e)
    try:
        assert y.max() == bounds[3], 'maximum of "y" ({}) is not equal to "bounds[3]" ({})'.format(x.max(), bounds[3])
    except AssertionError as e:
        print(e)

    print(bounds)

    tmp_p0 = [subset.max(), int((bounds[1] - bounds[0]) / 2) + bounds[0], int((bounds[3] - bounds[2]) / 2) + bounds[2],
              1, 1, 0, 0]

    try:
        for i, guess in enumerate(p0):
            if guess is not None:
                try:
                    p0[i] = float(guess)
                except ValueError as e:
                    p0[i] = tmp_p0[i]
                    if debug:
                        print(
                            'Error in fit_gaussian_2d_to_imagesubset():\n\tCannot convert guess paramater no {} ({})to float, using default {}\n{}\n'.format(
                                i, guess, p0[i], e))
            else:
                p0[i] = tmp_p0[i]
                if debug:
                    print(
                        'Warning in fit_gaussian_2d_to_imagesubset():\n\tNo fit guess no {} provided, using default {}\n'.format(
                            i, tmp_p0[i]))
    except TypeError as e:
        if debug:
            print(
                'Error in fit_gaussian_2d_to_imagesubset():\n\tCannot treat input parameter guesses\n\t\t{}\n\tusing defaults\n\t\t{}\n{}\n'.format(
                    p0, tmp_p0, e))

    try:
        warnings.simplefilter("error", OptimizeWarning)
        popt, pcov = curve_fit(gaussian_2d, (x, y), subset.ravel(),
                               p0=p0, bounds=([0, 0, 0, 0, 0, -45, 0], [np.inf, wx, wy, wx, wy, 45, np.inf]))
    except OptimizeWarning as e:
        if retryfitting:
            if debug:
                print(
                    'Warning in fig_gaussian_2d_to_imagesubset():\n{}\n\tTrying automated default guess {}\n'.format(e,
                                                                                                                     tmp_p0))
            p0 = tmp_p0
            if debug:
                warnings.simplefilter("default", OptimizeWarning)
            else:
                warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(gaussian_2d, (x, y), subset.ravel(),
                                   p0=p0, bounds=([0, 0, 0, 0, 0, -45, 0], [np.inf, wx, wy, wx, wy, 45, np.inf]))

        else:
            if debug:
                warnings.simplefilter("default", OptimizeWarning)
            else:
                warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(gaussian_2d, (x, y), subset.ravel(),
                                   p0=p0, bounds=([0, 0, 0, 0, 0, -45, 0], [np.inf, wx, wy, wx, wy, 45, np.inf]))

    if np.any(pcov == np.inf):
        if debug:
            print('\tFit using guess: {} failed (retryfitting {}.'.format(p0, retryfitting))
        perr = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    else:
        perr = np.sqrt(np.diag(pcov))
    return {'parameters': popt, 'parameter uncertainties': perr, 'x': x, 'y': y,
            'bounds': [bounds[0], bounds[1], bounds[2], bounds[3]], 'x min': bounds[0], 'x max': bounds[1],
            'y min': bounds[2], 'y max': bounds[3], 'extent': (bounds[2] - 0.5, bounds[3] + 0.5, bounds[1] + 0.5,
                                                               bounds[0] - 0.5)}


def add_contour(x, y, z, axis, number_of_contours=3):
    axis.contour(y, x, z, number_of_contours)
    return axis
