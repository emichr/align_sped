from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import warnings

import hyperspy.api as hs
from scipy.optimize import curve_fit, OptimizeWarning


def check_gaussian_2d_inputs(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Check that the inputs can be converted to proper types

    This function checks that the input parameters to `gaussian_2d()` can be converted to correct types,
    without exiting the code.

    Parameters
    ----------
    xdata_tuple : 2-tuple
        Tuple containing row and column arrays as its element. The elements should be NxM numpy arrays where the first
        array contain the row indices, i.e. a_ij=i, and the second array contain the column indices, i.e. a_ij=j.
    amplitude : int, string or float
        The amplitude of the gaussian.
    xo : int, float or string
        The row position of the gaussian.
    yo : int, float or string
        The column position of the gaussian.
    sigma_x : int, float or string
        The width of the gaussian, in the row direction.
    sigma_y : int, float or string
        The width of the gaussian, in the column direction.
    theta : int, float or string
        The rotation of the gaussian, measured clockwise in degrees.
    offset : int, float or string
        The offset (or baseline) of the gaussian.

    Returns
    -------
    dict
        Dictionary containing the converted arguments.
    vertical positions : dict key (string)
        Value contains the row array.
    horizontal positions : dict key (string)
        Value contains the column array.
    amplitude : dict key (string)
        Value contains the amplitude as float.
    vertical origin : dict key (string)
        Value contains the horizontal origin of the gaussian as a float.
    horizontal origin : dict key (string)
        Value contains the vertical origin of the gaussian as a float.
    vertical sigma : dict key (string)
        Value contains the width of the gaussian in the vertical direction (row) as a float.
    horizontal sigma : dict key (string)
        Value contains the width of the gaussian in the horizontal direction (column) as a float.
    rotation : dict key (string)
        Value contains the angle that the gaussian is rotated with as a float.
    offset : dict key (string)
        Value contains the offset (baseline) of the gaussian as a float.
    ok inputs : dict key (string)
        Value contains True of False. If all values were converted successfully, returns as True, otherwise returns
        False.

    Raises
    ------

    Notes
    -----

    Examples
    --------

    Create two valid row/column arrays of shape (N, M) and provide valid inputs.

    >>> X, Y = numpy.mgrid[:N, :M]
    >>> check = check_gaussian_2d_inputs((X, Y), 100, '20', 37, '0.1', 5, 20.0, -10.1)
    >>> check['vertical positions']
    X
    >>> check['horizontal positions']
    Y
    >>> check['amplitude']
    100.0
    >>> check['vertical origin']
    20.0
    >>> check['horizontal origin']
    37.0
    >>> check['vertical sigma']
    0.1
    >>> check['horizontal sigma']
    5.0
    >>> check['rotation']
    20.0
    >>> check['offset']
    -10.1
    >>> check['ok inputs']
    True

    Provide the function with invalid parameters:

    >>> X, Y = numpy.mgrid[:N, :M]
    >>> check = check_gaussian_2d_inputs(90, 'one hundred', '20a', 'invalid', '0,1', 'ajk2', 'invalid', 'weee')
    Value error in gaussian_2d():
    could not convert theta invalid to float: can't multiply sequence by non-int of type 'float'
    Value error in gaussian_2d(): could not extract x_pos and y_pos from x_data tuple 90: 'int' object is not iterable
    Value error in gaussian_2d(): xo 20a is not a valid float
    could not convert string to float: '20a'
    Value error in gaussian_2d(): yo invalid is not a valid float
    could not convert string to float: 'invalid'
    Value error in gaussian_2d(): xo 0,1 is not a valid float
    could not convert string to float: '0,1'
    Value error in gaussian_2d(): xo ajk2 is not a valid float
    could not convert string to float: 'ajk2'
    Value error in gaussian_2d(): offset weee is not a valid float
    could not convert string to float: 'weee'
    Value error in gaussian_2d(): amplitude one hundred is not a valid float
    could not convert string to float: 'one hundred'
    >>> check['vertical positions']
    None
    >>> check['horizontal positions']
    None
    >>> check['amplitude']
    None
    >>> check['vertical origin']
    None
    >>> check['horizontal origin']
    None
    >>> check['vertical sigma']
    None
    >>> check['horizontal sigma']
    None
    >>> check['rotation']
    None
    >>> check['offset']
    None
    >>> check['ok inputs']
    False

    """

    result = {'vertical positions': None, 'horizontal positions': None, 'amplitude': None, 'vertical origin': None,
              'horizontal origin': None, 'vertical sigma': None, 'horizontal sigma': None, 'rotation': None,
              'offset': None, 'ok inputs': None}
    try:
        theta = float(theta * np.pi / 180.0)
        result['rotation'] = theta
    except ValueError as e:
        print('Value error in {}:\n\tcould not convert theta {} to float: {}'.format(r'gaussian_2d()', theta, e))
        result['ok inputs'] = False
    except TypeError as e:
        print('Value error in {}:\n\tcould not convert theta {} to float: {}'.format(r'gaussian_2d()', theta, e))
        result['ok inputs'] = False
    try:
        (x_pos, y_pos) = xdata_tuple
        result['vertical positions'] = x_pos
        result['horizontal positions'] = y_pos
    except ValueError as e:
        print('Value error in {}: could not extract x_pos and y_pos from x_data tuple {}: {}'.format(r'gaussian_2d()',
                                                                                                     xdata_tuple, e))
        result['ok inputs'] = False
    except TypeError as e:
        print('Value error in {}: could not extract x_pos and y_pos from x_data tuple {}: {}'.format(r'gaussian_2d()',
                                                                                                     xdata_tuple, e))
        result['ok inputs'] = False
    try:
        xo = float(xo)
        result['vertical origin'] = xo
    except ValueError as e:
        print('Value error in {}: xo {} is not a valid float\n\t{}'.format(r'gaussian_2d()', xo, e))
        result['ok inputs'] = False

    try:
        yo = float(yo)
        result['horizontal origin'] = yo
    except ValueError as e:
        print('Value error in {}: yo {} is not a valid float\n\t{}'.format(r'gaussian_2d()', yo, e))
        result['ok inputs'] = False

    try:
        sigma_x = float(sigma_x)
        result['vertical sigma'] = sigma_x
    except ValueError as e:
        print('Value error in {}: xo {} is not a valid float\n\t{}'.format(r'gaussian_2d()', sigma_x, e))
        result['ok inputs'] = False

    try:
        sigma_y = float(sigma_y)
        result['horizontal sigma'] = sigma_y
    except ValueError as e:
        print('Value error in {}: xo {} is not a valid float\n\t{}'.format(r'gaussian_2d()', sigma_y, e))
        result['ok inputs'] = False

    try:
        offset = float(offset)
        result['offset'] = offset
    except ValueError as e:
        print('Value error in {}: offset {} is not a valid float\n\t{}'.format(r'gaussian_2d()', offset, e))
        result['ok inputs'] = False

    try:
        amplitude = float(amplitude)
        result['amplitude'] = amplitude
    except ValueError as e:
        print('Value error in {}: amplitude {} is not a valid float\n\t{}'.format(r'gaussian_2d()', amplitude, e))
        result['ok inputs'] = False

    if result['ok inputs'] is None:
        result['ok inputs'] = True

    return result


def gaussian_2d_fast(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x_pos, y_pos) = xdata_tuple
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(- (a * ((x_pos - xo) ** 2) + 2 * b * (x_pos - xo) * (y_pos - yo)
                                       + c * ((y_pos - yo) ** 2)))
    return g.ravel()


def gaussian_2d(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Compute two dimensional gaussian with arbitrary rotation.

    Calculates a twodimensional gaussian with two independent widths and arbitrarily rotated by the standard rotation
    transformation. The input parameters will be converted to floats, except for the position arrays.

    Parameters
    ----------
    xdata_tuple : 2-tuple
        Tuple containing row and column arrays as its element. The elements should be NxM numpy arrays where the first
        array contain the row indices, i.e. a_ij=i, and the second array contain the column indices, i.e. a_ij=j.
    amplitude : int, string or float
        The amplitude of the gaussian.
    xo : int, float or string
        The row position of the gaussian.
    yo : int, float or string
        The column position of the gaussian.
    sigma_x : int, float or string
        The width of the gaussian, in the row direction.
    sigma_y : int, float or string
        The width of the gaussian, in the column direction.
    theta : int, float or string
        The rotation of the gaussian, measured clockwise in degrees.
    offset : int, float or string
        The offset (or baseline) of the gaussian.

    Returns
    -------
    numpy 1d array
        The raveled array of length N*M containing the amplitude of the gaussian. Reshape into array format by calling
        `.reshape(numpy.shape(x_tuple[0]))` on the 1d array.

    Raises
    ------


    Notes
    -----


    Examples
    --------
    >>> X, Y = numpy.mgrid[:N, :M]
    >>> gaussian = gaussian_2d((X, Y), 100, N/2, M/2, N/100, M/100, 20, 30)
    >>> gaussian = gaussian.reshape(numpy.shape(X))
    >>> fig, ax = matplotlib.pyplot.figure()
    >>> ax.imshow(gaussian, interpolation='nearest')
    >>> matplotlib.pyplot.show()

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


def check_fit_gaussian_2d_to_imagesubset_inputs(image, **kwargs):
    try:
        debug = kwargs['debug']
        assert isinstance(debug, bool), 'debug must be bool'
    except KeyError as e:
        debug = False
    except AssertionError as e:
        print(e)
        debug = False
    except:
        debug = False

    (wx, wy) = np.shape(image)

    bounds = [0, wx, 0, wy]
    try:
        subset_bounds = kwargs['subset_bounds']
        try:
            assert isinstance(subset_bounds,
                              (tuple, list)), 'if subset_bounds is not tuple or list, it must be int or float'
            try:
                assert len(subset_bounds) == 4, 'subset_bounds must be of length 4 if it is tuple or list'
            except AssertionError as e:
                subset_bounds = (0, wx, 0, wy)
                if debug:
                    print(e)
        except AssertionError as e:
            if debug:
                print(e)
            try:
                assert isinstance(subset_bounds, (int, float)), 'subset_bounds are not tuple, list, int or float'
                subset_bounds = (int(wx / 2 - subset_bounds), int(wx / 2 + subset_bounds), int(wy / 2 - subset_bounds),
                                 int(wy / 2 + subset_bounds))
                try:
                    assert subset_bounds[1] > \
                           subset_bounds[0] >= 0, 'The vertical subset bounds cannot be larger than the original frame'
                    assert subset_bounds[3] > \
                           subset_bounds[
                               2] >= 0, 'The horizontal subset bounds cannot be larger than the original frame'
                except AssertionError as e:
                    subset_bounds = (0, wx, 0, wy)
                    if debug:
                        print(e)
            except AssertionError as e:
                subset_bounds = (0, wx, 0, wy)
                if debug:
                    print(e)
    except KeyError as e:
        if debug:
            print(e)
        subset_bounds = (None, None, None, None)
    except:
        subset_bounds = (None, None, None, None)

    try:
        p0 = kwargs['p0']
        assert isinstance(p0, (list, tuple)), 'Initial guess p0 must be list or tuple'
        assert len(p0) == 7, 'Length of initial guess p0 must be 7'
    except KeyError as e:
        p0 = [None] * 7
    except AssertionError as e:
        if debug:
            print(e)
        p0 = [None] * 7
    except:
        p0 = [None] * 7

    try:
        retryfitting = kwargs['retryfitting']
        assert isinstance(retryfitting, bool), 'retryfitting must be bool'
    except KeyError as e:
        retryfitting = True
        if debug:
            print(e)
    except AssertionError as e:
        retryfitting = True
        if debug:
            print(e)
    except:
        retryfitting = True

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

    result = {'debug': debug, 'bounds': bounds, 'p0': p0, 'retryfitting': retryfitting}
    return image, result


def fit_gaussian_2d_to_imagesubset_fast(image, bounds=10):
    (wx, wy) = np.shape(image)

    bounds = (int(wx / 2 - bounds), int(wx / 2 + bounds), int(wy / 2 - bounds),
              int(wy / 2 + bounds))

    subset = image[bounds[0]:bounds[1] + 1, bounds[2]:bounds[3] + 1].copy()
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(subset)
    # plt.show()
    x, y = np.mgrid[bounds[0]:bounds[1] + 1, bounds[2]:bounds[3] + 1]

    p0 = [subset.max(), int((bounds[1] - bounds[0]) / 2) + bounds[0], int((bounds[3] - bounds[2]) / 2) + bounds[2], 1,
          1, 0, 0]

    popt, pcov = curve_fit(gaussian_2d, (x, y), subset.ravel(), p0=p0,
                           bounds=([p0[0] / 2, 0, 0, 0, 0, -45, 0], [p0[0] * 2, wx, wy, wx / 2, wy / 2, 45, np.inf]))

    return popt, x, y, [bounds[0], bounds[1], bounds[2], bounds[3]], (
        bounds[2] - 0.5, bounds[3] + 0.5, bounds[1] + 0.5, bounds[0] - 0.5)


def fit_gaussian_2d_to_imagesubset(image, **kwargs):  # subset_bounds=(None, None, None, None),
    # p0=[None, None, None, None, None, None, None], retryfitting=True, debug=False):
    """Fit a twodimensional gaussian to part of image.

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

    Raises
    ------

    Notes
    -----

    Examples
    --------

    Make a sample image and fit a gaussian to it on a small part. Plot the images side by side at the end, with countours.

    >>> N, M = 256, 256 #Define width of image
    >>> X, Y = numpy.mgrid[:256, :256] #Represent image pixels by two NxM arrays
    >>> A, Vo, Ho, Vw, Hw, Rot, C = 254, 100, 180, 4, 8, 15, 0 #Define gaussian parameters amplitude, vertical origin, horizontal origin, vertical width, horizontal width, rotation and baseline, respectively.
    >>> G = gaussian_2d((X, Y), 254, 100, 180, 4, 8, 15, 0).reshape(numpy.shape(X)) #Compute twodimensional gaussian
    >>> fitresult = fit_gaussian_2d_to_imagesubset(G, subset_bounds=(50, 150, 100, 200)) #Fit a gaussian to a part of the image
    >>> g_1 = sa.gaussian_2d((X, Y), *fitresult['parameters']).reshape(np.shape(X))
    >>> g_2 = sa.gaussian_2d((fitresult['x'], fitresult['y']), *fitresult['parameters']).reshape(np.shape(fitresult['x']))
    >>> fig, axes = matplotlib.pyplot..subplots(1, 2) #Create figure and axes
    >>> axes[0].imshow(G, interpolation='nearest', cmap=plt.get_cmap('RdBu')) #show original image
    >>> axes[1].imshow(G[fitresult['x min']:fitresult['x max'] + 1, fitresult['y min']:fitresult['y max'] + 1], interpolation='nearest', cmap=plt.get_cmap('RdBu'), extent=fitresult['extent']) #Show the part of image where the gaussian was fitted
    >>> add_contour(X, Y, g_1, axes[0])
    >>> add_contour(fitresult['x'], fitresult['y'], g_2, axes[1])
    >>> matplotlib.pyplot.show() #Show the images

    """

    # bounds = kwargs['bounds']
    # debug = kwargs['debug']
    # p0 = kwargs['p0']
    # retryfitting = kwargs['retryfitting']

    try:
        debug = kwargs['debug']
        assert isinstance(debug, bool), 'debug must be bool'
    except KeyError as e:
        debug = False
    except AssertionError as e:
        print(e)
        debug = False
    except:
        debug = False

    (wx, wy) = np.shape(image)

    bounds = [0, wx, 0, wy]
    try:
        subset_bounds = kwargs['subset_bounds']
        try:
            assert isinstance(subset_bounds,
                              (tuple, list)), 'if subset_bounds is not tuple or list, it must be int or float'
            try:
                assert len(subset_bounds) == 4, 'subset_bounds must be of length 4 if it is tuple or list'
            except AssertionError as e:
                subset_bounds = (0, wx, 0, wy)
                if debug:
                    print(e)
        except AssertionError as e:
            if debug:
                print(e)
            try:
                assert isinstance(subset_bounds, (int, float)), 'subset_bounds are not tuple, list, int or float'
                subset_bounds = (int(wx / 2 - subset_bounds), int(wx / 2 + subset_bounds), int(wy / 2 - subset_bounds),
                                 int(wy / 2 + subset_bounds))
                try:
                    assert subset_bounds[1] > \
                           subset_bounds[0] >= 0, 'The vertical subset bounds cannot be larger than the original frame'
                    assert subset_bounds[3] > \
                           subset_bounds[
                               2] >= 0, 'The horizontal subset bounds cannot be larger than the original frame'
                except AssertionError as e:
                    subset_bounds = (0, wx, 0, wy)
                    if debug:
                        print(e)
            except AssertionError as e:
                subset_bounds = (0, wx, 0, wy)
                if debug:
                    print(e)
    except KeyError as e:
        if debug:
            print(e)
        subset_bounds = (None, None, None, None)
    except:
        subset_bounds = (None, None, None, None)

    try:
        p0 = kwargs['p0']
        assert isinstance(p0, (list, tuple)), 'Initial guess p0 must be list or tuple'
        assert len(p0) == 7, 'Length of initial guess p0 must be 7'
    except KeyError as e:
        p0 = [None] * 7
    except AssertionError as e:
        if debug:
            print(e)
        p0 = [None] * 7
    except:
        p0 = [None] * 7

    try:
        retryfitting = kwargs['retryfitting']
        assert isinstance(retryfitting, bool), 'retryfitting must be bool'
    except KeyError as e:
        retryfitting = True
        if debug:
            print(e)
    except AssertionError as e:
        retryfitting = True
        if debug:
            print(e)
    except:
        retryfitting = True

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

    # print(bounds[0], bounds[1]+1, bounds[2], bounds[3]+1)
    # print(np.shape(subset), np.shape(x), x.min(), x.max())

    try:
        assert x.min() == bounds[0], 'minimum of "x" ({}) is not equal to "bounds[0]" ({})'.format(x.min(), bounds[0])
    except AssertionError as e:
        if debug:
            print(e)
    try:
        assert x.max() == bounds[1], 'maximum of "x" ({}) is not equal to "bounds[1]" ({})'.format(x.max(), bounds[1])
    except AssertionError as e:
        if debug:
            print(e)
    try:
        assert y.min() == bounds[2], 'minimum of "y" ({}) is not equal to "bounds[2]" ({})'.format(y.min(), bounds[2])
    except AssertionError as e:
        if debug:
            print(e)
    try:
        assert y.max() == bounds[3], 'maximum of "y" ({}) is not equal to "bounds[3]" ({})'.format(x.max(), bounds[3])
    except AssertionError as e:
        if debug:
            print(e)

    # print(bounds)

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
        if debug:
            print('Fitting curve. x, y has shapes {0}, {1}, while subset has shape {2}'.format(np.shape(x), np.shape(y),
                                                                                               np.shape(subset)))
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
    except RuntimeError as e:
        if debug:
            print(e)
        popt, pcov = [0] * 7, np.zeros((7, 7))
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


def align_stack(stack, **kwargs):
    """
    Align a stack of diffraction patterns

    The stack should be in a `hyperspy` format

    Parameters
    ----------
    stack : hyperspy._signals.signal2d.Signal2D
        Stack of images to align
    kwargs : keyword arguments
        To be passed on to fit_gaussian_2d_to_imagesubset()

    Returns
    -------

    """
    try:
        limit = kwargs['limit']
        limit = int(limit)
    except KeyError as e:
        limit = None
    except ValueError as e:
        limit = None

    try:
        debug = kwargs['debug']
        debug = bool(debug)
    except KeyError as e:
        debug = False
    except ValueError as e:
        debug = False
    except:
        debug = False

    try:
        print_frameno = kwargs['print_frameno']
        print_frameno = bool(print_frameno)
    except KeyError as e:
        print_frameno = False
    except ValueError as e:
        print_frameno = False
    except:
        print_frameno = False

    n_cols = stack.axes_manager['x'].get('size')['size']
    n_rows = stack.axes_manager['y'].get('size')['size']
    n_tot = n_cols * n_rows

    if limit is None:
        limit = n_tot

    # print(n_cols, n_rows)

    Popt = np.zeros((n_rows, n_cols, 7))
    Perr = np.zeros((n_rows, n_cols, 7))
    rowno = 0
    colno = 0

    for frameno, frame in enumerate(stack):
        if print_frameno:
            print('Handling frame {}: row {}, column {}'.format(frameno, rowno, colno))
            # print(np.shape(frame.data))

        if colno == n_cols:
            colno = 0
            rowno += 1

        fit_result = fit_gaussian_2d_to_imagesubset_fast(frame.data, **kwargs)
        Popt[rowno, colno, :] = fit_result['parameters']
        Perr[rowno, colno, :] = fit_result['parameter uncertainties']
        if debug:
            print(fit_result['parameters'])
        colno += 1

        if frameno == limit:
            # print('Bounds (final frame): {}, {}, {}, {}'.format(fit_result['x min'], fit_result['x max'],
            #                                                     fit_result['y min'], fit_result['y max']))
            #
            # X, Y = np.mgrid[0:np.shape(frame.data)[0], 0:np.shape(frame.data)[1]]
            # #print(np.shape(X), np.shape(frame.data))
            # g_1 = gaussian_2d((X, Y), *fit_result['parameters']).reshape(np.shape(frame.data))
            # g_2 = gaussian_2d((fit_result['x'], fit_result['y']), *fit_result['parameters']).reshape(
            #     np.shape(fit_result['x']))
            # f, ax = plt.subplots(1, 2)
            # ax[0].imshow(frame.data, interpolation='nearest')
            # add_contour(X, Y, g_1, ax[0])
            # ax[1].imshow(
            #     frame.data[fit_result['x min']:fit_result['x max'], fit_result['y min']:fit_result['y max']],
            #     interpolation='nearest', extent=fit_result['extent'])
            # add_contour(fit_result['x'], fit_result['y'], g_2, ax[1])
            # plt.show()
            return {'Popt': Popt[:rowno + 1, :colno + 1, :], 'Perr': Perr[:rowno + 1, :colno + 1, :],
                    'x': fit_result['x'], 'y': fit_result['y'], 'extent': fit_result['extent']}

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(frame.data, interpolation='nearest')
    # print('Bounds (final frame): {}, {}, {}, {}, {}'.format(fit_result['x min'], fit_result['x max']+1, fit_result['y min'], fit_result['y max']+1))
    ax[1].imshow(frame.data[fit_result['x min']:fit_result['x max'] + 1, fit_result['y min']:fit_result['y max'] + 1],
                 interpolation='nearest')
    plt.show()

    return {'Popt': Popt, 'Perr': Perr, 'x': fit_result['x'], 'y': fit_result['y'], 'extent': fit_result['extent']}


def align_stack_fast(stack, limit=400, bounds=10, save=False):
    """
    Align a stack of diffraction patterns

    The stack should be in a `hyperspy` format

    Parameters
    ----------
    stack : hyperspy._signals.signal2d.Signal2D
        Stack of images to align
    kwargs : keyword arguments
        To be passed on to fit_gaussian_2d_to_imagesubset()

    Returns
    -------

    """

    n_cols = stack.axes_manager['x'].get('size')['size']
    n_rows = stack.axes_manager['y'].get('size')['size']
    n_tot = n_cols * n_rows
    if limit is None:
        limit = n_tot

    # print(n_cols, n_rows)

    Popt = np.zeros((n_tot, 7))
    Popt_2 = np.zeros((n_rows, n_cols, 7))
    Perr = np.zeros((n_tot, 7))
    rowno = 0
    colno = 0

    for frameno, frame in enumerate(stack):
        if colno == n_cols:
            colno = 0
            rowno += 1
        if not frameno % (int(limit / 100)):
            print('{}% done (now on Frame {} of {}: row {}, col {})'.format(int(frameno / limit * 100), frameno, limit, rowno, colno))

        try:
            fit_popt, fit_x, fit_y, fit_bounds, fit_extent = fit_gaussian_2d_to_imagesubset_fast(frame.data, bounds)
        except RuntimeError as e:
            if save:
                np.save('Popts_1', Popt)
                np.save('Popts_2', Popt_2)
            print(e)
        Popt[frameno, :] = fit_popt
        Popt_2[rowno, colno, :] = fit_popt
        colno += 1

        if frameno == limit or frameno == n_tot - 1:
            if save:
                np.save('Popts_1', Popt)
                np.save('Popts_2', Popt_2)
            return Popt, Popt_2[:, :, :]
