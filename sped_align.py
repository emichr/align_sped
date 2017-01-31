from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import hyperspy.api as hs


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



