from __future__ import division

import numpy as np
import hyperspy.api as hs
import time
import datetime as dt
from scipy.optimize import curve_fit
from skimage.transform import warp_coords
from scipy.ndimage import map_coordinates


def gaussian_2d_fast(xdata_tuple, amplitude, xo, yo,
                     sigma_x, sigma_y, theta, offset):
    """Compute a flattened gaussian at each coordinate

    Parameters
    ----------
    xdata_tuple : 2-tuple
        Tuple containing coordinate matrices at which points the guassian should be computed. `X` (first element - rows) and `Y` (second element - columns) generated through e.g. `X, Y = numpy.mgrid[0:N, 0:M]`.
    amplitude : float
        The amplitude of the gaussian.
    xo : float
        The origin of the guassian in the `X` direction (row).
    yo : float
        The origin of the gaussian in the `Y` direction (column).
    sigma_x : float
        The "width" of the unrotated gaussian in the `X` direction (row).
    sigma_y : float
        The "width" of the unrotated gaussian in the `Y` direction (column).
    theta : float
        The rotation of the gaussian in degrees, measured clockwise.
    offset : float
        The baseline of the gaussian (:math:`G(x\\rightarrow\\infty, y\\rightarrow\\infty`).

    Returns
    -------
    g : numpy.ndarray
        The values of the gaussian at the input coordinates as an unraveled array. See Examples for more information

    Notes
    -----
    The gaussian of amplitude :math:`A` and baseline :math:`C` is expressed as:

    .. math:: G(x, y, \\theta) = C + A*\\exp\\left(x'\\left(x, y, \\theta\\right)^{2} + y\\left(x, y, \\theta\\right)'^{2}\\right),

    where :math:`x'` and :math:`y'` are rotated according to conventional algebra:

    .. math:: \\begin{bmatrix} x'\\left(x, y, \\theta\\right)^2 \\\\ y'\\left(x, y, \\theta\\right)^2\\end{bmatrix} = \\begin{bmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta\\end{bmatrix}\\,\\begin{bmatrix}\\bar{x}\\left(x, y\\right)\\\\\\bar{y}\\left(x, y\\right)\\end{bmatrix}.

    :math:`\\bar{x}\\left(x, y\\right)` and :math:`\\bar{y}\\left(x, y\\right)` are functions of the cartesian coordinates :math:`x` and :math:`y` through the substitution of:

    .. math:: \\bar{x}\\left(x, y\\right)=\\frac{\\left(x-x_0\\right)^2}{2\\sigma_x}\\,\\,\\, \\bar{y}\\left(x, y\\right)=\\frac{\\left(y-y_0\\right)^2}{2\\sigma_y},

    where :math:`\\sigma_x` and :math:`\\sigma_y` are the widths of the resulting gaussian.

    In the code, this matrix multiplication and the substitutions has been carried out "by hand". There might be some numerical issues considering the implementation.


    Examples
    --------

    Compute a rotated gaussian at the specified points:

    >>> n, m = 256, 512
    >>> X, Y = numpy.mgrid[:n, :m]
    >>> A, C, origin, widths, rotation = 1337, 42, (144, 256), (10, 23), 14
    >>> G = gaussian_2d_fast((X, Y), 1337, 144, 256, 10, 23, 14, 42)
    >>> G = G.reshape(np.shape(X))

    Make a supersposition of two gaussians and fit a 2D gaussian to this sample image:

    >>> n, m = 256, 512
    >>> X, Y = numpy.mgrid[:n, :m]
    >>> A, C, origin, widths, rotation = 1337, 42, (144, 256), (10, 23), 14
    >>> G = gaussian_2d_fast((X, Y), 1337, 144, 256, 10, 23, 14, 42)
    >>> img = G.reshape(np.shape(X))
    >>> A, C, origin, widths, rotation = 512, 71, (150, 300), (10, 23), 35
    >>> img += gaussian_2d_fast((X, Y), 512, 150, 300, 10, 23, 35, 71)
    >>> img = img.reshape(np.shape(X))
    >>> popt, pcov = scipy.optimize.curve_fit(gaussian_2d_fast, (X, Y), G)
    >>> fitted_gaussian = gaussian_2d_fast((X, Y), *popt)
    >>> fitted_gaussian = fitted_gaussian.reshape(np.shape(X))
    >>> fig, ax = matplotlib.pyplot.subplots(1, 2)
    >>> ax[0].imshow(img, interpolation='nearest')
    >>> ax[1].imshow(fitted_gaussian, interpolation='nearest')
    >>> ax[0].set_title('Original image')
    >>> ax[1].set_title('Fitted gaussian')
    >>> matplotlib.pyplot.show()
    """
    (x_pos, y_pos) = xdata_tuple
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(
        theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(
        2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(
        theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(- (
        a * ((x_pos - xo) ** 2) + 2 * b * (x_pos - xo) * (
            y_pos - yo)
        + c * ((y_pos - yo) ** 2)))
    return g.ravel()


def fit_gaussian_2d_to_imagesubset_fast(image, bounds=10):
    """Fit a guassian to a subset of an image

    Parameters
    ----------
    image : numpy.ndarray
        The image to be fitted.
    bounds : int or float, optional
        The number of pixels to the right, left, top and bottom of the image center that should be considered when fitting the gaussian (defaults to 10).

    Returns
    -------
    popt : numpy.ndarray
        Optimal values for the parameters so that the sum of the squared error
        of ``f(xdata, *popt) - ydata`` is minimized. See `scipy.optimize.curvefit()`

    See Also
    --------
    scipy.optimize.curvefit

    Examples
    --------

    Provide an image of shape `(n, m)` (a superposition of two gaussians) and fit a gaussian to a square of widths 15 pixels around its center. Show the result

    >>> n, m = 256, 512
    >>> X, Y = numpy.mgrid[:n, :m]
    >>> G = gaussian_2d_fast((X, Y), 1337, 144, 256, 10, 23, 14, 42)
    >>> img = G.reshape(np.shape(X))
    >>> A, C, origin, widths, rotation = 512, 71, (150, 300), (10, 23), 35
    >>> img += gaussian_2d_fast((X, Y), 512, 150, 300, 10, 23, 35, 71)
    >>> popt = fit_gaussian_2d_to_imagesubset_fast(G, bounds=15)
    >>> fitted_gaussian = gaussian_2d_fast((X, Y), *popt)
    >>> fitted_gaussian.reshape(np.shape(X))
    >>> fig, ax = matplotlib.pyplot.subplots(1, 2)
    >>> ax[0].imshow(G, interpolation='nearest')
    >>> ax[1].imshow(fitted_gaussian, interpolation='nearest')
    >>> ax[0].set_title('Original image')
    >>> ax[1].set_title('Fitted gaussian')
    >>> matplotlib.pyplot.show()

    """
    (wx, wy) = np.shape(image)

    bounds = (int(wx / 2 - bounds), int(wx / 2 + bounds),
              int(wy / 2 - bounds),
              int(wy / 2 + bounds))

    subset = image[bounds[0]:bounds[1] + 1,
             bounds[2]:bounds[3] + 1].copy()
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(subset)
    # plt.show()
    x, y = np.mgrid[bounds[0]:bounds[1] + 1,
           bounds[2]:bounds[3] + 1]

    p0 = [subset.max(),
          int((bounds[1] - bounds[0]) / 2) + bounds[0],
          int((bounds[3] - bounds[2]) / 2) + bounds[2], 1,
          1, 0, 0]

    popt, pcov = curve_fit(gaussian_2d_fast, (x, y),
                           subset.ravel(), p0=p0,
                           bounds=(
                               [p0[0] / 2, 0, 0, 0, 0, -45,
                                0],
                               [p0[0] * 2, wx, wy, wx / 2,
                                wy / 2, 45, np.inf]))

    return popt


def add_contour(x, y, z, axis, number_of_contours=3):
    axis.contour(y, x, z, number_of_contours)
    return axis


def align_stack_fast(stack, limit=None, bounds=10,
                     save=False, savedir=None, name=None):
    """Find 000 position of each frame in stack and align the stack to these positions

    Parameters
    ----------
    stack : hyperspy.api.signals.Signal2D
        The stack containing the diffraction patterns to be aligned.
    limit : int, float or str, optional
        The total number of frames to consider before ending, useful for debugging (default is None which implies treating the whole stack).
    bounds : int, float or str, optional
        Defines the region where a 2D gaussian is fitted to the dataset (the default is 10). The region is defined as `frame[center[0]-bounds:center[0]+bounds+1, center[1]-bounds:center[1]+bounds+1`. The region should be chosen large enough to (preferably) only contain the 000 reflection.
    save : {False, True}, optional
        Whether or not to save the aligned stack and the gaussian fits as hdf5 files.
    savedir : str, optional
        The directory to store the results, if `save=True` (defaults to `''` which implies storing results in the cwd).
    name : str, optional
        The name to give the aligned stack (defaults to `aligned_stack`).

    Returns
    -------
    Popt : numpy.ndarray
        The fit results containing all the data to produce the fitted gaussian. The array has shape (N*M, 7) where N and M are the scan dimensions of the stack (total number of frames), and 7 is the number of parameters to describe the gaussian.
    g : hyperspy.api.signals.Signal2D
        A stack of the intensities of the fitted gaussians.
    aligned_stack : hyperspy.api.signals.Signal2D
        A stack containing the aligned diffraction patterns of the input stack.

    See Also
    --------
    gaussian_2d_fast : Compute 2D gaussian
    fit_gaussian_2d_to_imagesubset_fast : Fit a 2D gaussian to a subset of image

    Examples
    --------

    Load a blockfile containing Scanning Precession Electron Diffraction data and align them.

    >>> s = hs.load('data.blo')
    >>> Popts, g, s_aligned = sa.align_stack_fast(s, limit=None, save=True)

    """

    assert isinstance(stack,
                      hs.signals.Signal2D), '`stack` is type {}, must be a `hyperspy.api.signals.Signal2D` object'.format(
        type(stack))
    assert len(np.shape(
        stack)) == 4, '`stack` has shape {}, must have a 4D shape'.format(
        np.shape(stack))
    try:
        n_cols = stack.axes_manager['x'].get('size')['size']
        n_rows = stack.axes_manager['y'].get('size')['size']
        n_tot = n_cols * n_rows
        if limit is None:
            limit = n_tot
        else:
            limit = int(limit)

        ndx = stack.axes_manager['dx'].get('size')['size']
        ndy = stack.axes_manager['dy'].get('size')['size']

        bounds = int(bounds)
        assert bounds < ndx / 2 and bounds < ndy / 2, 'The bounds ({}) cannot be larger than half the diffraction pattern width (whole widths: {}, {}).'.format(
            bounds, ndx, ndy)

        save = bool(save)

        if save:
            if savedir is None:
                savedir = ''
            else:
                savedir = str(savedir)

            if name is None:
                name = 'aligned_stack'
            else:
                name = str(name)
    except ValueError as e:
        print(e)
        return None
    except TypeError as e:
        print(e)
        return None

    def shift(xy):
        return xy - np.array([72 - fit_popt[2], 72 - fit_popt[1]])[None, :]

    dx, dy = np.mgrid[0:ndx, 0:ndy]

    Popt = np.zeros((n_tot, 7))
    G = np.zeros((n_rows, n_cols, ndx, ndy), dtype=np.uint8)
    shifted_stack = np.zeros((n_rows, n_cols, ndx, ndy), dtype=np.uint8)

    rowno = 0
    colno = 0

    times = [time.time()]
    for frameno, frame in enumerate(stack):
        if colno == n_cols:
            colno = 0
            rowno += 1
        if not frameno % (int(limit / 100)):
            times.append(time.time() - times[0])
            print(
                '{}% done (now on Frame {} of {}: row {}, col {})'.format(
                    int(frameno / limit * 100), frameno,
                    limit, rowno, colno))

        try:
            fit_popt = fit_gaussian_2d_to_imagesubset_fast(frame.data,
                                                           bounds)

        except RuntimeError as e:
            g = hs.signals.Signal2D(G)
            g.metadata = stack.metadata.deepcopy()
            g.axes_manager = stack.axes_manager.copy()
            new_metadata = {'Postprocessing': {
                'date': dt.datetime.now().strftime('%c'),
                'version': 'latest',
                'type': 'Gaussian fit', 'limit': limit, 'save': save,
                'savedirectory': savedir, 'name': name,
                'time elapsed': times[-1]}}
            g.metadata.add_dictionary(new_metadata)
            # g.axes_manager = stack.axes_manager.copy()

            aligned_stack = hs.signals.Signal2D(shifted_stack)
            aligned_stack.metadata = stack.metadata.deepcopy()
            aligned_stack.axes_manager = stack.axes_manager.copy()
            new_metadata = {'Postprocessing': {
                'date': dt.datetime.now().strftime('%c'),
                'version': 'latest',
                'type': 'aligned stack', 'limit': limit, 'save': save,
                'savedirectory': savedir, 'name': name,
                'time elapsed': times[-1]}}
            aligned_stack.metadata.add_dictionary(new_metadata)

            if save:
                np.save(savedir + 'Popts_1', Popt)
                g.save(savedir + 'G.hdf5')
                aligned_stack.save(savedir + name + '.hdf5')
            print(e)
            return Popt, g, aligned_stack

        Popt[frameno, :] = fit_popt
        G[rowno, colno, :, :] = gaussian_2d_fast((dx, dy),
                                                 *fit_popt).reshape(
            np.shape(dx)).astype(np.uint8)
        coords = warp_coords(shift, frame.data.shape)
        shifted_stack[rowno, colno, :, :] = map_coordinates(frame.data,
                                                            coords).astype(
            np.uint8)

        colno += 1

        if frameno == limit or frameno == n_tot - 1:
            g = hs.signals.Signal2D(G)
            g.metadata = stack.metadata.deepcopy()
            #print('Stack metadata:\n', stack.metadata)
            #print('G metadata (deepcopied):\n', g.metadata)
            g.axes_manager = stack.axes_manager.copy()
            new_metadata = {'Postprocessing': {
                'date': dt.datetime.now().strftime('%c'),
                'version': 'latest',
                'type': 'Gaussian fit',
                'limit': limit,
                'save': save,
                'savedirectory': savedir,
                'name': name,
                'time elapsed': times[-1]}}
            g.metadata.add_dictionary(new_metadata)
            #print('G metadata (added dict):\n',g.metadata)

            aligned_stack = hs.signals.Signal2D(shifted_stack)
            aligned_stack.metadata = stack.metadata.deepcopy()
            aligned_stack.axes_manager = stack.axes_manager.copy()
            new_metadata = {'Postprocessing': {
                'date': dt.datetime.now().strftime('%c'),
                'version': 'latest',
                'type': 'aligned stack', 'limit': limit, 'save': save,
                'savedirectory': savedir, 'name': name,
                'time elapsed': times[-1]}}
            aligned_stack.metadata.add_dictionary(new_metadata)

            if save:
                np.save(savedir + 'Popts_1', Popt)
                g.save(savedir + 'G.hdf5')
                aligned_stack.save(savedir + name + '.hdf5')
            return Popt, g, aligned_stack


def shift_stack(stack, popt, n_cols, n_rows, ndx, ndy, limit=None,
                save=False, savedir=None, name=None):
    """Shift a stack

    Shifts the input stack, using the second and third elements in popt, and makes a new stack of the shifted images.

    Parameters
    ----------
    stack : hyperspy.api.signals.Signal2D
        A stack of e.g. Precession Electron Diffraction patterns to be shifted. Assumed to have shape (Nx, Ny, nx, ny)
    popt : numpy.ndarray
        A numpy array of shape (Nx*Ny, 7), where the second and third elements are used as shifts in row and column directions, respectively
    n_cols : int, float or str
        Number of columns in the stack (equal to Ny)
    n_rows : int, float or str
        Number of rows in the stack (equal to Nx)
    ndx : int, float or str
        Number of vertical (?) pixels in the diffraction pattern
    ndy : int, float or str
        Number of horizontal (?) pixels in the diffraction pattern
    limit : int, float, str or None, optional
        Number of frames to shift before breaking, useful for debugging (the default is `None` indicating that all frames will be shifted)
    save : {False, True}, optional
        Whether or not the shifted stack should be saved
    savedir : int, float or str, optional
        Used to specify a directory to store the shifted stack if `save=True` (default is `''` indicating that the signal will be stored in the cwd)
    name : int, float or str, optional
        Used to specify the name of the signal if `save=True` (default is `'shifted_stack'`

    Returns
    -------
    shifted_signal : hyperspy.api.signals.Signal2D
        The shifted stack as a hyperspy signal
    times : list of floats
        The time spent since the start of the for loop, sampled at every 1% of total frames.

    Raises
    ------

    Notes
    -----

    Examples
    --------

    Load a premade `popt` file and a signal to shift, then shift the signal

    >>> s = hs.load('C:\\Users\\emilc\\Desktop\\SPED_align_laptop\\2017_01_11_6060-20-1-C-4_001.blo')
    >>> n_cols = s.axes_manager['x'].get('size')['size']
    >>> n_rows = s.axes_manager['y'].get('size')['size']
    >>> ndx = s.axes_manager['dx'].get('size')['size']
    >>> ndy = s.axes_manager['dy'].get('size')['size']
    >>> Popts = np.load(savedir+'Popts_1.npy')
    >>> shifted_signal = sa.shift_stack(s, Popts, n_cols=n_cols, n_rows=n_rows, ndx=ndx, ndy=ndy, save=True, savedir='C:\\Users\\emilc\\Desktop\\SPED_align_laptop\\2017_01_11_6060-20-1-C-4_001_shifted')

    """
    assert isinstance(stack,
                      hs.signals.Signal2D), '`stack` is type {}, must be a `hyperspy.api.signals.Signal2D` object'.format(
        type(stack))
    assert isinstance(popt,
                      np.ndarray), '`popt` is type {}, must be a `numpy.ndarray`'.format(
        type(popt))
    assert len(np.shape(
        stack)) == 4, '`stack` has shape {}, must have a 4D shape'.format(
        np.shape(stack))
    assert len(np.shape(
        popt)) == 2, '`popt` has shape {}, must have a 2D shape (flattened array)'.format(
        np.shape(popt))
    assert np.shape(popt)[0] == np.shape(stack)[0] * np.shape(stack)[1] and \
           np.shape(popt)[
               1] == 7, 'First dimension of `popt` ({}) must equal the product of the first ({}) and second ({}) dimension of `stack`.'.format(
        np.shape(popt)[0], np.shape(stack)[0], np.shape(stack)[1])
    try:
        n_cols, n_rows, ndx, ndy = int(n_cols), int(n_rows), int(ndx), int(
            ndy)
        if limit is None:
            limit = n_rows * n_cols
        else:
            limit = int(limit)

        save = bool(save)

        if save:
            if savedir is None:
                savedir = ''
            else:
                savedir = str(savedir)

            if name is None:
                name = 'shifted_stack'
            else:
                name = str(name)
    except ValueError as e:
        print(e)
        return None
    except TypeError as e:
        print(e)
        return None

    # Define function for shifting image (called by skimage.transform.warp_coords)
    def shift(xy):
        """Shift image coordinates somehow

        Parameters
        ----------
        xy : numpy.ndarray
            Coordinate array

        Returns
        -------
        numpy.ndarray
            I think this function returns a numpy.ndarray... But I am not sure yet: On Todo!

        """
        return xy - np.array(
            [72 - popt[frameno, 2], 72 - popt[frameno, 1]])[None, :]

    # Total number of frames in the stack
    n_tot = n_cols * n_rows

    # Allocate memory and define the new (shifted) stack
    shifted_stack = np.zeros((n_rows, n_cols, ndx, ndy), dtype=np.uint8)

    # Define row and column counters
    rowno = 0
    colno = 0
    # Make a list over rumtime times.
    times = [time.time()]

    # loop over all the frames in the stack, and for each frame, use a corresponding image shift found by e.g.
    # `align_stack_fast()` to shift the frame
    for frameno, frame in enumerate(stack):
        if colno == n_cols:
            colno = 0
            rowno += 1
        # Print some output for every 1/100 th frame
        if not frameno % (int(limit / 100)):
            curr_t_diff = time.time() - times[0]
            times.append(curr_t_diff)

            print(
                '{}% done (now on Frame {} of {}: row {}, col {})\n\tTime passed since start: {} seconds.'.format(
                    int(frameno / limit * 100), frameno,
                    limit, rowno, colno, curr_t_diff))

        coords = warp_coords(shift, frame.data.shape)
        shifted_stack[rowno, colno, :, :] = map_coordinates(frame.data,
                                                            coords).astype(
            np.uint8)

        colno += 1

        # Check if the limit is reached or if the loop is finished.
        if frameno == limit or frameno == n_tot - 1:
            shifted_signal = hs.signals.Signal2D(shifted_stack)
            shifted_signal.metadata = stack.metadata.deepcopy()
            new_metadata = {'Postprocessing': {
                'date': dt.datetime.now().strftime('%c%'),
                'version': 'latest',
                'type': 'Shifted stack', 'limit': limit, 'save': save,
                'savedirectory': savedir, 'name': name,
                'time elapsed': times[-1]}}
            shifted_signal.metadata.add_dictionary(new_metadata)
            shifted_signal.axes_manager = stack.axes_manager.copy()

            # If the user wants to save the shifted signal, save it as `.hdf5` file
            if save:
                shifted_signal.save(savedir + name + '.hdf5')
            return shifted_signal, times

    # Just in case the if statement above bugs, return the signal anyways
    shifted_signal = hs.signals.Signal2D(shifted_stack)
    shifted_signal.metadata = shifted_signal.metadata = stack.metadata.deepcopy()
    new_metadata = {
        'Postprocessing': {'date': dt.datetime.now().strftime('%c%'),
                           'version': 'latest',
                           'type': 'Shifted stack', 'limit': limit,
                           'save': save,
                           'savedirectory': savedir, 'name': name,
                           'time elapsed': times[-1]}}
    shifted_signal.metadata.add_dictionary(new_metadata)
    shifted_signal.axes_manager = stack.axes_manager.copy()
    return shifted_signal, times
