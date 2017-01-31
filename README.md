# align_sped
Python code to align 000 reflection in a SPED stack

Requires hyperspy (hyperspy.api) and scikit image (skimage)

main functionality is in "align_sped.py" file

<h1>"align_sped.py" defines:</h1>
<ul>
  <li>gaussian_2d(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    <ul>
      <li>Generate twodimensional gaussian distribution with arbitrary rotation.
        <ul>
          <li>This function will be possible to fit to data by calling "scipy.optimize.curvefit()"</li>
        </ul>
      </li>
      <li>xdata_tuple is tuple with two numpy NxM arrays (made by e.g. "numpy.mgrid(np.arange(N), np.arange(M))")</li>
      <li>"amplitude", "xo", "yo", "sigma_x", "sigma_y", "theta", and "offset" will be converted to floats
        <ul>
          <li>"xo" and "yo" are scalars that defines the centre of the gaussian</li>
          <li>"sigma_x" and "sigma_y" are scalars that defines the width of the gaussian</li>
          <li>"amplitude" is a scalar defining the amplitude of the gaussian</li>
          <li>"offset" is a scalar defining the base value of the gaussian (baseline)</li>
          <li>"theta" is a scalar definint the rotation of the gaussian (in degrees)</li>
          </ul>
      </li>
      <li>Returns a raveled array containing the intensity of a rotated gaussian distribution
        <ul>
          <li>Reshape into "correct" shape by calling e. g. "numpy.reshape(np.shape(xdata_tuple[0]))"</li>
        </ul>
      </li>
    </ul>
  </li>    
</ul>

