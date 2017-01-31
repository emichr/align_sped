# align_sped
Python code to align 000 reflection in a SPED stack

Requires hyperspy (hyperspy.api) and scikit image (skimage)

main functionality is in "align_sped.py" file

<h1>"align_sped.py" defines:</h1>
<ul>
  <li><i><b>gaussian_2d(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)</b></i>
    <ul>
      <li>Generate twodimensional gaussian distribution with arbitrary rotation.
        <ul>
          <li>This function will be possible to fit to data by calling <i>scipy.optimize.curvefit()</i></li>
        </ul>
      </li>
      <li>xdata_tuple is tuple with two numpy NxM arrays (made by e.g. <i>numpy.mgrid(np.arange(N), np.arange(M))</i>)</li>
      <li><i>amplitude</i>, <i>xo</i>, <i>yo</i>, <i>sigma_x</i>, <i>sigma_y</i>, <i>theta</i>, and <i>offset</i> will be converted to floats
        <ul>
          <li><i>xo</i> and <i>yo</i> are scalars that defines the centre of the gaussian</li>
          <li><i>sigma_x</i> and <i>sigma_y</i> are scalars that defines the width of the gaussian</li>
          <li><i>amplitude</i> is a scalar defining the amplitude of the gaussian</li>
          <li><i>offset</i> is a scalar defining the base value of the gaussian (baseline)</li>
          <li><i>theta</i> is a scalar definint the rotation of the gaussian (in degrees)</li>
          </ul>
      </li>
      <li>Returns a raveled array containing the intensity of a rotated gaussian distribution
        <ul>
          <li>Reshape into "correct" shape by calling e. g. <i>numpy.reshape(np.shape(xdata_tuple[0]))</i></li>
        </ul>
      </li>
    </ul>
  </li>    
</ul>

