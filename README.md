# align_sped
<h1>Python code to align 000 reflection in a <i>Scanning Precession Electron Diffraction</i> (SPED) stack</h1>

<h2>Main functionality is in <i>align_sped.py</i> file</h2>

<h3>Contains:</h3>
<ul>
  <li><i>align_sped.py</i>
    <ul>
      <li><i>gaussian_2d()</i></li>
    </ul>
  </li>
  <li><i>test_gaussian_2d.py</i>
    <ul>
      <li><i>test_gaussian_2d_size()</i></li>
      <li><i>test_gaussian_2d_centre()</i></li>
    </ul>
  </li>
</ul>

<h3>Depends on:</h3>
<ul>
<li><i>hyperspy</i> (hyperspy.api)</li>
<li><i>scikit image</i> (skimage)</li>
</ul>

<h4><i>align_sped.py</i> defines:</h4>
<ul>
  <li><i><b>gaussian_2d(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)</b></i>
    <ul>
      <li>Generate twodimensional gaussian distribution with arbitrary rotation.
        <ul>
          <li>This function will be possible to fit to data by calling <i>scipy.optimize.curvefit()</i></li>
        </ul>
      </li>
      <li><i>xdata_tuple</i> is a tuple consisting of two numpy NxM arrays (made by e.g. <i>numpy.mgrid[:N, :M]</i>)</li>
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
          <li>Reshape into "correct" shape by calling e. g. <i>gaussian_2d(...).reshape(np.shape(xdata_tuple[0]))</i></li>
        </ul>
      </li>
    </ul>
  </li>    
</ul>

<h4><i>test_gaussian_2d.py</i> defines:</h4>
<ul>
  <li><i><b>test_gaussian_2d_size()</b></i>
    <ul>
      <li>Tests that the size of the returned gaussian is correct</li>
    </ul>
  </li> 
  <li><i><b>test_gaussian_2d_center()</b></i>
    <ul>
      <li>Tests that the centre of the gaussian coincides with the maximum value of the returned array
        <ul>
          <li>Requires large <i>sigma_x</i> and <i>sigma_y</i> values</li>
        </ul>
      </li>
    </ul>
  </li>
  <li><i><b>test_gaussian_2d_fittable()</b></i>
    <ul>
      <li>Tests that the 2D gaussian is fittable using <i>scipy.optimize.curve_fit()</i> method</li>
    </ul>
  </li>
</ul>
