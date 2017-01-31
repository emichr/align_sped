from sped_align import gaussian_2d

import numpy as np

def test_gaussian_2d_size():
  N = 10
  M = 15
  X, Y = np.mgrid(np.arange(N), np.arange(M))
  G = gaussian_2d((X, Y), 1.0, 5.0, 5.0, 1.0, 1.0, 0.0, 0.0).reshape(np.shape(X))
  assert len(G[:,0]) == N
  
  assert len(G[0,:]) == M
  
