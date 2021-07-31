import numpy as np
from scipy.optimize import root_scalar

def project_on_ellipse(A, c, y):
  """ Projects the vector y into the set S = {x: x^T A x <= c}

  inputs:
      A: Numpy array represents the covariance matrix (It should be symmetric PSD)
      c: level set: $ x^T A x \leq c $
      y: vector to be projected
  
  outputs:
      projection: numpy array of the projected version of y
      a bool represents of indeed y is in the desired set.
   """

  def check(A, y):
    return np.matmul(y.T, np.matmul(A, y)) <= 1

  def solve(A, y):
    inv_part = lambda t: np.linalg.inv(np.identity(A.shape[0]) + t*A)
    intermediate_part = lambda t: np.matmul(inv_part(t), np.matmul(A, inv_part(t)))
    f = lambda t: np.matmul(y.T, np.matmul(intermediate_part(t), y)) - 1

    return root_scalar(f, method='bisect', bracket=[0.01, 1000]).root

  A = A/c
  #Check if y belongs to our region
  if check(A, y):
    print('your vector is already in the desired set')
    return y

  print('projecting your vector ...')
  t = solve(A, y)
  projection = np.linalg.solve(np.identity(A.shape[0]) + t*A, y)
  print('Done!')
  return projection, check(A, projection)
