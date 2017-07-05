"""
Numerical approximation of the reaction-diffusion equation.
Dirichlet boundary conditions and the method BTCS (Backward-Time Central-Space) are used. 
This script is simplified to test the speed of parallel computing
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time

def main(t0,tmax,pas_t,x0,xmax,pas_x):
  #Initial time taking
  start_time = time.time()
  
  #Time definition
  delta_t = (tmax-t0)/(pas_t)
  times = np.linspace(t0,tmax,pas_t + 1)

  #Space definition
  delta_x = (xmax-x0)/(pas_x)
  space = np.linspace(x0,xmax,pas_x + 1)

  #Parameters definition
  k = 1
  r = k*delta_t/delta_x**2

  #Creation of the matrix that will contain the results
  final_matrix2 = np.zeros((pas_t + 1,pas_x + 1))

  #Initial conditions
  matrix2 = CI(space,xmax)
  final_matrix2[0,:] = matrix2[:]

  b2 = np.zeros(pas_x + 1)

  #Creation of the matrices that will be used to sparse the matrix A
  main  = np.zeros(pas_x + 1)
  lower = np.zeros(pas_x)
  upper = np.zeros(pas_x)
  main[:] = 1 + 2*r
  lower[:] = -r 
  upper[:] = -r  

  #Implementation of boundary conditions (Dirichlet)
  main[0] = 1
  main[pas_x] = 1
  lower[0]=0
  lower[pas_x-1]=0
  upper[0]=0
  upper[pas_x-1]=0

  #Sparsing of the matrix A
  A = scipy.sparse.diags(
      diagonals = [main, lower, upper],
      offsets = [0, -1, 1], shape = (pas_x + 1, pas_x + 1),
      format = 'csr')

  #Compute the solution 
  for n in range(0, pas_t):
    b2[1 : pas_x]=matrix2[1 : pas_x]+delta_t*Reac(matrix2[1 : pas_x])
    b2[0]=0
    b2[pas_x]=0
    matrix2[:]=scipy.sparse.linalg.spsolve(A, b2)

    final_matrix2[n][:]=matrix2[:]

  #Final time taking and elapsed time computation
  end_time = time.time()
  elapsed_time = end_time-start_time
  return round(elapsed_time,4)

def CI(x,xmax) :
  """
  Function that compute the initial condition

  Arguments:
    x (array(1,n)) : Space interval or space sub-interval
    xmax (float) : Right boundary of the space interval
    
  Returns:
    array(1,n) : 
  """
  return np.sin(np.pi*x/xmax)
  
def Reac(u) :
  """
  Function that compute the reaction term

  Arguments:
    u (array(1,n)) : Solutions at a given time step t
    
  Returns:
    array(1,n) : Values of the reaction
  """
  return np.exp(-u)

if __name__ == "__main__":
  test=10

  #Time definition
  t0 = 0
  tmax = 1
  pas_t = 500*test**2

  #Space definition
  x0 = 0
  xmax = 2
  pas_x = 21*test
  main(t0,tmax,pas_t,x0,xmax,pas_x)
