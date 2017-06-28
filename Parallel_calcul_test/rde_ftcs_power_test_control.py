"""
Numerical approximation of the reaction-diffusion equation.
Dirichlet boundary conditions and the method FTCS (Forward-Time Central-Space) are used.
The output files are csv files and it is possible to visualize the solutions with a mp4 movie. 
"""

# -*- coding :  utf-8 -*-

from __future__ import division
import numpy as np
import sys
import time

def main(t0,tmax,pas_t,x0,xmax,pas_x):
  #Initial time taking
  start_time = time.time()

  #Definition of the time interval
  delta_t = (tmax-t0)/(pas_t)
  times = np.linspace(t0,tmax,pas_t + 1)

  #Definition of the space interval
  delta_x = (xmax-x0)/(pas_x)
  space = np.linspace(x0,xmax,pas_x + 1)

  #Parameters definition
  k = 1
  r = k*delta_t/delta_x**2
  if r>=  0.5 : 
    print("k*delta_t/delta_x**2 is equal to ",r,", r is higher than 0.5, therefore the program is stopped")
    sys.exit(0)

  #Creation of the matrix that will contain the results
  final_matrix = np.zeros((pas_t + 1,pas_x + 1))

  #Initial conditions
  matrix = CI(space,xmax)
  matrix1 = np.zeros((pas_x + 1))
  final_matrix[0][0 : pas_x + 1] = matrix[0 : pas_x + 1]

  #Compute the solution in parallel 
  for i in range(1,pas_t + 1) :  
    #Compute the values at the next time step
    matrix1[1 : pas_x] = matrix[1 : pas_x] + r*(matrix[2 : pas_x + 1] - 2*matrix[1 : pas_x] + matrix[0 : pas_x - 1]) + delta_t*Reac(matrix[1 : pas_x])
  
    #Implementation of boundary conditions (Dirichlet)
    matrix1[0]  =  0.0  
    matrix1[pas_x]  =  0.0  
  
    matrix = matrix1
  
    #Storage of the obtained values into the final matrix 
    final_matrix[i][ : ] = matrix[ : ]
  
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
  test=20

  #Time definition
  t0 = 0
  tmax = 1
  pas_t = 500*test**2

  #Space definition
  x0 = 0
  xmax = 2
  pas_x = 21*test
  main(t0,tmax,pas_t,x0,xmax,pas_x)
