"""
Numerical approximation with parallel computing of the reaction-diffusion equation.
Dirichlet boundary conditions and the method FTCS (Forward-Time Central-Space) are used.
The output files are csv files and it is possible to visualize the solutions with a mp4 movie. 
"""

# -*- coding :  utf-8 -*-

from __future__ import division
import numpy as np
import sys
from mpi4py import MPI
import time

def main(t0,tmax,pas_t,x0,xmax,pas_x,comm,rank,size):
  #Initial time taking
  start_time = time.time()


  #Definition of time interval
  delta_t = (tmax-t0)/(pas_t)
  times = np.linspace(t0,tmax,pas_t + 1)

  #Definition of the space interval
  
  #For the good development of the program, we need (pas_x - 1) to be divisible by size,
  #it is neccesary to have good overlaps between the matrices
  if (pas_x - 1)%size != 0 :
    pas_x = pas_x + (size-(pas_x - 1)%size)
  delta_x = (xmax-x0)/(pas_x)
  pas_x_s = pas_x//(size)
  space = np.linspace(x0,xmax,pas_x + 1)

  #Parameters definition
  k = 1
  r = k*delta_t/delta_x**2
  if r>=  0.5 : 
    print("k*delta_t/delta_x**2 is equal to ",r,", r is higher than 0.5, therefore the program is stopped")
    sys.exit(0)

  #Creation of the matrix that will contain the results
  final_matrix = np.zeros((pas_t + 1,pas_x_s + 2))

  #Initial conditions
  matrix = CI(space[rank*pas_x_s : rank*pas_x_s + pas_x_s + 2],xmax)
  matrix1 = np.zeros((pas_x_s + 2))
  final_matrix[0][0 : pas_x_s + 2] = matrix[0 : pas_x_s + 2]

  #Compute the solution in parallel 
  for i in range(1,pas_t + 1) :  
    if 0 < rank : 
    # Send matrix[1] to rank-1
      comm.send(matrix[1], dest = rank-1, tag = 1)

    # Receive matrix[pas_x_s] from rank + 1
    if rank < size-1 : 
      matrix[pas_x_s + 1]  =  comm.recv(source = rank + 1, tag = 1)

    # Send matrix[pas_x_s] to rank + 1
    if rank < size-1 : 
      comm.send(matrix[pas_x_s], dest = rank + 1, tag = 2)

    # Receive matrix[0] from rank-1
    if 0 < rank : 
      matrix[0]  =  comm.recv(source = rank-1, tag = 2)

    #Compute the values at the next time step
    matrix1[1 : pas_x_s + 1] = matrix[1 : pas_x_s + 1] + r*(matrix[2 : pas_x_s + 2] - 2*matrix[1 : pas_x_s + 1] + matrix[0 : pas_x_s]) +   delta_t*Reac(matrix[1 : pas_x_s + 1])
  
    #Implementation of boundary conditions (Dirichlet)
    if rank  ==  0 : 
      matrix1[0]  =  0.0  
    elif rank  ==  size-1 : 
      matrix1[pas_x_s + 1]  =  0.0  
  
    matrix = matrix1
  
    #Storage of the obtained values into the final matrix 
    final_matrix[i][ : ] = matrix[ : ]

  #Keep only the necessary solutions (delete overlaps between the matrices)
  if rank == 0 : 
    final_matrix_2 = final_matrix[ : ,0 : pas_x_s + 1]
  elif rank == size-1 : 
    final_matrix_2 = final_matrix[ : ,1 : pas_x_s + 2]
  else : 
    final_matrix_2 = final_matrix[ : ,1 : pas_x_s + 1]

  #Gather the matrixes of each core
  gathered_list = comm.gather(final_matrix_2)
  if rank==0:
    gathered_matrix = np.concatenate(gathered_list,axis = 1)
  
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
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  test=20

  #Time definition
  t0 = 0
  tmax = 1
  pas_t = 500*test**2

  #Space definition
  x0 = 0
  xmax = 2
  pas_x = 21*test
  main(t0,tmax,pas_t,x0,xmax,pas_x,comm,rank,size)
