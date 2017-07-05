"""
Numerical approximation with parallel computing of the reaction-diffusion equation.
Dirichlet boundary conditions and the method BTCS (Backward-Time Central-Space) are used.
This script is simplified to test the speed of parallel computing
"""

from mpi4py import MPI
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time

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

def main(t0,tmax,pas_t,x0,xmax,pas_x,comm,rank,size):
  #Initial time taking
  start_time = time.time()
  
  #Time definition
  delta_t = (tmax-t0)/(pas_t)
  times = np.linspace(t0,tmax,pas_t + 1)

#Space definition
#For the good development of the program, we need (pas_x - 3) to be divisible by the number of cores.
#In fact, (pas_x - 3) correspond to (number of point - number of overlaps between two matrices)
#This is neccesary to have good overlaps between the matrices
  if (pas_x - 3)%size != 0 :
    pas_x = pas_x + (size-(pas_x - 3)%size)
  if size >= 4 :
    pas_x_s = pas_x//(size)
  else : 
    pas_x_s = pas_x//(size) - 3//size
  delta_x = (xmax-x0)/(pas_x)
  space = np.linspace(x0,xmax,pas_x + 1)

  #Parameters definition
  k = 1
  r = k*delta_t/delta_x**2

  #Creation of the matrices that will contain the results
  final_matrix2 = np.zeros((pas_t + 1,pas_x_s + 4))

  #Initial conditions
  matrix2 = CI(space[rank*pas_x_s:rank*pas_x_s + pas_x_s + 4],xmax)
  final_matrix2[0,:] = matrix2[:]

  b2 = np.zeros(pas_x_s + 4)

  #Creation of the matrices that will be used to sparse the matrix A
  main  = np.zeros(pas_x_s + 4)
  lower = np.zeros(pas_x_s + 3)
  upper = np.zeros(pas_x_s + 3)
  main[:] = 1 + 2*r
  lower[:] = -r 
  upper[:] = -r  

  #Implementation of boundary conditions (Dirichlet)
  if rank == 0 :
    main[0] = 1
    upper[0] = 0

  if rank == size-1 :
    main[pas_x_s + 3] = 1
    lower[pas_x_s + 2] = 0

  #Sparsing of the matrix A
  A = scipy.sparse.diags(
      diagonals = [main, lower, upper],
      offsets = [0, -1, 1], shape = (pas_x_s + 4, pas_x_s + 4),
      format = 'csr')
    
  #print(rank,A.todense(),"\n")

  #Compute the solution in parallel 
  for n in range(0, pas_t+1) :
    if 0 < rank :
    # Send matrix[2:4] to rank-1
      comm.Send([matrix2[2:4],MPI.FLOAT], dest = rank - 1, tag = 1)

    # Send matrix[-4:-2]] à rank+1
    if rank < size-1 :
      comm.Send([matrix2[-4:-2],MPI.FLOAT], dest = rank + 1, tag = 2)

    # Receive matrix[-2:] from rank+1
    if rank < size-1 :
      comm.Recv([matrix2[-2:],MPI.FLOAT],source = rank + 1, tag = 1)

    # Receive matrix[:2] from rank-1
    if 0 < rank :
      comm.Recv([matrix2[:2],MPI.FLOAT],source = rank - 1, tag = 2)
    
    #Update the values of the matrix B
    b2[:] = matrix2[:] + delta_t*Reac(matrix2[:])
  
    #Implementation of boundary conditions (Dirichlet)
    if rank == 0 :
      b2[0] = 0
    if rank == size-1 :
      b2[pas_x_s + 3] = 0
    
    #Compute the values at the next time step
    matrix2[:] = scipy.sparse.linalg.spsolve(A, b2)
  
    #Storage of the obtained values into the final matrices
    final_matrix2[n][:] = matrix2[:]
  
  #Keep only the necessary solutions (delete overlaps between the matrices)
  if rank == 0 :
    final_matrix2b = final_matrix2[:,0:pas_x_s + 2]
  elif rank == size-1 :
    final_matrix2b = final_matrix2[:,2:pas_x_s + 4]
  else :
    final_matrix2b = final_matrix2[:,2:pas_x_s + 2]

  #Gather the matrixes of each core
  gathered_list2 = comm.allgather(final_matrix2b) 
  gathered_matrix2 = np.concatenate(gathered_list2,axis = 1)

  #Final time taking and elapsed time computation
  end_time = time.time()
  elapsed_time = end_time-start_time
  return round(elapsed_time,4)

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  test=10

  #Time definition
  t0 = 0
  tmax = 1
  pas_t = 500*test**2

  #Space definition
  x0 = 0
  xmax = 2
  pas_x = 21*test
  main(t0,tmax,pas_t,x0,xmax,pas_x,comm,rank,size)
