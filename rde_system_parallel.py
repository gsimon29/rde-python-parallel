"""
Numerical approximation with parallel computing of a system of two reaction-diffusion equations.
Von Neumann boundary conditions and the Crank-Nicolson method are used.
The output files are csv files and it is possible to visualize the solutions with a mp4 movie. 

The system is :

u_t = u_xx + r1(u,v)
v_t = v_xx + r2(u,v) 

We take Lokta-Voltera system as an example

To run the program in the terminal, you can use :
mpiexec -n 4 python3 rde_system_parallel.py 
"""

from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg
import os
import sys
import time
import random

def Grap(space,X1,X2,t,x0,xmax,ymin,ymax,i,rank) :
  """
  Create the graph of the solution at a given time step t and save it.

  Arguments:
    space (array(1,n)) : Interval [x0,xmax]
    X1,X2 (array(1,n)) : Numerical solutions found at the time step t, their lenght must be equal to the lenght of space
    t (float) : Time
    x0,xmax (float) : Left and right boundaries of the space interval
    ymin,ymax (float) : Minimal value and maximal value of the solutions
    i (int) : Used to name the graph file
    rank (int) : Rank of the core that run the program
  """
  plt.plot(space,X1,label="Preys")
  plt.plot(space,X2,label="Predators")
  plt.text(x0 + (xmax-x0)/10,ymax, "t=" + str("%.3f"% t) + "s", horizontalalignment = 'center', verticalalignment = 'center')
  plt.title ("Lokta-Voltera System")
  plt.xlabel ( 'Space')
  plt.ylabel ( 'Number of individuals')
  plt.ylim(ymin,ymax + 0.1*abs(ymax))
  plt.xlim(x0,xmax)
  plt.legend(loc=1,prop={'size':7.5})
  plt.savefig('rde' + str(rank) + "0"*(10-len(str(i))) + str(i) + '.png', transparent=False)
  plt.clf()

def Film(matrix1,matrix2,pas_t,times,space,x0,xmax,rank,size) :
  """
  Create a movie to visualize the results

  Arguments:
    matrix1,matrix2 (array(n,m)) : Numerical solutions
    pas_t (int) : Number of points used to divide the time interval
    times (array(1,n)) : Time
    space (array(1,m)) : Interval [x0,xmax]
    x0,xmax (float) : Left and right boundaries of the space interval
    rank (int) : Rank of the core that run the program
    size (int) : Number of cores that run the program in the same time
  """
  #Compute the minimum and the maximum of the solutions
  y_min = min(matrix1.min(),matrix2.min())
  y_max = max(matrix1.max(),matrix2.max())
  
  #Create a graph for some time steps with a constant gap
  gap=pas_t//500
  for j in range(rank*pas_t//(gap*size),(rank + 1)*pas_t//(gap*size)) :
    i=gap*j
    Grap(space,matrix1[i],matrix2[i],times[i],x0,xmax,y_min,y_max,i,rank)
  
  #Convert the images into a mp4 file
  cmd = 'convert rde' + str(rank) + '*.png movie-rde-system' + str(rank) + '.mp4'
  os.system(cmd)
  
  #Delete the images
  for j in range(rank*pas_t//(gap*size),(rank + 1)*pas_t//(gap*size)) :
    i=gap*j
    os.remove('rde' + str(rank) + "0"*(10-len(str(i))) + str(i) + '.png') 
  
  comm.allgather(1) #It enable to run the next step when the cores have finished the previous one
  
  if rank==0 :
    #Gather the mp4 files into one
    cmd1 = 'convert movie-rde-system*.mp4 lokta-voltera-without-perturbation.mp4'
    os.system(cmd1)
  
    #Delete the temporary mp4 files
    for i in range(size) :
      os.remove('movie-rde-system' + str(i) + '.mp4')


def CI1(x,pas_x_s,rank,size) :
  """
  Function that compute the initial condition

  Arguments:
    x (array(1,n)) : Space interval or space sub-interval
    pas_x_s (int) : Space step used for one core
    rank (int) : Number of the core that run the program
    size (int) : Total number of cores
    
  Returns:
    array(1,n) : 
  """
  CI = np.zeros(len(x))
  if rank == 1 :
    CI[pas_x_s//4:3*(pas_x_s//4)] = 2
  return CI
    
  
def CI2(x,pas_x_s,rank,size) :
  """
  Function that compute the initial condition

  Arguments:
    x (array(1,n)) : Space interval or space sub-interval
    pas_x_s (int) : Space step used for one core
    rank (int) : Number of the core that run the program
    size (int) : Total number of cores
    
  Returns:
    array(1,n) : 
  """
  CI = np.zeros(len(x))
  if rank == 2 :
    CI[pas_x_s//6:5*(pas_x_s//6)] = 2
  return CI
  
def Reac1(u,v) :
  """
  Function that compute the reaction term of u

  Arguments:
    u,v (array(1,n)) : Solutions of u and v at a given time step t
    
  Returns:
    array(1,n) : Values of the reaction
  """
  a = 2/3
  b = 4/3
  return a*u-b*u*v

def Reac2(u,v) :
  """
  Function that compute the reaction term of v

  Arguments:
    u,v (array(1,n)) : Solutions of u and v at a given time step t
    
  Returns:
    array(1,n) : Values of the reaction
  """
  d = 1
  g = 1
  return d*u*v-g*v

#Initial time taking
start_time = time.time()

#Initiation of the parallel computing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
  
#Time definition
t0 = 0
tmax = 15
pas_t = (tmax-t0)*1000
delta_t = (tmax-t0)/(pas_t)
times = np.linspace(t0,tmax,pas_t + 1)

#Space definition
x0 = 0
xmax = 10
pas_x = (xmax-x0)*10

#For the good development of the program, we need (pas_x - 5) to be divisible by the number of cores.
#In fact, (pas_x - 5) correspond to (number of point - number of overlaps between two matrices)
#This is neccesary to have good overlaps between the matrices
if (pas_x - 5)%size != 0 :
  pas_x = pas_x + (size-(pas_x - 5)%size) 
if size >= 6:
  pas_x_s = pas_x//size
else :
  pas_x_s = pas_x//size - 5//size
  
delta_x = (xmax-x0)/(pas_x)
space = np.linspace(x0,xmax,pas_x + 1)

#Parameters definition
k1 = 0.5
k2 = 0.2
r1 = k1*delta_t/delta_x**2
r2 = k2*delta_t/delta_x**2

#Creation of the matrix that will contain the results
final_matrix1 = np.zeros((pas_t + 1,pas_x_s + 6))
final_matrix2 = np.zeros((pas_t + 1,pas_x_s + 6))

#Initial conditions
matrix1 = CI1(space[rank*pas_x_s:rank*pas_x_s + pas_x_s + 6],pas_x_s,rank,size)
matrix2 = CI2(space[rank*pas_x_s:rank*pas_x_s + pas_x_s + 6],pas_x_s,rank,size)
final_matrix1[0,:] = matrix1[:]
final_matrix2[0,:] = matrix2[:]

b1 = np.zeros(pas_x_s + 6)
b2 = np.zeros(pas_x_s + 6)

#Creation of the matrices that will be used to sparse the matrix A
main1  = np.zeros(pas_x_s + 6)
lower1 = np.zeros(pas_x_s + 5)
upper1 = np.zeros(pas_x_s + 5)
main2  = np.zeros(pas_x_s + 6)
lower2 = np.zeros(pas_x_s + 5)
upper2 = np.zeros(pas_x_s + 5)

main1[:] = 1 + r1
lower1[:] = -r1/2 
upper1[:] = -r1/2 
main2[:] = 1 + r2
lower2[:] = -r2/2 
upper2[:] = -r2/2 

#Implementation of boundary conditions (Von Neumann)
if rank == 0 :
  upper1[0] = -r1
  upper2[0] = -r2
if rank == size-1 :
  lower1[pas_x_s + 4] = -r1
  lower2[pas_x_s + 4] = -r2

#Sparsing of the matrix A
A1 = scipy.sparse.diags(
    diagonals = [main1, lower1, upper1],
    offsets = [0, -1, 1], shape = (pas_x_s + 6, pas_x_s + 6),
    format = 'csr')
A2 = scipy.sparse.diags(
    diagonals = [main2, lower2, upper2],
    offsets = [0, -1, 1], shape = (pas_x_s + 6, pas_x_s + 6),
    format = 'csr')
    
#print(rank,A.todense(),"\n")

#Creation of the matrices that will be used to receive information from the other cores
recv1 = np.zeros((4))
recv2 = np.zeros((4))
recv3 = np.zeros((4))
recv4 = np.zeros((4))

#Compute the solution in parallel 
for n in range(1, pas_t+1) :
  if 0 < rank :
  # Send matrix[3:7] to rank-1
    comm.Send([matrix1[3:7],MPI.FLOAT], dest = rank - 1, tag = 1)
    
    comm.Send([matrix2[3:7],MPI.FLOAT], dest = rank - 1, tag = 2)

  # Send matrix[pas_x_s] and matrix[pas_x_s+1] à rank+1
  if rank < size-1 :
    comm.Send([matrix1[-7:-3],MPI.FLOAT], dest = rank + 1, tag = 3)
    
    comm.Send([matrix2[-7:-3],MPI.FLOAT], dest = rank + 1, tag = 4)
    
  # Receive matrix[pas_x_s + 2] and matrix[pas_x_s + 3] from rank+1
  if rank < size-1 :
    comm.Recv([recv1, MPI.FLOAT],source = rank + 1, tag = 1)
    matrix1[-3:] = recv1[:-1]
    next_val1 = recv1[-1]
    
    comm.Recv([recv2, MPI.FLOAT],source = rank + 1, tag = 2)
    matrix2[-3:] = recv2[:-1]
    next_val2 = recv2[-1]

  # Receive matrix[0] and matrix[1] from rank-1
  if 0 < rank :
    comm.Recv([recv3, MPI.FLOAT],source = rank - 1, tag = 3)
    prev_val1 = recv3[0]
    matrix1[:3] = recv3[1:]
    
    comm.Recv([recv4, MPI.FLOAT],source = rank - 1, tag = 4)
    prev_val2 = recv4[0]
    matrix2[:3] = recv4[1:]
    
  #Update the values of the matrix B
  b1[1:-1] = (1-r1)*matrix1[1:-1]+r1*(matrix1[:-2]+matrix1[2:])/2 + delta_t*Reac1(matrix1[1:-1],matrix2[1:-1])
  b2[1:-1] = (1-r2)*matrix2[1:-1]+r2*(matrix2[:-2]+matrix2[2:])/2 + delta_t*Reac2(matrix1[1:-1],matrix2[1:-1])

  if rank==0 :
    b1[0]=(1-r1)*matrix1[0] + r1*matrix1[1] + delta_t*Reac1(matrix1[0],matrix2[0])
    b2[0]=(1-r2)*matrix2[0] + r2*matrix2[1] + delta_t*Reac2(matrix1[0],matrix2[0])
  else : 
    b1[0]=(1-r1)*matrix1[0] + r1*(prev_val1+matrix1[1])/2 + delta_t*Reac1(matrix1[0],matrix2[0])
    b2[0]=(1-r2)*matrix2[0] + r2*(prev_val2+matrix2[1])/2 + delta_t*Reac2(matrix1[0],matrix2[0])
    
  if rank==size-1 :
    b1[-1]=(1-r1)*matrix1[-1] + r1*matrix1[-2] + delta_t*Reac1(matrix1[-1],matrix2[-1])
    b2[-1]=(1-r2)*matrix2[-1] + r2*matrix2[-2] + delta_t*Reac2(matrix1[-1],matrix2[-1])
  else : 
    b1[-1]=(1-r1)*matrix1[-1] + r1*(next_val1+matrix1[-2])/2 + delta_t*Reac1(matrix1[-1],matrix2[-1])
    b2[-1]=(1-r2)*matrix2[-1] + r2*(next_val2+matrix2[-2])/2 + delta_t*Reac2(matrix1[-1],matrix2[-1])
      
  #Compute the values at the next time step
  matrix1[:] = scipy.sparse.linalg.spsolve(A1, b1)
  matrix2[:] = scipy.sparse.linalg.spsolve(A2, b2)
  
  #Storage of the obtained values into the final matrix
  final_matrix1[n][:] = matrix1[:]
  final_matrix2[n][:] = matrix2[:]
  
#Keep only the necessary solutions (delete overlaps between the matrices)
if rank == 0 :
  final_matrix1b = final_matrix1[:,0:pas_x_s + 3]
  final_matrix2b = final_matrix2[:,0:pas_x_s + 3]
elif rank == size-1 :
  final_matrix1b = final_matrix1[:,3:pas_x_s + 6]
  final_matrix2b = final_matrix2[:,3:pas_x_s + 6]
else :
  final_matrix1b = final_matrix1[:,3:pas_x_s + 3]
  final_matrix2b = final_matrix2[:,3:pas_x_s + 3]

#Gather the matrices of each core
gathered_list1 = comm.allgather(final_matrix1b)
gathered_list2 = comm.allgather(final_matrix2b)
gathered_matrix1 = np.concatenate(gathered_list1,axis = 1)  
gathered_matrix2 = np.concatenate(gathered_list2,axis = 1)

#Name of the path of the output files
path_sol_u="results-rde-system-u.csv"
path_sol_v="results-rde-system-v.csv"

#Create DataFrames and convert them into csv files
round_number=4
formats="%."+str(round_number)+"f"
if rank==0:
  tab1=pd.DataFrame(data=gathered_matrix1, index=np.round(times,round_number), columns=np.round(space,round_number))
  tab1.to_csv(path_sol_u,float_format=formats)
if rank==1:
  tab2=pd.DataFrame(data=gathered_matrix2, index=np.round(times,round_number), columns=np.round(space,round_number))
  tab2.to_csv(path_sol_v,float_format=formats)

#Create the mp4 file
Film(gathered_matrix1,gathered_matrix2,pas_t + 1,times,space,x0,xmax,rank,size)

#Final time taking and elapsed time computation
end_time = time.time()
elapsed_time = end_time-start_time
print("\nThe program have run for", round(elapsed_time,4),"s on the core ",rank)
