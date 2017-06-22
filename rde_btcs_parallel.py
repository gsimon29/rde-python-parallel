"""
Numerical approximation with parallel computing of the reaction-diffusion equation.
Dirichlet boundary conditions and the method BTCS (Backward-Time Central-Space) are used.
The equation is solved with and without the reaction term.
The transition matrix from a time step to another is sparsed with the function scipy.sparse.diags .
The output files are csv files and it is possible to visualize the solutions with a mp4 movie. 
"""

# -*- coding: utf-8 -*-

from __future__ import division
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg
import os
import sys
import time

def Sol_exa(t,X,xmax) :
  """
  Compute the exact solution at a time step t if the solution
  is known and given par the user
  
  Arguments:
    t (float) : Time.
    X (array) : Space interval or space sub-interval
    xmax (float) : Right boundary of the space interval

  Returns:
    array : Values of the solution at the given time step t
  """
  
  #Exact solution for IC sin(pi*x/x_max) 
  return np.sin(np.pi*X/xmax)*np.exp(-np.pi**2*t/xmax**2)

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
  
  SolExa = Sol_exa(t,space,xmax)
  plt.plot(space,SolExa,label="Exact solution without reaction")
  plt.plot(space,X1,label="Numerical solution without reaction")
  plt.plot(space,X2,label="Numerical solution with reaction exp(-u)")
  plt.text(x0 + (xmax-x0)/10,ymax, "t=" + str("%.3f"% t) + "s", horizontalalignment = 'center', verticalalignment = 'center')
  plt.title ("Numerical solution of the reaction-diffusion equation with Dirichlet BC")
  plt.xlabel ( 'Space')
  plt.ylabel ( 'Concentration')
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
  gap=40
  for j in range(rank*pas_t//(gap*size),(rank + 1)*pas_t//(gap*size)) :
    i=gap*j
    Grap(space,matrix1[i],matrix2[i],times[i],x0,xmax,y_min,y_max,i,rank)
  
  #Convert the images into a mp4 file
  cmd = 'convert rde' + str(rank) + '*.png movie-rde-btcs' + str(rank) + '.mp4'
  os.system(cmd)
  
  #Delete the images
  for j in range(rank*pas_t//(gap*size),(rank + 1)*pas_t//(gap*size)) :
    i=gap*j
    os.remove('rde' + str(rank) + "0"*(10-len(str(i))) + str(i) + '.png') 
  
  comm.allgather(1) #It enable to run the next step when the cores have finished the previous one
  
  if rank==0 :
    #Gather the mp4 files into one
    cmd1 = 'convert movie-rde-btcs*.mp4 Movies-rde-btcs.mp4'
    os.system(cmd1)
  
    #Delete the temporary mp4 files
    for i in range(size) :
      os.remove('movie-rde-btcs' + str(i) + '.mp4')


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

#Initial time taking
start_time = time.time()

#Initiation of the parallel computing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
  
#Time definition
t0 = 0
tmax = 1
pas_t = 5000
delta_t = (tmax-t0)/(pas_t)
times = np.linspace(t0,tmax,pas_t + 1)

#Space definition
x0 = 0
xmax = 2
pas_x = 23

#For the good development of the program, we need (pas_x + 1) to be divisible by size,
#it is neccesary to have good overlaps between the matrices
if (pas_x + 1)%size != 0 :
  pas_x = pas_x + (size-(pas_x + 1)%size)
pas_x_s = pas_x//(size)
delta_x = (xmax-x0)/(pas_x)
space = np.linspace(x0,xmax,pas_x + 1)

#Parameters definition
k = 1
r = k*delta_t/delta_x**2

#Creation of the matrices that will contain the results
final_matrix1 = np.zeros((pas_t + 1,pas_x_s + 4))
final_matrix2 = np.zeros((pas_t + 1,pas_x_s + 4))

#Initial conditions
matrix1 = CI(space[rank*pas_x_s:rank*pas_x_s + pas_x_s + 4],xmax)
matrix2 = CI(space[rank*pas_x_s:rank*pas_x_s + pas_x_s + 4],xmax)
final_matrix1[0,:] = matrix1[:]
final_matrix2[0,:] = matrix2[:]

b1 = np.zeros(pas_x_s + 4)
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

#Compute the solution in parallel 
for n in range(0, pas_t+1) :
  if 0 < rank :
  # Send matrix[2] and matrix[3] to rank-1
    comm.send(matrix1[2], dest = rank-1, tag = 1)
    comm.send(matrix1[3], dest = rank-1, tag = 2)
    comm.send(matrix2[2], dest = rank-1, tag = 3)
    comm.send(matrix2[3], dest = rank-1, tag = 4)

  # Receive matrix[pas_x_s + 2] and matrix[pas_x_s + 3] from rank+1
  if rank < size-1 :
    matrix1[pas_x_s + 2] = comm.recv(source = rank + 1, tag = 1)
    matrix1[pas_x_s + 3] = comm.recv(source = rank + 1, tag = 2)
    matrix2[pas_x_s + 2] = comm.recv(source = rank + 1, tag = 3)
    matrix2[pas_x_s + 3] = comm.recv(source = rank + 1, tag = 4)

  # Send matrix[pas_x_s] and matrix[pas_x_s+1] à rank+1
  if rank < size-1 :
    comm.send(matrix1[pas_x_s], dest = rank + 1, tag = 5)
    comm.send(matrix1[pas_x_s + 1], dest = rank + 1, tag = 6)
    comm.send(matrix2[pas_x_s], dest = rank + 1, tag = 7)
    comm.send(matrix2[pas_x_s + 1], dest = rank + 1, tag = 8)

  # Receive matrix[0] and matrix[1] from rank-1
  if 0 < rank :
    matrix1[0] = comm.recv(source = rank-1, tag = 5)
    matrix1[1] = comm.recv(source = rank-1, tag = 6)
    matrix2[0] = comm.recv(source = rank-1, tag = 7)
    matrix2[1] = comm.recv(source = rank-1, tag = 8)
    
  #Update the values of the matrix B
  b1[:] = matrix1[:]
  b2[:] = matrix2[:] + delta_t*Reac(matrix2[:])
  
  #Implementation of boundary conditions (Dirichlet)
  if rank == 0 :
    b1[0] = 0
    b2[0] = 0
  if rank == size-1 :
    b1[pas_x_s + 3] = 0
    b2[pas_x_s + 3] = 0
    
  #Compute the values at the next time step
  matrix1[:] = scipy.sparse.linalg.spsolve(A, b1)
  matrix2[:] = scipy.sparse.linalg.spsolve(A, b2)
  
  #Storage of the obtained values into the final matrices
  final_matrix1[n][:] = matrix1[:]
  final_matrix2[n][:] = matrix2[:]
  
#Keep only the necessary solutions (delete overlaps between the matrices)
if rank == 0 :
  final_matrix1b = final_matrix1[:,0:pas_x_s + 2]
  final_matrix2b = final_matrix2[:,0:pas_x_s + 2]
elif rank == size-1 :
  final_matrix1b = final_matrix1[:,2:pas_x_s + 4]
  final_matrix2b = final_matrix2[:,2:pas_x_s + 4]
else :
  final_matrix1b = final_matrix1[:,2:pas_x_s + 2]
  final_matrix2b = final_matrix2[:,2:pas_x_s + 2]

#Gather the matrixes of each core
gathered_list1 = comm.allgather(final_matrix1b)
gathered_list2 = comm.allgather(final_matrix2b)
gathered_matrix1 = np.concatenate(gathered_list1,axis = 1)  
gathered_matrix2 = np.concatenate(gathered_list2,axis = 1)

#Name of the path of the output files
path_no_reac="Results/results-rde-btcs-no-reac.csv"
path_reac="Results/results-rde-btcs-with-reac.csv"

#Create DataFrames and convert them into csv files
round_number=4
formats="%."+str(round_number)+"f"
if rank==0:
  tab1=pd.DataFrame(data=gathered_matrix1, index=np.round(times,round_number), columns=np.round(space,round_number))
  tab1.to_csv(path_no_reac,float_format=formats)
if rank==1:
  tab2=pd.DataFrame(data=gathered_matrix2, index=np.round(times,round_number), columns=np.round(space,round_number))
  tab2.to_csv(path_reac,float_format=formats)
  
#Create the mp4 file
Film(gathered_matrix1,gathered_matrix2,pas_t + 1,times,space,x0,xmax,rank,size)

#Final time taking and elapsed time computation
end_time = time.time()
elapsed_time = end_time-start_time
print("\nThe program have run for", round(elapsed_time,4),"s on the core ",rank)
