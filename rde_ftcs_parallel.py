"""
Numerical approximation with parallel computing of the reaction-diffusion equation.
Dirichlet boundary conditions and the method FTCS (Forward-Time Central-Space) are used.
The output files are csv files and it is possible to visualize the solutions with a mp4 movie. 
"""

# -*- coding :  utf-8 -*-

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpi4py import MPI
import os
import sys
import time

def SolExacte(t,X,xmax) : 
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

def Grap(space,X,t,x0,xmax,ymin,ymax,i,rank) : 
  """
  Create the graph of the solution at a given time step t and save it.

  Arguments:
    space (array(1,n)) : Interval [x0,xmax]
    X (array(1,n)) : Numerical solution found at the time step t, its lenght must be equal to the lenght of space
    t (float) : Time
    x0,xmax (float) : Left and right boundaries of the space interval
    ymin,ymax (float) : Minimal value and maximal value of the solutions
    i (int) : Used to name the graph file
    rank (int) : Rank of the core that run the program
  """

  SolExa = SolExacte(t,space,xmax)
  plt.plot(space,SolExa,label = "Exact solution without reaction")
  plt.plot(space,X,label = "Numerical solution with reaction exp(-u)")
  plt.text(x0 + (xmax-x0)/10,ymax, "t=" + str("%.3f"% t) + "s", horizontalalignment = 'center', verticalalignment = 'center')
  plt.title ("Numerical solution of the reaction-diffusion equation with Dirichlet BC")
  plt.xlabel ( 'x')
  plt.ylabel ( 'y')
  plt.ylim(ymin,1.1*ymax)
  plt.xlim(x0,xmax)
  plt.legend(loc=1,prop={'size':7.5})
  plt.savefig('rde' + str(rank) + "0"*(10-len(str(i))) + str(i) + '.png', transparent = False)
  plt.clf()

def Film(gathered_matrix,pas_t,times,space,x0,xmax,rank,size) : 
  """
  Create a movie to visualize the results

  Arguments:
    gathered_matrix (array(n,m)) : Numerical solutions
    pas_t (int) : Number of points used to divide the time interval
    times (array(1,n)) : Time
    space (array(1,m)) : Interval [x0,xmax]
    x0,xmax (float) : Left and right boundaries of the space interval
    rank (int) : Rank of the core that run the program
    size (int) : Number of cores that run the program in the same time
  """

  #Compute the minimum and the maximum of the solutions
  ymin = gathered_matrix.min()
  ymax = gathered_matrix.max()
  
  #Create a graph for some time steps with a constant gap
  gap=40
  for j in range((rank*pas_t)//(gap*size),((rank + 1)*pas_t)//(gap*size)) : 
    i=gap*j
    Grap(space,gathered_matrix[i],times[i],x0,xmax,ymin,ymax,i,rank)
  
  #Convert the images into a mp4 file
  cmd  =  'convert rde' + str(rank) + '*.png Movie-rde-ftcs' + str(rank) + '.mp4'
  os.system(cmd)
  
  #Delete the images
  for j in range((rank*pas_t)//(gap*size),((rank + 1)*pas_t)//(gap*size)) : 
    i=gap*j
    os.remove('rde' + str(rank) + "0"*(10-len(str(i))) + str(i) + '.png') 
  
  comm.allgather(1) #It enable to run the next step when the cores have finished the previous one
  
  if rank==0: 
    #Gather the mp4 files into one
    cmd1  =  'convert Movie-rde-ftcs*.mp4 Movies-rde-ftcs.mp4'
    os.system(cmd1)
  
    #Delete the temporary mp4 files
    for i in range(size) :  
      os.remove('Movie-rde-ftcs' + str(i) + '.mp4')

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
pas_x = 21

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
  matrix1[1 : pas_x_s + 1] = matrix[1 : pas_x_s + 1] + r*(matrix[2 : pas_x_s + 2] - 2*matrix[1 : pas_x_s + 1] + matrix[0 : pas_x_s]) + delta_t*Reac(matrix[1 : pas_x_s + 1])
  
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
gathered_list = comm.allgather(final_matrix_2)
gathered_matrix = np.concatenate(gathered_list,axis = 1)

#Name of the path of the output file
path="Results/results-rde-ftcs.csv"

#Create DataFrames and convert them into csv files
round_number=4
formats="%."+str(round_number)+"f"
if rank==0:
  tab1=pd.DataFrame(data=gathered_matrix, index=np.round(times,round_number), columns=np.round(space,round_number))
  tab1.to_csv(path,float_format=formats)

#Create the mp4 file
Film(gathered_matrix,pas_t + 1,times,space,x0,xmax,rank,size)
  
#Final time taking and elapsed time computation
end_time = time.time()
elapsed_time = end_time-start_time
print("\nThe program have run for", round(elapsed_time,4),"s on the core ",rank)
