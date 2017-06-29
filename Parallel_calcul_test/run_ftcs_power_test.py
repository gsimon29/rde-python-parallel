# -*- coding :  utf-8 -*-

from __future__ import division
import numpy as np
from mpi4py import MPI
import sys
import time
import csv
import rde_ftcs_power_test_control as control
import rde_ftcs_power_test_parallel as parallel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

i0 = 4
imax = 5

first_time=True

#Time definition
t0 = 0
tmax = 1

#Space definition
x0 = 0
xmax = 2

results=np.zeros(((imax+2-i0)))
#If the parallel computing is used we put 1 (True) is the last row, otherwise we put 0 (False)
if size==1:
  results[-1] = 0
else:
  results[-1] = 1

j=0
for i in range(i0,imax+1) :
  pas_t = 11**i
  pas_x = 3**i
  if size==1:
    results[j] = control.main(t0,tmax,pas_t,x0,xmax,pas_x)
  else:
    results[j] = parallel.main(t0,tmax,pas_t,x0,xmax,pas_x,comm,rank,size)
  j=j+1
with open("power_test.csv","a") as f :
  writer = csv.writer(f, delimiter=',')
  if first_time and size==1 :
    column_name1=np.array([11**i for i in range(i0,imax+1)])
    column_name2=np.array([3**i for i in range(i0,imax+1)])
    writer.writerow(column_name1)
    writer.writerow(column_name2)
  writer.writerow(results)



