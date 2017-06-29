# -*- coding :  utf-8 -*-

from __future__ import division
import numpy as np
from mpi4py import MPI
import sys
import time
import csv
import rde_ftcs_power_test_control as ftcs_control
import rde_ftcs_power_test_parallel as ftcs_parallel
import rde_btcs_power_test_control as btcs_control
import rde_btcs_power_test_parallel as btcs_parallel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

i0 = 6
imax = 6

first_time=True

#Time definition
t0 = 0
tmax = 1

#Space definition
x0 = 0
xmax = 2

ftcs_results=np.zeros(((imax+2-i0)))
btcs_results=np.zeros(((imax+2-i0)))
if size==1:
  ftcs_results[-1] = 0
  btcs_results[-1] = 2
else:
  ftcs_results[-1] = 1
  btcs_results[-1] = 3

j=0
for i in range(i0,imax+1) :
  pas_t = 10**i
  pas_x = 3**i
  if size==1:
    #ftcs_results[j] = ftcs_control.main(t0,tmax,pas_t,x0,xmax,pas_x)
    btcs_results[j] = btcs_control.main(t0,tmax,pas_t,x0,xmax,pas_x)
  else:
    #ftcs_results[j] = ftcs_parallel.main(t0,tmax,pas_t,x0,xmax,pas_x,comm,rank,size)
    btcs_results[j] = btcs_parallel.main(t0,tmax,pas_t,x0,xmax,pas_x,comm,rank,size)
  j=j+1
with open("test.csv","a") as f :
  writer = csv.writer(f, delimiter=',')
  if first_time and size==1 :
    name1=["0=Exp_norm","1=Exp_parra","2=Imp_norm","3=Imp_parra"]
    name2=["First_row=nb_points_t","Second_row=nb_points_x"]
    writer.writerow(name1)
    writer.writerow(name2)
    row_name1=np.array([10**i for i in range(i0,imax+1)])
    row_name2=np.array([3**i for i in range(i0,imax+1)])
    writer.writerow(row_name1)
    writer.writerow(row_name2)
  #writer.writerow(ftcs_results)
  writer.writerow(btcs_results)



