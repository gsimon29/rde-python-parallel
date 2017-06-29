import os
#Define the number of cores
n=4

#Run the following lines into the terminal
os.system("python3 run_power_test.py")
os.system("mpiexec -n "+str(n)+" python3 run_power_test.py")
