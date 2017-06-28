import os
os.system("python3 run_ftcs_power_test.py")
os.system("mpiexec -n 4 python3 run_ftcs_power_test.py")
