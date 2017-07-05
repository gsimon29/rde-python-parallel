---
output:
  pdf_document: default
  html_document: default
---
# github.com/gsimon29/rde-python-parallel
Python scripts that use MPI parallel computing to approximate the solution of Reaction-Diffusion Equations

__rde_btcs_parallel.py__ :
Numerical approximation with parallel computing of the reaction-diffusion equation.
Dirichlet boundary conditions and the method BTCS (Backward-Time Central-Space) are used.
The equation is solved with and without the reaction term.
The transition matrix from a time step to another is sparsed with the function scipy.sparse.diags .
The output files are csv files and it is possible to visualize the solutions with a mp4 movie. 

__rde_ftcs_parallel.py__ :
Numerical approximation with parallel computing of the reaction-diffusion equation.
Dirichlet boundary conditions and the method FTCS (Forward-Time Central-Space) are used.
The output files are csv files and it is possible to visualize the solutions with a mp4 movie. 

__rde_cn_parallel.py__ :
Numerical approximation with parallel computing of the reaction-diffusion equation.
Von Neumann boundary conditions and the Crank-Nicolson method are used.
The equation is solved with and without the reaction term.
The transition matrix from a time step to another is sparsed with the function scipy.sparse.diags .
The output files are csv files and it is possible to visualize the solutions with a mp4 movie. 

__rde_system_parallel.py__
Numerical approximation with parallel computing of a system of two reaction-diffusion equations.
Von Neumann boundary conditions and the Crank-Nicolson method are used.
The output files are csv files and it is possible to visualize the solutions with a mp4 movie. 

The system is :

u_t = u_xx + r1(u,v)

v_t = v_xx + r2(u,v) 

We take Lokta-Voltera system as an example.

To run the program in the terminal, you can use :
mpiexec -n 4 python3 rde_system_parallel.py 