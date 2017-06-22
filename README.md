---
output:
  pdf_document: default
  html_document: default
---
# github.com/gsimon29/rde-python-parallel
Python scripts that use parallel computing to approximate the solution of Reaction-Diffusion Equations

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