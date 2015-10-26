# specfem3d
SPECFEM3D example using OpenMP.

First, unzip the data input:
gunzip DB/proc000000_reg1_database.dat.gz

Source code used:
- specfem3D_single_precision_with_Deville.c
Serial version from SPECFEM3D examples (https://github.com/geodynamics/specfem3d)

- specfem3D-omp-task-deps.c
Parallel OpenMP version with dependent tasks, based on OmpSs version.

 
