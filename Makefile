
CFLAGS= -g -Wall -O2 -std=gnu99
#CFLAGS=  -Wall -O3 -ffast-math -std=gnu99 
CC = gcc
LDFLAGS = -lm

all: specfem3D-serial specfem3D-omp-task-deps

specfem3D-omp-task-deps: specfem3D-omp-task-deps.c kernels.c
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LDFLAGS)

specfem3D-serial: specfem3D_single_precision_with_Deville.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf *.o specfem3D_seq specfem3D_omp *~
