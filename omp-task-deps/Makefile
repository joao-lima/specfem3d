
CFLAGS=  -Wall -O3 -ffast-math -std=gnu99 
#CFLAGS += -DCONFIG_VERBOSE
CFLAGS += -DCONFIG_BENCHMARK
CC = gcc
LDFLAGS = -lm

all: specfem3D-omp-task-deps

specfem3D-omp-task-deps: specfem3D-omp-task-deps.c kernels.c
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *~ specfem3D-omp-task-deps 

.PHONY: specfem3D-omp-task-deps 

