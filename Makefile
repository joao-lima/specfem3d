
CFLAGS= -g -Wall -O3 -std=gnu99 
CFLAGS += -DCONFIG_VERBOSE
#CFLAGS += -DCONFIG_BENCHMARK
#CFLAGS=  -Wall -O3 -ffast-math -std=gnu99 
CC = gcc
LDFLAGS = -lm

all: specfem3D-serial teste specfem3D-omp-task-deps specfem3D-omp-for specfem3D-omp-for-static safe_version

specfem3D-omp-task-deps: specfem3D-omp-task-deps.c kernels.c
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LDFLAGS)

teste: teste.c
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LDFLAGS)	

specfem3D-omp-for: specfem3D-omp-for.c 
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LDFLAGS)

specfem3D-serial: specfem3D_single_precision_with_Deville.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

specfem3D-profiler: specfem3D_single_precision_with_Deville.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

specfem3D-omp-for-static: specfem3D-omp-for-static.c
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LDFLAGS)

safe_version: safe_version.c
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LDFLAGS)


clean:
	rm -rf *.o *~ specfem3D-serial teste specfem3D-omp-task-deps specfem3D-omp-for specfem3D-omp-for-static safe_version

.PHONY: specfem3D-omp-task-deps teste specfem3D-omp-for specfem3D-serial specfem3D-profiler specfem3D-omp-for-static safe_version

