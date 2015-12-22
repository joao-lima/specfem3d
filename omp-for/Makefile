
#CFLAGS= -g -Wall -O3 -std=gnu99 
CFLAGS=  -Wall -O3 -ffast-math -std=gnu99 
#CFLAGS += -DCONFIG_VERBOSE
CFLAGS += -DCONFIG_BENCHMARK
CFLAGS += -I../
CC = gcc
LDFLAGS = -lm

all: specfem3D-omp-for

specfem3D-omp-for: specfem3D-omp-for.c 
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *~ specfem3D-omp-for 

.PHONY: specfem3D-omp-for

