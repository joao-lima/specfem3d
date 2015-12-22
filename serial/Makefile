
#CFLAGS= -g -Wall -O3 -std=gnu99 
CFLAGS=  -Wall -O3 -ffast-math -std=gnu99 
#CFLAGS += -DCONFIG_VERBOSE
CFLAGS += -DCONFIG_BENCHMARK
CFLAGS += -I../
CC = gcc
LDFLAGS = -lm

all: specfem3D-serial 

specfem3D-serial: specfem3D_single_precision_with_Deville.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *~ specfem3D-serial 

.PHONY: specfem3D-serial clean

