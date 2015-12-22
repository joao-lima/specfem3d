
#CFLAGS= -g -Wall -O3 -std=gnu99 
CFLAGS=  -Wall -O3 -ffast-math -std=gnu99 
#CFLAGS += -DCONFIG_VERBOSE
CFLAGS += -DCONFIG_BENCHMARK
CFLAGS += -I../
CC = gcc
LDFLAGS = -lm

all: specfem3D-aos-serial 

specfem3D-aos-serial: specfem3D-serial.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *~ specfem3D-aos-serial 

.PHONY: specfem3D-aos-serial clean

