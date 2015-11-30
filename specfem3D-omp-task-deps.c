// compute_max changed; now not indexed through ibool
// NDIM moved to rightmost dimension
/*
!=====================================================================
!
!          S p e c f e m 3 D  G l o b e  V e r s i o n  4 . 0
!          --------------------------------------------------
!
!          Main authors: Dimitri Komatitsch and Jeroen Tromp
!    Seismological Laboratory, California Institute of Technology, USA
!             and University of Pau / CNRS / INRIA, France
! (c) California Institute of Technology and University of Pau / CNRS / INRIA
!                            February 2008
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================
*/

//
// All the arrays below use static memory allocation,
// using constant sizes defined in values_from_mesher.h.
// This is done purposely to improve performance (Fortran compilers
// can optimize much more when the size of the loops and arrays
// is known at compile time).
// NGLLX, NGLLY and NGLLZ are set equal to 5,
// therefore each element contains NGLLX * NGLLY * NGLLZ = 125 points.
//

//
// All the calculations are done in single precision.
// We do not need double precision in SPECFEM3D.
//

//#define CONFIG_VERBOSE	1
//#define CONFIG_BENCHMARK	1

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

// use the Deville et al. (2002) inlined products or not
#define USE_DEVILLE_INLINED_PRODUCTS

// Rosa, you can uncomment this to use the paths to your local copies of the data files
#define USE_PATHS_ROSA

// include values created by the mesher
// done for performance only using static allocation to allow for loop unrolling
#ifndef USE_PATHS_ROSA
#include "../multi_GPU_MPI/DATABASES_FOR_SOLVER/values_from_mesher_C.h"
#else
#include "./DB/values_from_mesher_C.h"
#endif

// constant value of the time step in the main time loop
#define deltatover2 0.5f*deltat
#define deltatsqover2 0.5f*deltat*deltat

// for the source time function
#define pi 3.141592653589793f
#define f0 (1.f / 50.f)
#define t0 (1.2f / f0)
#define a pi*pi*f0*f0

// number of GLL integration points in each direction of an element (degree plus one)
#define NGLLX 5
#define NGLLY 5
#define NGLLZ 5

// for the Deville et al. (2002) inlined products
#define NGLL2  25		// NGLLX^2

// 3-D simulation
#define NDIM 3

// displacement threshold above which we consider that the code became unstable
#define STABILITY_THRESHOLD 1.e+25f

//#define NTSTEP_BETWEEN_OUTPUT_INFO 1000
#define NTSTEP_BETWEEN_OUTPUT_INFO 1000

// approximate density of the geophysical medium in which the source is located
// this value is only a constant scaling factor therefore it does not really matter
#define rho 4500.f

// CellSs or SMPSs: Increase block size in order to reduce overhead.
// DK DK we should see if there are roundoff problems here when NSPEC is not a multiple of NUMBER_OF_THREADS_USED
#define BS_NGLOB 4000
#define BS_NSPEC 60

// define some constants to group the arrays to reduce the number of arguments
// to send to the tasks
#define XIX 0
#define XIY 1
#define XIZ 2
#define ETAX 3
#define ETAY 4
#define ETAZ 5
#define GAMMAX 6
#define GAMMAY 7
#define GAMMAZ 8

#define KAPPA 0
#define MU 1

#define X 0
#define Y 1
#define Z 2

#define YZ 0
#define XZ 1
#define XY 2

#define FLAG_hprime_xx 0
#define FLAG_hprime_xxT 1
#define FLAG_hprimewgll_xx 2
#define FLAG_hprimewgll_xxT 3
#define FLAG_wgllwgll_YZ 4
#define FLAG_wgllwgll_XZ 5
#define FLAG_wgllwgll_XY 6

const int dim = NDIM;
const int dim_x = NGLLX;
const int dim_y = NGLLY;
const int dim_z = NGLLZ;
const int nglob = NGLOB;
const int bs_nglob = BS_NGLOB;
const int bs_nspec = BS_NSPEC;


long t_start, t_end;

// declare all the functions
static long usecs ();

static void clear (int actual_size, float displ[actual_size][NDIM],
		   float veloc[actual_size][NDIM],
		   float accel[actual_size][NDIM]);

static void gather (int actual_size,
		    float displ[NGLOB][NDIM],
		    int ibool[actual_size][NGLLZ][NGLLY][NGLLX],
		    float dummy_loc[actual_size][NGLLZ][NGLLY][NGLLX][NDIM]);

static void scatter (int actual_size,
		     int ibool[actual_size][NGLLZ][NGLLY][NGLLX],
		     float sum_terms[actual_size][NGLLZ][NGLLY][NGLLX][NDIM],
		     float accel[NGLOB][NDIM]);

static void process_element (int actual_size,
			     float
			     dummy_loc[actual_size][NGLLZ][NGLLY][NGLLX]
			     [NDIM],
#ifdef USE_DEVILLE_INLINED_PRODUCTS
			     float all_matrices[NDIM + 4][NGLLX][NGLLX],
#else
			     float hprime_xx[NGLLX][NGLLX],
			     float hprimewgll_xx[NGLLX][NGLLX],
			     float wgllwgll_all[NGLLZ][NGLLY][NDIM],
#endif
			     float
			     jacobian_matrix[actual_size][NGLLZ][NGLLY][NGLLX]
			     [NDIM * NDIM],
			     float
			     kappa_and_mu[actual_size][NGLLZ][NGLLY][NGLLX]
			     [2],
			     float
			     sum_terms[actual_size][NGLLZ][NGLLY][NGLLX]
			     [NDIM]);

static void update_disp_vel (int actual_size,
			     float displ[actual_size][NDIM],
			     float veloc[actual_size][NDIM],
			     float accel[actual_size][NDIM]);

static void update_acc_vel (int actual_size,
			    float accel[actual_size][NDIM],
			    float veloc[actual_size][NDIM],
			    float rmass_inverse[actual_size]);

static void compute_max (int actual_size,
			 float displ[NGLOB][NDIM], float *max);

extern void kern_clear (int actual_size, float displ[actual_size][NDIM],
			float veloc[actual_size][NDIM],
			float accel[actual_size][NDIM]);

extern void kern_gather (int actual_size,
			 float displ[NGLOB][NDIM],
			 int ibool[actual_size][NGLLZ][NGLLY][NGLLX],
			 float
			 dummy_loc[actual_size][NGLLZ][NGLLY][NGLLX][NDIM]);

extern void kern_scatter (int actual_size,
			  int ibool[actual_size][NGLLZ][NGLLY][NGLLX],
			  float
			  sum_terms[actual_size][NGLLZ][NGLLY][NGLLX][NDIM],
			  float accel[NGLOB][NDIM]);

extern void kern_process_element (int actual_size,
				  float
				  dummy_loc[actual_size][NGLLZ][NGLLY][NGLLX]
				  [NDIM],
#ifdef USE_DEVILLE_INLINED_PRODUCTS
				  float all_matrices[NDIM + 4][NGLLX][NGLLX],
#else
				  float hprime_xx[NGLLX][NGLLX],
				  float hprimewgll_xx[NGLLX][NGLLX],
				  float wgllwgll_all[NGLLZ][NGLLY][NDIM],
#endif
				  float
				  jacobian_matrix[actual_size][NGLLZ][NGLLY]
				  [NGLLX][NDIM * NDIM],
				  float
				  kappa_and_mu[actual_size][NGLLZ][NGLLY]
				  [NGLLX][2],
				  float
				  sum_terms[actual_size][NGLLZ][NGLLY][NGLLX]
				  [NDIM]);

extern void kern_update_disp_vel (int actual_size,
				  float displ[actual_size][NDIM],
				  float veloc[actual_size][NDIM],
				  float accel[actual_size][NDIM]);

extern void kern_update_acc_vel (int actual_size,
				 float accel[actual_size][NDIM],
				 float veloc[actual_size][NDIM],
				 float rmass_inverse[actual_size]);

extern void kern_compute_max (int actual_size,
			      float displ[NGLOB][NDIM], float *max);

extern unsigned long long kern_get (int i);

////////////////////////////////////////////////////////////////////////
//                                   TASKS                            //
////////////////////////////////////////////////////////////////////////


#if 0
#pragma omp task output(**displ, **veloc, **accel)
#endif
void
clear (int actual_size, float displ[actual_size][NDIM],
       float veloc[actual_size][NDIM], float accel[actual_size][NDIM])
{
  kern_clear (actual_size, displ, veloc, accel);
}


////////////////////////////////////////////////////////////////
// Updates
///////////////////////////////////////////////////////////////

// update the displacement and velocity vectors and clear the acceleration
// vector for the assinged chunck

//#pragma omp task inout(**displ, **veloc, **accel)
void update_disp_vel (int actual_size,
		 float displ[actual_size][NDIM],
		 float veloc[actual_size][NDIM],
		 float accel[actual_size][NDIM])
{
  kern_update_disp_vel (actual_size, displ, veloc, accel);
}

// update the acceleration and velocity vectors on assigned chunk of points

//#pragma omp task input(rmass_inverse) inout(**accel,**veloc)
void update_acc_vel (int actual_size,
		float accel[actual_size][NDIM],
		float veloc[actual_size][NDIM],
		float rmass_inverse[actual_size])
{
  kern_update_acc_vel (actual_size, accel, veloc, rmass_inverse);
}

/////////////////////////////////////////////////////////////////////
//   gather - scatter
/////////////////////////////////////////////////////////////////////


// localize data for the element from the global vectors to the local mesh
// using indirect addressing (contained in array ibool)

//#pragma omp task input(**displ, ****ibool) output(*****dummy_loc)
void gather (int actual_size,
	float displ[NGLOB][NDIM],
	int ibool[actual_size][NGLLZ][NGLLY][NGLLX],
	float dummy_loc[actual_size][NGLLZ][NGLLY][NGLLX][NDIM])
{
  kern_gather (actual_size, displ, ibool, dummy_loc);
}

// scattered update must be atomic because we did not use mesh coloring to make
// mesh subsets independent. We could use mesh coloring instead, as in Figure 6 of
// Dimitri Komatitsch, David Michea and Gordon Erlebacher,
// Porting a high-order finite-element earthquake modeling application to NVIDIA graphics cards using CUDA,
// Journal of Parallel and Distributed Computing,
// vol. 69(5), p. 451-460, doi: 10.1016/j.jpdc.2009.01.006 (2009).
// http://www.univ-pau.fr/~dkomati1/published_papers/GPGPU_JPDC_2009.pdf

// sum contributions from the element to the global mesh using indirect addressing
// Sequentiality imposed by dependence on whole accel

//#if defined( TEST_CONCURRENT )
//#pragma omp task input(*****sum_terms,****ibool) concurrent(**accel)
//#elif defined( TEST_INOUT )
//#pragma omp task input(*****sum_terms,****ibool) inout(**accel)
//#elif defined( TEST_COMMUTATIVE )
//#pragma omp task input(*****sum_terms,****ibool) commutative(**accel)
//#else
//#error Define TEST_CONCURRENT, TEST_INOUT, or TEST_COMMUTATIVE
//#endif
void scatter (int actual_size,
	 int ibool[actual_size][NGLLZ][NGLLY][NGLLX],
	 float sum_terms[actual_size][NGLLZ][NGLLY][NGLLX][NDIM],
	 float accel[NGLOB][NDIM])
{
  kern_scatter (actual_size, ibool, sum_terms, accel);
}

////////////////////////////////////////////////////////////////////
//      Element
///////////////////////////////////////////////////////////////////
#ifdef USE_DEVILLE_INLINED_PRODUCTS
//#pragma omp task input (*****dummy_loc, ***all_matrices, *****jacobian_matrix, *****kappa_and_mu) output (*****sum_terms)
void process_element (int actual_size,
		 float dummy_loc[actual_size][NGLLZ][NGLLY][NGLLX][NDIM],
		 float all_matrices[NDIM + 4][NGLLX][NGLLX],
		 float jacobian_matrix[actual_size][NGLLZ][NGLLY][NGLLX][NDIM * NDIM],
		 float kappa_and_mu[actual_size][NGLLZ][NGLLY][NGLLX][2],
		 float sum_terms[actual_size][NGLLZ][NGLLY][NGLLX][NDIM])
{

  kern_process_element (actual_size, dummy_loc, all_matrices, jacobian_matrix,
			kappa_and_mu, sum_terms);
}
#else // of USE_DEVILLE_INLINED_PRODUCTS
//#pragma omp task input (actual_size,dummy_loc,hprime_xx,hprimewgll_xx,wgllwgll_all,jacobian_matrix,kappa_and_mu) output (sum_terms)
void process_element (int actual_size,
		 float dummy_loc[actual_size][NGLLZ][NGLLY][NGLLX][NDIM],
		 float hprime_xx[NGLLX][NGLLX],
		 float hprimewgll_xx[NGLLX][NGLLX],
		 float wgllwgll_all[NGLLZ][NGLLY][NDIM],
		 float jacobian_matrix[actual_size][NGLLZ][NGLLY][NGLLX][NDIM
									 *
									 NDIM],
		 float kappa_and_mu[actual_size][NGLLZ][NGLLY][NGLLX][2],
		 float sum_terms[actual_size][NGLLZ][NGLLY][NGLLX][NDIM])
{
}
#endif // of USE_DEVILLE_INLINED_PRODUCTS

//#pragma omp task input( displ ) inout(*max) 
void compute_max (int actual_size, float displ[actual_size][NDIM], float *max)
{
  kern_compute_max (actual_size, displ, max);
}

////////////////////////////////////////////////////////////////////////
//                               END   TASKS                          //
////////////////////////////////////////////////////////////////////////

int
main (int argc, char *argv[])
{
  int ret;

// global displacement, velocity and acceleration vectors
  static float displ[NGLOB][NDIM];
  static float veloc[NGLOB][NDIM];
  static float accel[NGLOB][NDIM];

// global diagonal mass matrix
  static float rmass_inverse[NGLOB];

// record a seismogram to check that the simulation went well
  static float seismogram[NSTEP];

// arrays with mesh parameters per slice
  static int ibool[NSPEC][NGLLZ][NGLLY][NGLLX];

  static float jacobian_matrix[NSPEC][NGLLZ][NGLLY][NGLLX][NDIM * NDIM];

  static float kappa_and_mu[NSPEC][NGLLZ][NGLLY][NGLLX][2];

  static float dummy_loc[NSPEC / BS_NSPEC][BS_NSPEC][NGLLZ][NGLLY][NGLLX][NDIM] __attribute__ ((aligned (16)));

// array with derivatives of Lagrange polynomials and precalculated products
  static float hprime_xx[NGLLX][NGLLX];
  static float hprimewgll_xx[NGLLX][NGLLX];
  static float wgllwgll_all[NGLLY][NGLLX][NDIM];
#ifdef USE_DEVILLE_INLINED_PRODUCTS
  static float hprime_xxT[NGLLX][NGLLX];
  static float hprimewgll_xxT[NGLLX][NGLLX];
  static float all_matrices[NDIM + 4][NGLLX][NGLLX];
#endif

  static float sum_terms[NSPEC / BS_NSPEC][BS_NSPEC][NGLLZ][NGLLY][NGLLX][NDIM];

// time step
  int it;

  int ispec, i, j, k, iglob_source;

// blocking controls
  int actual_size;

  float Usolidnorm, time, memory_size;

  int i_map_ibool;

// to read external files
  FILE *IIN;

#if defined(CONFIG_VERBOSE)
  printf ("\nNSPEC = %d\n", NSPEC);
  printf ("NGLOB = %d\n\n", NGLOB);
  printf ("NSTEP = %d\n", NSTEP);
  printf ("deltat = %f\n\n", deltat);
#endif

// make sure that we can use the Deville et al. (2002) inlined products
#ifdef USE_DEVILLE_INLINED_PRODUCTS
  if (NGLLX != 5 || NGLLY != 5 || NGLLZ != 5)
    {
      fprintf (stderr,
	       "we must have NGLLX = NGLLY = NGLLZ = 5 to be able to use the Deville et al. (2002) inlined products, exiting...\n");
      exit (1);
    }
#endif

// estimate total memory size (the size of a real number is 4 bytes)
// we perform the calculation in single precision rather than integer
// to avoid integer overflow in the case of very large meshes
#if defined(CONFIG_VERBOSE)
  memory_size = 4.f * ((3.f * NDIM + 1.f) * NGLOB + 13.f * (float) (NGLLX * NGLLY * NGLLZ) * (float) (NSPEC));
  printf ("approximate total memory size used = %f Mb\n\n", memory_size / 1024.f / 1024.f);
#endif


// read the mesh from external file
#ifndef USE_PATHS_ROSA
  printf ("reading file ../multi_GPU_MPI/DATABASES_FOR_SOLVER/proc000000_reg1_database.dat\n");
  if ((IIN = fopen ("../multi_GPU_MPI/DATABASES_FOR_SOLVER/proc000000_reg1_database.dat", "r")) == NULL)
    {
#else
#if defined(CONFIG_VERBOSE)
  printf ("reading file ./DB/proc000000_reg1_database.dat\n");
#endif
  if ((IIN = fopen ("./DB/proc000000_reg1_database.dat", "r")) == NULL) {
#endif
      fprintf (stderr, "Cannot open file ../multi_GPU_MPI/DATABASES_FOR_SOLVER/proc000000_reg1_database.dat, exiting...\n");
      exit (1);
    }

  for (ispec = 0; ispec < NSPEC; ispec++)
    {
      for (k = 0; k < NGLLZ; k++)
	{
	  for (j = 0; j < NGLLY; j++)
	    {
	      for (i = 0; i < NGLLX; i++)
		{
// read real numbers here
		  ret= fscanf (IIN, "%e\n", &jacobian_matrix[ispec][k][j][i][XIX]);
		  ret= fscanf (IIN, "%e\n", &jacobian_matrix[ispec][k][j][i][XIY]);
		  ret= fscanf (IIN, "%e\n", &jacobian_matrix[ispec][k][j][i][XIZ]);
		  ret= fscanf (IIN, "%e\n", &jacobian_matrix[ispec][k][j][i][ETAX]);
		  ret= fscanf (IIN, "%e\n", &jacobian_matrix[ispec][k][j][i][ETAY]);
		  ret= fscanf (IIN, "%e\n", &jacobian_matrix[ispec][k][j][i][ETAZ]);
		  ret= fscanf (IIN, "%e\n", &jacobian_matrix[ispec][k][j][i][GAMMAX]);
		  ret= fscanf (IIN, "%e\n", &jacobian_matrix[ispec][k][j][i][GAMMAY]);
		  ret= fscanf (IIN, "%e\n", &jacobian_matrix[ispec][k][j][i][GAMMAZ]);
		  ret= fscanf (IIN, "%e\n", &kappa_and_mu[ispec][k][j][i][KAPPA]);
		  ret= fscanf (IIN, "%e\n", &kappa_and_mu[ispec][k][j][i][MU]);

// read an integer here
		  ret= fscanf (IIN, "%d\n", &ibool[ispec][k][j][i]);
// subtract one because indices start at zero in C but this array was created by a Fortran
// program and therefore starts at one in the file stored on the disk
		  ibool[ispec][k][j][i]--;
		}
	    }
	}
    }
  for (i = 0; i < NGLOB; i++)
    {
      ret = fscanf (IIN, "%e\n", &rmass_inverse[i]);
// the real exactly diagonal mass matrix is read (not its inverse)
// therefore invert it here once and for all
      rmass_inverse[i] = 1.f / rmass_inverse[i];
    }
  fclose (IIN);

// read the derivation matrices from external file
#ifndef USE_PATHS_ROSA
  printf ("reading file ../multi_GPU_MPI/DATABASES_FOR_SOLVER/matrices.dat\n");
  if ((IIN = fopen ("../multi_GPU_MPI/DATABASES_FOR_SOLVER/matrices.dat", "r")) == NULL) {
#else
#if defined(CONFIG_VERBOSE)
  printf ("reading file ./DB/matrices.dat\n");
#endif
  if ((IIN = fopen ("./DB/matrices.dat", "r")) == NULL)
    {
#endif
      fprintf (stderr,
	       "Cannot open file ../multi_GPU_MPI/DATABASES_FOR_SOLVER/matrices.dat, exiting...\n");
      exit (1);
    }

  for (j = 0; j < NGLLY; j++)
    {
      for (i = 0; i < NGLLX; i++)
	{
	  ret= fscanf (IIN, "%e\n", &hprime_xx[j][i]);
	  ret= fscanf (IIN, "%e\n", &hprimewgll_xx[j][i]);
	  ret= fscanf (IIN, "%e\n", &wgllwgll_all[j][i][YZ]);
	  ret= fscanf (IIN, "%e\n", &wgllwgll_all[j][i][XZ]);
	  ret= fscanf (IIN, "%e\n", &wgllwgll_all[j][i][XY]);

#ifdef USE_DEVILLE_INLINED_PRODUCTS
// DK DK also store the transpose matrices
	  hprime_xxT[i][j] = hprime_xx[j][i];
	  hprimewgll_xxT[i][j] = hprimewgll_xx[j][i];
#endif

	}
    }

// DK DK store all the matrices in one array in order to reduce the number of arguments sent to the tasks
#ifdef USE_DEVILLE_INLINED_PRODUCTS
#pragma omp parallel for
  for (j = 0; j < NGLLY; j++)
    {
      for (i = 0; i < NGLLX; i++)
	{
	  all_matrices[FLAG_hprime_xx][j][i] = hprime_xx[j][i];
	  all_matrices[FLAG_hprime_xxT][j][i] = hprime_xxT[j][i];

	  all_matrices[FLAG_hprimewgll_xx][j][i] = hprimewgll_xx[j][i];
	  all_matrices[FLAG_hprimewgll_xxT][j][i] = hprimewgll_xxT[j][i];

	  all_matrices[FLAG_wgllwgll_YZ][j][i] = wgllwgll_all[j][i][YZ];
	  all_matrices[FLAG_wgllwgll_XZ][j][i] = wgllwgll_all[j][i][XZ];
	  all_matrices[FLAG_wgllwgll_XY][j][i] = wgllwgll_all[j][i][XY];
	}
    }
#endif

  fclose (IIN);




// clear initial vectors before starting the time loop
// it is not really crucial to parallelize this task because it is done only once
// before entering the time loop therefore it could remain serial
#pragma omp parallel for
  for (i = 0; i < NGLOB; i += BS_NGLOB)
    {
      actual_size = ((NGLOB - i) >= BS_NGLOB ? BS_NGLOB : (NGLOB - i));
      clear (actual_size, (void *) &displ[i][0], (void *) &veloc[i][0], (void *) &accel[i][0]);
    }

#if defined(CONFIG_VERBOSE)
  printf ("\nstarting the time loop\n\n");
#endif
  t_start = usecs ();

#pragma omp parallel
#pragma omp single
  {
// start of the time loop (which must remain serial obviously)
    for (it = 1; it <= NSTEP; it++) {
// compute maximum of norm of displacement from time to time and display it
// in order to monitor the simulation
// this can remain serial because it is done only every NTSTEP_BETWEEN_OUTPUT_INFO time steps
	if ((it % NTSTEP_BETWEEN_OUTPUT_INFO) == 0 || it == 5 || it == NSTEP) {
	    Usolidnorm = -1.f;
	    for (i = 0; i < NGLOB; i += BS_NGLOB) {
		actual_size = ((NGLOB - i) >= BS_NGLOB ? BS_NGLOB : (NGLOB - i));
#pragma omp task firstprivate(actual_size, i, bs_nglob, dim) depend(inout: Usolidnorm) depend(in: displ[0:bs_nglob][0:dim])
		compute_max (actual_size, (void *) &displ[i][0], &Usolidnorm);
	      }
//#pragma css wait on (&Usolidnorm)
#pragma omp taskwait
#if defined(CONFIG_VERBOSE)
	    printf ("Time step # %d out of %d\n", it, NSTEP);
	    printf ("Max norm displacement vector U in the solid (m) = %.8g\n\n", Usolidnorm);
#endif
// check stability of the code, exit if unstable
	    if (Usolidnorm > STABILITY_THRESHOLD) {
		fprintf (stderr, "code became unstable and blew up\n");
		exit (1);
	      }
	  }

// big loop over all the global points (not elements) in the mesh to update
// the displacement and velocity vectors and clear the acceleration vector
	for (i = 0; i < NGLOB; i += BS_NGLOB) {
	    actual_size = ((NGLOB - i) >= BS_NGLOB ? BS_NGLOB : (NGLOB - i));
#pragma omp task firstprivate(actual_size, i, bs_nglob, dim) depend(inout: displ[0:bs_nglob][0:dim], veloc[0:bs_nglob][0:dim], accel[0:bs_nglob][0:dim])
	    update_disp_vel (actual_size, (void *) &displ[i][0], (void *) &veloc[i][0], (void *) &accel[i][0]);
	  }

// big loop over all the elements in the mesh to localize data once and for all
// from the global vectors to the local mesh
// using indirect addressing (contained in array ibool)

// big loop over all the elements in the mesh to compute the elemental contribution
// to the acceleration vector of each element of the finite-element mesh

// need to ensure ALL displ has been produced.
// Limitation of current dependence detection mechanism.
// this is waiting for displ[] to be entirely filled because we use it in the gather below.
#pragma omp taskwait
//This barrier is need because the data is accessed with a different association
	for (ispec = 0; ispec < NSPEC; ispec += BS_NSPEC) {
	    actual_size = ((NSPEC - ispec) >= BS_NSPEC ? BS_NSPEC : (NSPEC - ispec));
#pragma omp task firstprivate(actual_size, ispec, bs_nspec, dim) depend(out: dummy_loc[0:actual_size][:dim_z][:dim_y][:dim_x]) depend(in: displ[0:bs_nspec][0:dim], ibool[0:actual_size][:dim_z][:dim_y][:dim_x])
	    gather (actual_size, displ, (void *) &ibool[ispec][0][0][0], dummy_loc[ispec / BS_NSPEC]);

#pragma omp task firstprivate(actual_size, ispec, bs_nspec, dim) depend(in: dummy_loc[:actual_size][:dim_z][:dim_y][:dim_x][:dim], all_matrices[:dim+4][:dim_x][:dim_x], jacobian_matrix[:actual_size][:dim_z][:dim_y][:dim_x][:dim*dim], kappa_and_mu[:actual_size][:dim_z][:dim_y][:dim_x][:2]) depend(out: sum_terms[:actual_size][:dim_z][:dim_y][:dim_x][:dim])
	    process_element (actual_size, dummy_loc[ispec / BS_NSPEC],
#ifdef USE_DEVILLE_INLINED_PRODUCTS
			     all_matrices,
#else
			     hprime_xx, hprimewgll_xx, wgllwgll_all,
#endif
			     (void *) &jacobian_matrix[ispec][0][0][0][0],
			     (void *) &kappa_and_mu[ispec][0][0][0][0],
			     sum_terms[ispec / BS_NSPEC]);

// sum contributions from each element to the global mesh using indirect addressing
#pragma omp task firstprivate(actual_size, ispec, bs_nspec, dim) depend(inout: accel[:nglob][:dim]) depend(in: ibool[:actual_size][:dim_z][:dim_y][:dim_x], sum_terms[:actual_size][:dim_z][:dim_y][:dim_x][:dim])
	    scatter (actual_size, (void *) &ibool[ispec][0][0][0], sum_terms[ispec / BS_NSPEC], accel);

	  } // end of main loop on all the elements

#pragma omp taskwait

// add the earthquake source at a given grid point
// this is negligible and is intrinsically serial because it is done by only
// one grid point out of several millions typically
// we subtract one to the element number of the source because arrays start at 0 in C
// compute current time
	time = (it - 1) * deltat;
	iglob_source = ibool[NSPEC_SOURCE - 1][1][1][1];
// we divide the amplitude of the source by rmass_inverse[iglob_source] here because
// we have merged the calculation of acceleration and velocity below in a single task
// and therefore the value of accel[] at that point will be
// multiplied by rmass_inverse[i] in that merged task
	accel[iglob_source][Z] += 1.e4f * (1.f - 2.f * a * (time - t0) * (time - t0)) * expf (-a * (time - t0) *
			(time - t0)) / (rho * rmass_inverse [iglob_source]);

// big loop over all the global points (not elements) in the mesh to update
// the acceleration and velocity vectors
	for (i = 0; i < NGLOB; i += BS_NGLOB) {
	    actual_size = ((NGLOB - i) >= BS_NGLOB ? BS_NGLOB : (NGLOB - i));
#pragma omp task firstprivate(actual_size, i, bs_nglob, dim) depend(inout: accel[:actual_size][:dim], veloc[:actual_size][:dim]) depend(in: rmass_inverse[:actual_size])
	    update_acc_vel (actual_size, (void *) &accel[i][0], (void *) &veloc[i][0],
			    (void *) &rmass_inverse[i]);
	  }

#pragma omp taskwait
// record a seismogram to check that the simulation went well
// we subtract one to the element number of the receiver because arrays start at 0 in C
	seismogram[it - 1] = displ[ibool[NSPEC_STATION - 1][1][1][1]][Z];

      }				// end of the serial time loop
#pragma omp taskwait
  }				// end omp parallel
  t_end = usecs ();

#if defined(CONFIG_BENCHMARK)
#if defined(_OPENMP)
  /* threads NSPEC NGLOB time */
  fprintf(stdout, "specfem3d;omp;%d;%d;%d;%d;%d;%d;%f;%.6f\n", omp_get_max_threads(), NSPEC, NGLOB, BS_NSPEC, BS_NGLOB, NSTEP, deltat, 
  	 (float) (t_end - t_start) / 1000000.f);
#else /* _OPENMP */
  fprintf(stdout, "specfem3d;serial;%d;%d;%d;%d;%d;%d;%f;%.6f\n", 1, NSPEC, NGLOB, 1, 1, NSTEP, deltat, 
  	 (float) (t_end - t_start) / 1000000.f);
#endif
#else /* CONFIG_BENCHMARK */
  printf ("elapsed time: %f seconds\n",
	  (float) (t_end - t_start) / 1000000.f);

// save the seismogram at the end of the run0
  char filename[50];
  sprintf (filename, "seismogram_omp_task_deps_%d.txt", getpid ());
  if ((IIN = fopen (filename, "w")) == NULL)
    {
      fprintf (stderr, "Cannot open file %s, exiting...\n", filename);
      exit (1);
    }
  for (it = 0; it < NSTEP; it++)
    {
      fprintf (IIN, "%e %e\n", it * deltat, seismogram[it]);
    }
  fclose (IIN);
#endif

#if defined(CONFIG_VERBOSE)
  printf ("\nEnd of the program\n");
#endif
#if 0
  printf ("### ");
  for (int i = 0; i < 13; ++i)
    printf ("%llu ", kern_get (i));
  printf ("\n");
#endif
}

//
// function to measure time
//

long usecs () {
  struct timeval t;

  gettimeofday (&t, NULL);
  return t.tv_sec * 1000000 + t.tv_usec;
}
