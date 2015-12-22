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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

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

long t_start, t_end;

// declare all the functions

#if 0
static inline unsigned long long
getTimeStart ()
{
  unsigned int hi, lo;

  __asm__ __volatile__ ("cpuid\n"
			"rdtsc\n":"=a" (lo), "=d" (hi)::"%rbx", "%rcx");
  return ((unsigned long long) lo) | (((unsigned long long) hi) << 32);
}

static inline unsigned long long
getTimeStop ()
{
  unsigned int hi, lo;

  __asm__ __volatile__ ("rdtscp\n"
			"mov %%edx, %0\n\t"
			"mov %%eax, %1\n\t"
			"cpuid\n":"=r" (hi), "=r" (lo)::"%rax", "%rbx",
			"%rcx", "%rdx");
  return ((unsigned long long) lo) | (((unsigned long long) hi) << 32);
}


static unsigned long long total[16] = { 0 };

extern void
kern_add (int i, unsigned long long value)
{
  __sync_add_and_fetch (&total[i], value);
}

extern unsigned long long
kern_get (int i)
{
  return __sync_add_and_fetch (&total[i], 0);
}
#endif

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


////////////////////////////////////////////////////////////////////////
//                                   TASKS                            //
////////////////////////////////////////////////////////////////////////


void
kern_clear (int actual_size, float displ[actual_size][NDIM],
	    float veloc[actual_size][NDIM], float accel[actual_size][NDIM])
{
  int i;

  for (i = 0; i < actual_size; i++)
    {
      displ[i][X] = 0.f;
      displ[i][Y] = 0.f;
      displ[i][Z] = 0.f;

      veloc[i][X] = 0.f;
      veloc[i][Y] = 0.f;
      veloc[i][Z] = 0.f;

      accel[i][X] = 0.f;
      accel[i][Y] = 0.f;
      accel[i][Z] = 0.f;
    }

}


////////////////////////////////////////////////////////////////
// Updates
///////////////////////////////////////////////////////////////

// update the displacement and velocity vectors and clear the acceleration
// vector for the assinged chunck

void
kern_update_disp_vel (int actual_size,
		      float displ[actual_size][NDIM],
		      float veloc[actual_size][NDIM],
		      float accel[actual_size][NDIM])
{
  int i;

// DK DK we CANNOT define this with the [NDIM] index first, as would be natural in C
// DK DK in order to have the fastest index [i] on the right, because these arrays
// DK DK are defined as accel[actual_size][NDIM] in the tasks but as accel[NGLOB][NDIM]
// DK DK in other routines and therefore the memory chunks would not correspond.
// DK DK In other words, in any array with a dimension [actual_size] that dimension must
// DK DK always be the leftmost index.
  for (i = 0; i < actual_size; i++)
    {
      displ[i][X] += deltat * veloc[i][X] + deltatsqover2 * accel[i][X];
      displ[i][Y] += deltat * veloc[i][Y] + deltatsqover2 * accel[i][Y];
      displ[i][Z] += deltat * veloc[i][Z] + deltatsqover2 * accel[i][Z];

      veloc[i][X] += deltatover2 * accel[i][X];
      veloc[i][Y] += deltatover2 * accel[i][Y];
      veloc[i][Z] += deltatover2 * accel[i][Z];

      accel[i][X] = 0.f;
      accel[i][Y] = 0.f;
      accel[i][Z] = 0.f;
    }
}

// update the acceleration and velocity vectors on assigned chunk of points

void
kern_update_acc_vel (int actual_size,
		     float accel[actual_size][NDIM],
		     float veloc[actual_size][NDIM],
		     float rmass_inverse[actual_size])
{
  int i;

  for (i = 0; i < actual_size; i++)
    {
      accel[i][X] *= rmass_inverse[i];
      accel[i][Y] *= rmass_inverse[i];
      accel[i][Z] *= rmass_inverse[i];

      veloc[i][X] += deltatover2 * accel[i][X];
      veloc[i][Y] += deltatover2 * accel[i][Y];
      veloc[i][Z] += deltatover2 * accel[i][Z];
    }
}

/////////////////////////////////////////////////////////////////////
//   gather - scatter
/////////////////////////////////////////////////////////////////////


// localize data for the element from the global vectors to the local mesh
// using indirect addressing (contained in array ibool)

void
kern_gather (int actual_size,
	     float displ[NGLOB][NDIM],
	     int ibool[actual_size][NGLLZ][NGLLY][NGLLX],
	     float dummy_loc[actual_size][NGLLZ][NGLLY][NGLLX][NDIM])
{
  int i, j, k, iglob, elem;

  for (elem = 0; elem < actual_size; elem++)
    {

      for (k = 0; k < NGLLZ; k++)
	for (j = 0; j < NGLLY; j++)
	  for (i = 0; i < NGLLX; i++)
	    {
	      iglob = ibool[elem][k][j][i];
	      dummy_loc[elem][k][j][i][X] = displ[iglob][X];
	      dummy_loc[elem][k][j][i][Y] = displ[iglob][Y];
	      dummy_loc[elem][k][j][i][Z] = displ[iglob][Z];
	    }

    }
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

void
kern_scatter (int actual_size,
	      int ibool[actual_size][NGLLZ][NGLLY][NGLLX],
	      float sum_terms[actual_size][NGLLZ][NGLLY][NGLLX][NDIM],
	      float accel[NGLOB][NDIM])
{
  int i, j, k, iglob, elem;

  for (elem = 0; elem < actual_size; elem++)
    {
      for (k = 0; k < NGLLZ; k++)
	{
	  for (j = 0; j < NGLLY; j++)
	    {
	      for (i = 0; i < NGLLX; i++)
		{
		  iglob = ibool[elem][k][j][i];
//		  double val =
//		    fabs (sum_terms[elem][k][j][i][X]) +
//		    fabs (sum_terms[elem][k][j][i][Y]) +
//		    fabs (sum_terms[elem][k][j][i][Z]);
#if defined(_OPENMP)
#pragma omp atomic
#endif
		  accel[iglob][X] += sum_terms[elem][k][j][i][X];
#if defined(_OPENMP)
#pragma omp atomic
#endif
		  accel[iglob][Y] += sum_terms[elem][k][j][i][Y];
#if defined(_OPENMP)
#pragma omp atomic
#endif
		  accel[iglob][Z] += sum_terms[elem][k][j][i][Z];
		}
	    }
	}

    }
}

////////////////////////////////////////////////////////////////////
//      Element
///////////////////////////////////////////////////////////////////

#ifdef USE_DEVILLE_INLINED_PRODUCTS
void
kern_process_element (int actual_size,
		      float dummy_loc[actual_size][NGLLZ][NGLLY][NGLLX][NDIM],
		      float all_matrices[NDIM + 4][NGLLX][NGLLX],
		      float
		      jacobian_matrix[actual_size][NGLLZ][NGLLY][NGLLX][NDIM *
									NDIM],
		      float kappa_and_mu[actual_size][NGLLZ][NGLLY][NGLLX][2],
		      float sum_terms[actual_size][NGLLZ][NGLLY][NGLLX][NDIM])
{

  float tempx1[NGLLZ][NGLLY][NGLLX];
  float tempx2[NGLLZ][NGLLY][NGLLX];
  float tempx3[NGLLZ][NGLLY][NGLLX];
  float tempy1[NGLLZ][NGLLY][NGLLX];
  float tempy2[NGLLZ][NGLLY][NGLLX];
  float tempy3[NGLLZ][NGLLY][NGLLX];
  float tempz1[NGLLZ][NGLLY][NGLLX];
  float tempz2[NGLLZ][NGLLY][NGLLX];
  float tempz3[NGLLZ][NGLLY][NGLLX];

  float newtempx1[NGLLZ][NGLLY][NGLLX];
  float newtempx2[NGLLZ][NGLLY][NGLLX];
  float newtempx3[NGLLZ][NGLLY][NGLLX];
  float newtempy1[NGLLZ][NGLLY][NGLLX];
  float newtempy2[NGLLZ][NGLLY][NGLLX];
  float newtempy3[NGLLZ][NGLLY][NGLLX];
  float newtempz1[NGLLZ][NGLLY][NGLLX];
  float newtempz2[NGLLZ][NGLLY][NGLLX];
  float newtempz3[NGLLZ][NGLLY][NGLLX];

  int i, j, k, elem;
  float xixl, xiyl, xizl, etaxl, etayl, etazl, gammaxl, gammayl, gammazl,
    jacobianl;
  float duxdxl, duxdyl, duxdzl, duydxl, duydyl, duydzl, duzdxl, duzdyl,
    duzdzl;
  float duxdxl_plus_duydyl, duxdxl_plus_duzdzl, duydyl_plus_duzdzl;
  float duxdyl_plus_duydxl, duzdxl_plus_duxdzl, duzdyl_plus_duydzl;
  float sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz;
  float fac1, fac2, fac3, lambdal, mul, lambdalplus2mul, kappal;

// DK DK pointers to simulate a "union" statement by pointing to the same memory block
  float *tempx1_2D_25_5, *tempy1_2D_25_5, *tempz1_2D_25_5;
  float *tempx3_2D_5_25, *tempy3_2D_5_25, *tempz3_2D_5_25;
  float *newtempx1_2D_25_5, *newtempy1_2D_25_5, *newtempz1_2D_25_5;
  float *newtempx3_2D_5_25, *newtempy3_2D_5_25, *newtempz3_2D_5_25;
  float *dummyx_loc_2D, *dummyy_loc_2D, *dummyz_loc_2D;

  unsigned long long start, stop;

// DK DK mapping for 2D arrays of size [25][5] or [5][25] to an offset in a 1D linear memory block
#define map_25_5(i,j)   ((j) +  5*(i))
#define map_5_25(i,j)   ((j) + 25*(i))

// DK DK assign the pointers to the common memory block
  tempx1_2D_25_5 = &tempx1[0][0][0];
  tempy1_2D_25_5 = &tempy1[0][0][0];
  tempz1_2D_25_5 = &tempz1[0][0][0];

  tempx3_2D_5_25 = &tempx3[0][0][0];
  tempy3_2D_5_25 = &tempy3[0][0][0];
  tempz3_2D_5_25 = &tempz3[0][0][0];

  newtempx1_2D_25_5 = &newtempx1[0][0][0];
  newtempy1_2D_25_5 = &newtempy1[0][0][0];
  newtempz1_2D_25_5 = &newtempz1[0][0][0];

  newtempx3_2D_5_25 = &newtempx3[0][0][0];
  newtempy3_2D_5_25 = &newtempy3[0][0][0];
  newtempz3_2D_5_25 = &newtempz3[0][0][0];

  for (elem = 0; elem < actual_size; elem++)
    {

// DK DK assign the pointers to the common memory block
      dummyx_loc_2D = &dummy_loc[elem][0][0][0][X];
      dummyy_loc_2D = &dummy_loc[elem][0][0][0][Y];
      dummyz_loc_2D = &dummy_loc[elem][0][0][0][Z];

// subroutines adapted from Deville, Fischer and Mund, High-order methods
// for incompressible fluid flow, Cambridge University Press (2002),
// pages 386 and 389 and Figure 8.3.1
      for (i = 0; i < NGLLX; i++)
	{
	  for (j = 0; j < NGLL2; j++)
	    {
	      tempx1_2D_25_5[map_25_5 (j, i)] =
		all_matrices[FLAG_hprime_xx][0][i] *
		dummyx_loc_2D[map_25_5 (j, 0) * 3] +
		all_matrices[FLAG_hprime_xx][1][i] *
		dummyx_loc_2D[map_25_5 (j, 1) * 3] +
		all_matrices[FLAG_hprime_xx][2][i] *
		dummyx_loc_2D[map_25_5 (j, 2) * 3] +
		all_matrices[FLAG_hprime_xx][3][i] *
		dummyx_loc_2D[map_25_5 (j, 3) * 3] +
		all_matrices[FLAG_hprime_xx][4][i] *
		dummyx_loc_2D[map_25_5 (j, 4) * 3];

	      tempy1_2D_25_5[map_25_5 (j, i)] =
		all_matrices[FLAG_hprime_xx][0][i] *
		dummyy_loc_2D[map_25_5 (j, 0) * 3] +
		all_matrices[FLAG_hprime_xx][1][i] *
		dummyy_loc_2D[map_25_5 (j, 1) * 3] +
		all_matrices[FLAG_hprime_xx][2][i] *
		dummyy_loc_2D[map_25_5 (j, 2) * 3] +
		all_matrices[FLAG_hprime_xx][3][i] *
		dummyy_loc_2D[map_25_5 (j, 3) * 3] +
		all_matrices[FLAG_hprime_xx][4][i] *
		dummyy_loc_2D[map_25_5 (j, 4) * 3];

	      tempz1_2D_25_5[map_25_5 (j, i)] =
		all_matrices[FLAG_hprime_xx][0][i] *
		dummyz_loc_2D[map_25_5 (j, 0) * 3] +
		all_matrices[FLAG_hprime_xx][1][i] *
		dummyz_loc_2D[map_25_5 (j, 1) * 3] +
		all_matrices[FLAG_hprime_xx][2][i] *
		dummyz_loc_2D[map_25_5 (j, 2) * 3] +
		all_matrices[FLAG_hprime_xx][3][i] *
		dummyz_loc_2D[map_25_5 (j, 3) * 3] +
		all_matrices[FLAG_hprime_xx][4][i] *
		dummyz_loc_2D[map_25_5 (j, 4) * 3];
	    }
	}

      for (k = 0; k < NGLLZ; k++)
	{
	  for (j = 0; j < NGLLX; j++)
	    {
	      for (i = 0; i < NGLLX; i++)
		{

		  tempx2[k][j][i] =
		    dummy_loc[elem][k][0][i][X] *
		    all_matrices[FLAG_hprime_xxT][j][0] +
		    dummy_loc[elem][k][1][i][X] *
		    all_matrices[FLAG_hprime_xxT][j][1] +
		    dummy_loc[elem][k][2][i][X] *
		    all_matrices[FLAG_hprime_xxT][j][2] +
		    dummy_loc[elem][k][3][i][X] *
		    all_matrices[FLAG_hprime_xxT][j][3] +
		    dummy_loc[elem][k][4][i][X] *
		    all_matrices[FLAG_hprime_xxT][j][4];

		  tempy2[k][j][i] =
		    dummy_loc[elem][k][0][i][Y] *
		    all_matrices[FLAG_hprime_xxT][j][0] +
		    dummy_loc[elem][k][1][i][Y] *
		    all_matrices[FLAG_hprime_xxT][j][1] +
		    dummy_loc[elem][k][2][i][Y] *
		    all_matrices[FLAG_hprime_xxT][j][2] +
		    dummy_loc[elem][k][3][i][Y] *
		    all_matrices[FLAG_hprime_xxT][j][3] +
		    dummy_loc[elem][k][4][i][Y] *
		    all_matrices[FLAG_hprime_xxT][j][4];

		  tempz2[k][j][i] =
		    dummy_loc[elem][k][0][i][Z] *
		    all_matrices[FLAG_hprime_xxT][j][0] +
		    dummy_loc[elem][k][1][i][Z] *
		    all_matrices[FLAG_hprime_xxT][j][1] +
		    dummy_loc[elem][k][2][i][Z] *
		    all_matrices[FLAG_hprime_xxT][j][2] +
		    dummy_loc[elem][k][3][i][Z] *
		    all_matrices[FLAG_hprime_xxT][j][3] +
		    dummy_loc[elem][k][4][i][Z] *
		    all_matrices[FLAG_hprime_xxT][j][4];

		}
	    }
	}

      for (j = 0; j < NGLLX; j++)
	{
	  for (i = 0; i < NGLL2; i++)
	    {
	      tempx3_2D_5_25[map_5_25 (j, i)] =
		dummyx_loc_2D[map_5_25 (0, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][0] +
		dummyx_loc_2D[map_5_25 (1, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][1] +
		dummyx_loc_2D[map_5_25 (2, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][2] +
		dummyx_loc_2D[map_5_25 (3, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][3] +
		dummyx_loc_2D[map_5_25 (4, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][4];

	      tempy3_2D_5_25[map_5_25 (j, i)] =
		dummyy_loc_2D[map_5_25 (0, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][0] +
		dummyy_loc_2D[map_5_25 (1, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][1] +
		dummyy_loc_2D[map_5_25 (2, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][2] +
		dummyy_loc_2D[map_5_25 (3, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][3] +
		dummyy_loc_2D[map_5_25 (4, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][4];

	      tempz3_2D_5_25[map_5_25 (j, i)] =
		dummyz_loc_2D[map_5_25 (0, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][0] +
		dummyz_loc_2D[map_5_25 (1, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][1] +
		dummyz_loc_2D[map_5_25 (2, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][2] +
		dummyz_loc_2D[map_5_25 (3, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][3] +
		dummyz_loc_2D[map_5_25 (4, i) * 3] *
		all_matrices[FLAG_hprime_xxT][j][4];

	    }
	}

      for (k = 0; k < NGLLZ; k++)
	{
	  for (j = 0; j < NGLLX; j++)
	    {
	      for (i = 0; i < NGLLX; i++)
		{

// compute derivatives of ux, uy and uz with respect to x, y and z
		  xixl = jacobian_matrix[elem][k][j][i][XIX];
		  xiyl = jacobian_matrix[elem][k][j][i][XIY];
		  xizl = jacobian_matrix[elem][k][j][i][XIZ];
		  etaxl = jacobian_matrix[elem][k][j][i][ETAX];
		  etayl = jacobian_matrix[elem][k][j][i][ETAY];
		  etazl = jacobian_matrix[elem][k][j][i][ETAZ];
		  gammaxl = jacobian_matrix[elem][k][j][i][GAMMAX];
		  gammayl = jacobian_matrix[elem][k][j][i][GAMMAY];
		  gammazl = jacobian_matrix[elem][k][j][i][GAMMAZ];
		  jacobianl =
		    1.f / (xixl * (etayl * gammazl - etazl * gammayl) -
			   xiyl * (etaxl * gammazl - etazl * gammaxl) +
			   xizl * (etaxl * gammayl - etayl * gammaxl));

		  duxdxl =
		    xixl * tempx1[k][j][i] + etaxl * tempx2[k][j][i] +
		    gammaxl * tempx3[k][j][i];
		  duxdyl =
		    xiyl * tempx1[k][j][i] + etayl * tempx2[k][j][i] +
		    gammayl * tempx3[k][j][i];
		  duxdzl =
		    xizl * tempx1[k][j][i] + etazl * tempx2[k][j][i] +
		    gammazl * tempx3[k][j][i];

		  duydxl =
		    xixl * tempy1[k][j][i] + etaxl * tempy2[k][j][i] +
		    gammaxl * tempy3[k][j][i];
		  duydyl =
		    xiyl * tempy1[k][j][i] + etayl * tempy2[k][j][i] +
		    gammayl * tempy3[k][j][i];
		  duydzl =
		    xizl * tempy1[k][j][i] + etazl * tempy2[k][j][i] +
		    gammazl * tempy3[k][j][i];

		  duzdxl =
		    xixl * tempz1[k][j][i] + etaxl * tempz2[k][j][i] +
		    gammaxl * tempz3[k][j][i];
		  duzdyl =
		    xiyl * tempz1[k][j][i] + etayl * tempz2[k][j][i] +
		    gammayl * tempz3[k][j][i];
		  duzdzl =
		    xizl * tempz1[k][j][i] + etazl * tempz2[k][j][i] +
		    gammazl * tempz3[k][j][i];

// precompute some sums to save CPU time
		  duxdxl_plus_duydyl = duxdxl + duydyl;
		  duxdxl_plus_duzdzl = duxdxl + duzdzl;
		  duydyl_plus_duzdzl = duydyl + duzdzl;
		  duxdyl_plus_duydxl = duxdyl + duydxl;
		  duzdxl_plus_duxdzl = duzdxl + duxdzl;
		  duzdyl_plus_duydzl = duzdyl + duydzl;

// compute isotropic elements
		  kappal = kappa_and_mu[elem][k][j][i][KAPPA];
		  mul = kappa_and_mu[elem][k][j][i][MU];

		  lambdalplus2mul = kappal + 1.33333333333333333333f * mul;	// 4./3. = 1.3333333
		  lambdal = lambdalplus2mul - 2.f * mul;

// compute stress sigma
		  sigma_xx =
		    lambdalplus2mul * duxdxl + lambdal * duydyl_plus_duzdzl;
		  sigma_yy =
		    lambdalplus2mul * duydyl + lambdal * duxdxl_plus_duzdzl;
		  sigma_zz =
		    lambdalplus2mul * duzdzl + lambdal * duxdxl_plus_duydyl;

		  sigma_xy = mul * duxdyl_plus_duydxl;
		  sigma_xz = mul * duzdxl_plus_duxdzl;
		  sigma_yz = mul * duzdyl_plus_duydzl;

// form dot product with test vector
		  tempx1[k][j][i] =
		    jacobianl * (sigma_xx * xixl + sigma_xy * xiyl +
				 sigma_xz * xizl);
		  tempy1[k][j][i] =
		    jacobianl * (sigma_xy * xixl + sigma_yy * xiyl +
				 sigma_yz * xizl);
		  tempz1[k][j][i] =
		    jacobianl * (sigma_xz * xixl + sigma_yz * xiyl +
				 sigma_zz * xizl);

		  tempx2[k][j][i] =
		    jacobianl * (sigma_xx * etaxl + sigma_xy * etayl +
				 sigma_xz * etazl);
		  tempy2[k][j][i] =
		    jacobianl * (sigma_xy * etaxl + sigma_yy * etayl +
				 sigma_yz * etazl);
		  tempz2[k][j][i] =
		    jacobianl * (sigma_xz * etaxl + sigma_yz * etayl +
				 sigma_zz * etazl);

		  tempx3[k][j][i] =
		    jacobianl * (sigma_xx * gammaxl + sigma_xy * gammayl +
				 sigma_xz * gammazl);
		  tempy3[k][j][i] =
		    jacobianl * (sigma_xy * gammaxl + sigma_yy * gammayl +
				 sigma_yz * gammazl);
		  tempz3[k][j][i] =
		    jacobianl * (sigma_xz * gammaxl + sigma_yz * gammayl +
				 sigma_zz * gammazl);

		}
	    }
	}

      for (j = 0; j < NGLL2; j++)
	{
	  for (i = 0; i < NGLLX; i++)
	    {
	      newtempx1_2D_25_5[map_25_5 (j, i)] =
		all_matrices[FLAG_hprimewgll_xxT][0][i] *
		tempx1_2D_25_5[map_25_5 (j, 0)] +
		all_matrices[FLAG_hprimewgll_xxT][1][i] *
		tempx1_2D_25_5[map_25_5 (j, 1)] +
		all_matrices[FLAG_hprimewgll_xxT][2][i] *
		tempx1_2D_25_5[map_25_5 (j, 2)] +
		all_matrices[FLAG_hprimewgll_xxT][3][i] *
		tempx1_2D_25_5[map_25_5 (j, 3)] +
		all_matrices[FLAG_hprimewgll_xxT][4][i] *
		tempx1_2D_25_5[map_25_5 (j, 4)];

	      newtempy1_2D_25_5[map_25_5 (j, i)] =
		all_matrices[FLAG_hprimewgll_xxT][0][i] *
		tempy1_2D_25_5[map_25_5 (j, 0)] +
		all_matrices[FLAG_hprimewgll_xxT][1][i] *
		tempy1_2D_25_5[map_25_5 (j, 1)] +
		all_matrices[FLAG_hprimewgll_xxT][2][i] *
		tempy1_2D_25_5[map_25_5 (j, 2)] +
		all_matrices[FLAG_hprimewgll_xxT][3][i] *
		tempy1_2D_25_5[map_25_5 (j, 3)] +
		all_matrices[FLAG_hprimewgll_xxT][4][i] *
		tempy1_2D_25_5[map_25_5 (j, 4)];

	      newtempz1_2D_25_5[map_25_5 (j, i)] =
		all_matrices[FLAG_hprimewgll_xxT][0][i] *
		tempz1_2D_25_5[map_25_5 (j, 0)] +
		all_matrices[FLAG_hprimewgll_xxT][1][i] *
		tempz1_2D_25_5[map_25_5 (j, 1)] +
		all_matrices[FLAG_hprimewgll_xxT][2][i] *
		tempz1_2D_25_5[map_25_5 (j, 2)] +
		all_matrices[FLAG_hprimewgll_xxT][3][i] *
		tempz1_2D_25_5[map_25_5 (j, 3)] +
		all_matrices[FLAG_hprimewgll_xxT][4][i] *
		tempz1_2D_25_5[map_25_5 (j, 4)];
	    }
	}

      for (k = 0; k < NGLLZ; k++)
	{
	  for (j = 0; j < NGLLX; j++)
	    {
	      for (i = 0; i < NGLLX; i++)
		{
		  newtempx2[k][j][i] =
		    tempx2[k][0][i] * all_matrices[FLAG_hprimewgll_xx][j][0] +
		    tempx2[k][1][i] * all_matrices[FLAG_hprimewgll_xx][j][1] +
		    tempx2[k][2][i] * all_matrices[FLAG_hprimewgll_xx][j][2] +
		    tempx2[k][3][i] * all_matrices[FLAG_hprimewgll_xx][j][3] +
		    tempx2[k][4][i] * all_matrices[FLAG_hprimewgll_xx][j][4];

		  newtempy2[k][j][i] =
		    tempy2[k][0][i] * all_matrices[FLAG_hprimewgll_xx][j][0] +
		    tempy2[k][1][i] * all_matrices[FLAG_hprimewgll_xx][j][1] +
		    tempy2[k][2][i] * all_matrices[FLAG_hprimewgll_xx][j][2] +
		    tempy2[k][3][i] * all_matrices[FLAG_hprimewgll_xx][j][3] +
		    tempy2[k][4][i] * all_matrices[FLAG_hprimewgll_xx][j][4];

		  newtempz2[k][j][i] =
		    tempz2[k][0][i] * all_matrices[FLAG_hprimewgll_xx][j][0] +
		    tempz2[k][1][i] * all_matrices[FLAG_hprimewgll_xx][j][1] +
		    tempz2[k][2][i] * all_matrices[FLAG_hprimewgll_xx][j][2] +
		    tempz2[k][3][i] * all_matrices[FLAG_hprimewgll_xx][j][3] +
		    tempz2[k][4][i] * all_matrices[FLAG_hprimewgll_xx][j][4];
		}
	    }
	}

      for (j = 0; j < NGLLX; j++)
	{
	  for (i = 0; i < NGLL2; i++)
	    {
	      newtempx3_2D_5_25[map_5_25 (j, i)] =
		tempx3_2D_5_25[map_5_25 (0, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][0] +
		tempx3_2D_5_25[map_5_25 (1, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][1] +
		tempx3_2D_5_25[map_5_25 (2, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][2] +
		tempx3_2D_5_25[map_5_25 (3, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][3] +
		tempx3_2D_5_25[map_5_25 (4, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][4];

	      newtempy3_2D_5_25[map_5_25 (j, i)] =
		tempy3_2D_5_25[map_5_25 (0, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][0] +
		tempy3_2D_5_25[map_5_25 (1, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][1] +
		tempy3_2D_5_25[map_5_25 (2, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][2] +
		tempy3_2D_5_25[map_5_25 (3, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][3] +
		tempy3_2D_5_25[map_5_25 (4, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][4];

	      newtempz3_2D_5_25[map_5_25 (j, i)] =
		tempz3_2D_5_25[map_5_25 (0, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][0] +
		tempz3_2D_5_25[map_5_25 (1, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][1] +
		tempz3_2D_5_25[map_5_25 (2, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][2] +
		tempz3_2D_5_25[map_5_25 (3, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][3] +
		tempz3_2D_5_25[map_5_25 (4, i)] *
		all_matrices[FLAG_hprimewgll_xx][j][4];
	    }
	}

      for (k = 0; k < NGLLZ; k++)
	{
	  for (j = 0; j < NGLLY; j++)
	    {
	      for (i = 0; i < NGLLX; i++)
		{

		  fac1 = all_matrices[FLAG_wgllwgll_YZ][k][j];
		  fac2 = all_matrices[FLAG_wgllwgll_XZ][k][i];
		  fac3 = all_matrices[FLAG_wgllwgll_XY][j][i];

		  sum_terms[elem][k][j][i][X] =
		    -(fac1 * newtempx1[k][j][i] + fac2 * newtempx2[k][j][i] +
		      fac3 * newtempx3[k][j][i]);
		  sum_terms[elem][k][j][i][Y] =
		    -(fac1 * newtempy1[k][j][i] + fac2 * newtempy2[k][j][i] +
		      fac3 * newtempy3[k][j][i]);
		  sum_terms[elem][k][j][i][Z] =
		    -(fac1 * newtempz1[k][j][i] + fac2 * newtempz2[k][j][i] +
		      fac3 * newtempz3[k][j][i]);

		}
	    }
	}
    }
}
#else // of USE_DEVILLE_INLINED_PRODUCTS
#if defined(_OPENMP)
//#pragma css task input (actual_size,dummy_loc,hprime_xx,hprimewgll_xx,wgllwgll_all,jacobian_matrix,kappa_and_mu) output (sum_terms)
#endif
void
process_element (int actual_size,
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

  float tempx1[NGLLZ][NGLLY][NGLLX];
  float tempx2[NGLLZ][NGLLY][NGLLX];
  float tempx3[NGLLZ][NGLLY][NGLLX];
  float tempy1[NGLLZ][NGLLY][NGLLX];
  float tempy2[NGLLZ][NGLLY][NGLLX];
  float tempy3[NGLLZ][NGLLY][NGLLX];
  float tempz1[NGLLZ][NGLLY][NGLLX];
  float tempz2[NGLLZ][NGLLY][NGLLX];
  float tempz3[NGLLZ][NGLLY][NGLLX];

  int i, j, k, l, elem;
  float xixl, xiyl, xizl, etaxl, etayl, etazl, gammaxl, gammayl, gammazl,
    jacobianl;
  float duxdxl, duxdyl, duxdzl, duydxl, duydyl, duydzl, duzdxl, duzdyl,
    duzdzl;
  float duxdxl_plus_duydyl, duxdxl_plus_duzdzl, duydyl_plus_duzdzl;
  float duxdyl_plus_duydxl, duzdxl_plus_duxdzl, duzdyl_plus_duydzl;
  float sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz;
  float hp1, hp2, hp3, fac1, fac2, fac3, lambdal, mul, lambdalplus2mul,
    kappal;
  float tempx1l, tempx2l, tempx3l, tempy1l, tempy2l, tempy3l, tempz1l,
    tempz2l, tempz3l;

  for (elem = 0; elem < actual_size; elem++)
    {

      for (k = 0; k < NGLLZ; k++)
	{
	  for (j = 0; j < NGLLY; j++)
	    {
	      for (i = 0; i < NGLLX; i++)
		{

		  tempx1l = 0.f;
		  tempx2l = 0.f;
		  tempx3l = 0.f;

		  tempy1l = 0.f;
		  tempy2l = 0.f;
		  tempy3l = 0.f;

		  tempz1l = 0.f;
		  tempz2l = 0.f;
		  tempz3l = 0.f;

		  for (l = 0; l < NGLLX; l++)
		    {
		      hp1 = hprime_xx[l][i];
		      tempx1l = tempx1l + dummy_loc[elem][k][j][l][X] * hp1;
		      tempy1l = tempy1l + dummy_loc[elem][k][j][l][Y] * hp1;
		      tempz1l = tempz1l + dummy_loc[elem][k][j][l][Z] * hp1;

		      hp2 = hprime_xx[l][j];
		      tempx2l = tempx2l + dummy_loc[elem][k][l][i][X] * hp2;
		      tempy2l = tempy2l + dummy_loc[elem][k][l][i][Y] * hp2;
		      tempz2l = tempz2l + dummy_loc[elem][k][l][i][Z] * hp2;

		      hp3 = hprime_xx[l][k];
		      tempx3l = tempx3l + dummy_loc[elem][l][j][i][X] * hp3;
		      tempy3l = tempy3l + dummy_loc[elem][l][j][i][Y] * hp3;
		      tempz3l = tempz3l + dummy_loc[elem][l][j][i][Z] * hp3;
		    }

// compute derivatives of ux, uy and uz with respect to x, y and z
		  xixl = jacobian_matrix[elem][k][j][i][XIX];
		  xiyl = jacobian_matrix[elem][k][j][i][XIY];
		  xizl = jacobian_matrix[elem][k][j][i][XIZ];
		  etaxl = jacobian_matrix[elem][k][j][i][ETAX];
		  etayl = jacobian_matrix[elem][k][j][i][ETAY];
		  etazl = jacobian_matrix[elem][k][j][i][ETAZ];
		  gammaxl = jacobian_matrix[elem][k][j][i][GAMMAX];
		  gammayl = jacobian_matrix[elem][k][j][i][GAMMAY];
		  gammazl = jacobian_matrix[elem][k][j][i][GAMMAZ];
		  jacobianl =
		    1.f / (xixl * (etayl * gammazl - etazl * gammayl) -
			   xiyl * (etaxl * gammazl - etazl * gammaxl) +
			   xizl * (etaxl * gammayl - etayl * gammaxl));

		  duxdxl =
		    xixl * tempx1l + etaxl * tempx2l + gammaxl * tempx3l;
		  duxdyl =
		    xiyl * tempx1l + etayl * tempx2l + gammayl * tempx3l;
		  duxdzl =
		    xizl * tempx1l + etazl * tempx2l + gammazl * tempx3l;

		  duydxl =
		    xixl * tempy1l + etaxl * tempy2l + gammaxl * tempy3l;
		  duydyl =
		    xiyl * tempy1l + etayl * tempy2l + gammayl * tempy3l;
		  duydzl =
		    xizl * tempy1l + etazl * tempy2l + gammazl * tempy3l;

		  duzdxl =
		    xixl * tempz1l + etaxl * tempz2l + gammaxl * tempz3l;
		  duzdyl =
		    xiyl * tempz1l + etayl * tempz2l + gammayl * tempz3l;
		  duzdzl =
		    xizl * tempz1l + etazl * tempz2l + gammazl * tempz3l;

// precompute some sums to save CPU time
		  duxdxl_plus_duydyl = duxdxl + duydyl;
		  duxdxl_plus_duzdzl = duxdxl + duzdzl;
		  duydyl_plus_duzdzl = duydyl + duzdzl;
		  duxdyl_plus_duydxl = duxdyl + duydxl;
		  duzdxl_plus_duxdzl = duzdxl + duxdzl;
		  duzdyl_plus_duydzl = duzdyl + duydzl;

// compute isotropic elements
		  kappal = kappa_and_mu[elem][k][j][i][KAPPA];
		  mul = kappa_and_mu[elem][k][j][i][MU];

		  lambdalplus2mul = kappal + 1.33333333333333333333f * mul;	// 4./3. = 1.3333333
		  lambdal = lambdalplus2mul - 2.f * mul;

// compute stress sigma
		  sigma_xx =
		    lambdalplus2mul * duxdxl + lambdal * duydyl_plus_duzdzl;
		  sigma_yy =
		    lambdalplus2mul * duydyl + lambdal * duxdxl_plus_duzdzl;
		  sigma_zz =
		    lambdalplus2mul * duzdzl + lambdal * duxdxl_plus_duydyl;

		  sigma_xy = mul * duxdyl_plus_duydxl;
		  sigma_xz = mul * duzdxl_plus_duxdzl;
		  sigma_yz = mul * duzdyl_plus_duydzl;

// form dot product with test vector
		  tempx1[k][j][i] =
		    jacobianl * (sigma_xx * xixl + sigma_xy * xiyl +
				 sigma_xz * xizl);
		  tempy1[k][j][i] =
		    jacobianl * (sigma_xy * xixl + sigma_yy * xiyl +
				 sigma_yz * xizl);
		  tempz1[k][j][i] =
		    jacobianl * (sigma_xz * xixl + sigma_yz * xiyl +
				 sigma_zz * xizl);

		  tempx2[k][j][i] =
		    jacobianl * (sigma_xx * etaxl + sigma_xy * etayl +
				 sigma_xz * etazl);
		  tempy2[k][j][i] =
		    jacobianl * (sigma_xy * etaxl + sigma_yy * etayl +
				 sigma_yz * etazl);
		  tempz2[k][j][i] =
		    jacobianl * (sigma_xz * etaxl + sigma_yz * etayl +
				 sigma_zz * etazl);

		  tempx3[k][j][i] =
		    jacobianl * (sigma_xx * gammaxl + sigma_xy * gammayl +
				 sigma_xz * gammazl);
		  tempy3[k][j][i] =
		    jacobianl * (sigma_xy * gammaxl + sigma_yy * gammayl +
				 sigma_yz * gammazl);
		  tempz3[k][j][i] =
		    jacobianl * (sigma_xz * gammaxl + sigma_yz * gammayl +
				 sigma_zz * gammazl);

		}
	    }
	}

      for (k = 0; k < NGLLZ; k++)
	{
	  for (j = 0; j < NGLLY; j++)
	    {
	      for (i = 0; i < NGLLX; i++)
		{

		  tempx1l = 0.f;
		  tempy1l = 0.f;
		  tempz1l = 0.f;

		  tempx2l = 0.f;
		  tempy2l = 0.f;
		  tempz2l = 0.f;

		  tempx3l = 0.f;
		  tempy3l = 0.f;
		  tempz3l = 0.f;

		  for (l = 0; l < NGLLX; l++)
		    {
		      fac1 = hprimewgll_xx[i][l];
		      tempx1l = tempx1l + tempx1[k][j][l] * fac1;
		      tempy1l = tempy1l + tempy1[k][j][l] * fac1;
		      tempz1l = tempz1l + tempz1[k][j][l] * fac1;

		      fac2 = hprimewgll_xx[j][l];
		      tempx2l = tempx2l + tempx2[k][l][i] * fac2;
		      tempy2l = tempy2l + tempy2[k][l][i] * fac2;
		      tempz2l = tempz2l + tempz2[k][l][i] * fac2;

		      fac3 = hprimewgll_xx[k][l];
		      tempx3l = tempx3l + tempx3[l][j][i] * fac3;
		      tempy3l = tempy3l + tempy3[l][j][i] * fac3;
		      tempz3l = tempz3l + tempz3[l][j][i] * fac3;
		    }

		  fac1 = wgllwgll_all[k][j][YZ];
		  fac2 = wgllwgll_all[k][i][XZ];
		  fac3 = wgllwgll_all[j][i][XY];

		  sum_terms[elem][k][j][i][X] =
		    -(fac1 * tempx1l + fac2 * tempx2l + fac3 * tempx3l);
		  sum_terms[elem][k][j][i][Y] =
		    -(fac1 * tempy1l + fac2 * tempy2l + fac3 * tempy3l);
		  sum_terms[elem][k][j][i][Z] =
		    -(fac1 * tempz1l + fac2 * tempz2l + fac3 * tempz3l);

		}
	    }
	}

    }

}
#endif // of USE_DEVILLE_INLINED_PRODUCTS

void
kern_compute_max (int actual_size, float displ[actual_size][NDIM], float *max)
{
  float current_value;
  int i, j, k, iglob;
  float local_max = -1.f;

  for (iglob = 0; iglob < actual_size; iglob++)
    {
      current_value =
	sqrtf (displ[iglob][X] * displ[iglob][X] +
	       displ[iglob][Y] * displ[iglob][Y] +
	       displ[iglob][Z] * displ[iglob][Z]);
      if (current_value > local_max)
	{
	  local_max = current_value;
	}
    }
  if (local_max > *max)
    {
      *max = local_max;
    }
}
