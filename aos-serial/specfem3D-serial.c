#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
 #if defined(_OPENMP)
#include <omp.h>
#endif

#include "../DB/values_from_mesher_C.h"

// constant value of the time step in the main time loop
#define deltatover2 0.5f*deltat
#define deltatsqover2 0.5f*deltat*deltat

#define pi 3.141592653589793f
#define f0 (1.f / 50.f)
#define t0 (1.2f / f0)
#define a pi*pi*f0*f0

#define NGLLZ 5
#define NGLLY 5
#define NGLLX 5
#define NGLL2 25

// 3-D simulation
#define NDIM 3

// displacement threshold above which we consider that the code became unstable
#define STABILITY_THRESHOLD 1.e+25f

// #define VERYSMALLVAL 1.e-24f
#define NTSTEP_BETWEEN_OUTPUT_INFO 100 // NSTEP

// approximate density of the geophysical medium in which the source is located
// this value is only a constant scaling factor therefore it does not really matter
#define rho 4500.f

#define PAD 7


//#define CONFIG_VERBOSE	1
#define CONFIG_BENCHMARK	1

long t_start, t_end;
static long usecs ();

static long _counters[64];
static long _t1, _t2;

static inline void timer_init(void)
{
	int i;
	for(i= 0; i < 64; i++)
		_counters[i]= 0;
}

static inline void timer_begin(int i)
{
	_t1 = usecs();
}

static inline void timer_end(int i)
{
	_t2 = usecs();
	_counters[i] += (_t2 - _t1);
}

static inline void report(int n)
{
	int i;
	for(i= 0; i < n; i++)
	{
		fprintf(stdout, "TIMER%d %.6f\n", i, (float) _counters[i] / 1000000.f);
		fflush(stdout);
	}
}


int main(){

  static struct u_tag {
    float dummyx_loc;
    float dummyx_loc_2D_25_5;
    float dummyx_loc_2D_5_25;

    float dummyy_loc;
    float dummyy_loc_2D_25_5;
    float dummyy_loc_2D_5_25;

    float dummyz_loc;
    float dummyz_loc_2D_25_5;
    float dummyz_loc_2D_5_25;

    int padding[PAD];
  } tag_u[NGLLX*NGLLY*NGLLZ];


 //utemp x1,y1,z1

  static struct utemp_tag {
    float tempx1;
    float tempx1_2D_25_5;

    float tempy1;
    float tempy1_2D_25_5;

    float tempz1;
    float tempz1_2D_25_5;

    int padding[PAD+3];

  } utemp[NGLLX*NGLLY*NGLLZ];

  static struct hprime_tag{
    float hprime_xx;
    float hprimewgll_xx;
    float wgllwgll_xy;
    float wgllwgll_xz;
    float wgllwgll_yz;
    float hprime_xxT;
    float hprimewgll_xxT;
    int padding[PAD];
  }tag_h[NGLLX*NGLLY];

  static struct utemp2_tag{
    float tempx2;
    float tempy2;
    float tempz2;
    int padding[PAD+6];
  }utemp2[NGLLX*NGLLY*NGLLZ];

  static struct utempx3_tag{
    float tempx3, tempy3, tempz3;
    float tempx3_2D_5_25, tempy3_2D_5_25, tempz3_2D_5_25;
    int padding[PAD+3];
  }utemp3[NGLLX*NGLLY*NGLLZ];

  static struct xi{
    float xix, xiy, xiz;
    float etax, etay, etaz;
    float gammax, gammay, gammaz;
    float kappav, muv;
    int padding[PAD-2];
  }tag_nspec[NSPEC][NGLLZ*NGLLY*NGLLX];


  float xixl,xiyl,xizl,etaxl,etayl,etazl,gammaxl,gammayl,gammazl,jacobianl;
  float duxdxl,duxdyl,duxdzl,duydxl,duydyl,duydzl,duzdxl,duzdyl,duzdzl;
  float duxdxl_plus_duydyl,duxdxl_plus_duzdzl,duydyl_plus_duzdzl;
  float duxdyl_plus_duydxl,duzdxl_plus_duxdzl,duzdyl_plus_duydzl;
  float sigma_xx,sigma_yy,sigma_zz,sigma_xy,sigma_xz,sigma_yz;
  float lambdal,mul,lambdalplus2mul,kappal;

  static struct unewtemp1{
    float newtempx1, newtempy1, newtempz1;
    float newtempx1_2D_25_5, newtempy1_2D_25_5, newtempz1_2D_25_5;
    float padding[PAD+3];
  }tag_new[NGLLZ*NGLLY*NGLLX];

  static struct unewtemp2{
    float newtempx2, newtempy2, newtempz2;
    float padding[PAD+6];
  }tag_new2[NGLLZ*NGLLY*NGLLZ];

  static struct unewtemp3{
    float newtempx3, newtempy3, newtempz3;
    float newtempx3_2D_5_25, newtempy3_2D_5_25, newtempz3_2D_5_25;
  }tag_new3[NGLLZ*NGLLY*NGLLX];


  float Usolidnorm,current_value,time,memory_size;

  // global diagonal mass matrix
   static float rmass_inverse[NGLOB];

  // record a seismogram to check that the simulation went well
   static float seismogram[NSTEP];
  int ispec,iglob,i,j,k;
  static int ibool[NSPEC][NGLLZ * NGLLY * NGLLX];

  // time step
  int it;
  clock_t timeloop_begin;
  float timeloop_total;


  //auxiliares para matriz
  int auxmat = 1, pos_mat=0;

  //auxiliares para indice
  int aux, aux2;

  //transformação de dipl para struct
  static struct vars{
    float displx,disply,displz;
    float velocx, velocy, velocz;
    float accelx, accely,accelz;
    int pad[PAD]; //padding para que fique uma linha de cache para cada
  }var[NGLOB];

  FILE *IIN;

  #if defined(CONFIG_VERBOSE)
   printf("\nNSPEC = %d\n",NSPEC);
   printf("NGLOB = %d\n\n",NGLOB);
   printf("NSTEP = %d\n",NSTEP);
   printf("deltat = %f\n\n",deltat);
  #endif

  if(NGLLX != 5 || NGLLY != 5 || NGLLZ != 5) {
         fprintf(stderr,"we must have NGLLX = NGLLY = NGLLZ = 5 to be able to use the Deville et al. (2002) routines, exiting...\n");
         exit(1);
       }

  #if defined(CONFIG_VERBOSE)
  memory_size = 4.f * ((3.f*NDIM + 1.f) * NGLOB + 12.f * (float)(NGLLX*NGLLY*NGLLZ)*(float)(NSPEC));
  printf("approximate total memory size used = %f Mb\n\n",memory_size/1024.f/1024.f);
  #endif

  #if 0
   printf("reading file DATABASES_FOR_SOLVER/proc000000_reg1_database.dat\n");
   if((IIN=fopen("DATABASES_FOR_SOLVER/proc000000_reg1_database.dat","r"))==NULL) {
           exit(1);
           fprintf(stderr,"Cannot open file DATABASES_FOR_SOLVER/proc000000_reg1_database.dat, exiting...\n");
         }
   fclose(IIN);
  #else
  #if defined(CONFIG_VERBOSE)
    printf("reading file ../DB/proc000000_reg1_database.dat\n");
  #endif
      if((IIN=fopen("../DB/proc000000_reg1_database.dat","r"))==NULL) {
            fprintf(stderr,"Cannot open file ../multi_GPU_MPI/DATABASES_FOR_SOLVER/proc000000_reg1_database.dat, exiting...\n");
            exit(1);
          }
  #endif


  for (ispec=0;ispec<NSPEC;ispec++) {
    for (i=0; i < NGLLX*NGLLY*NGLLZ; i++){
 // read real numbers here
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].xix);
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].xiy);
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].xiz);
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].etax);
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].etay);
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].etaz);
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].gammax);
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].gammay);
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].gammaz);
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].kappav);
         fscanf(IIN, "%e\n", &tag_nspec[ispec][i].muv);

 // read an integer here
         fscanf(IIN, "%d\n", &ibool[ispec][i]);
 // subtract one because indices start at zero in C but this array was created by a Fortran
 // program and therefore starts at one in the file stored on the disk
          ibool[ispec][i]--;

        }
  }
  for (i=0;i<NGLOB;i++) {
    fscanf(IIN, "%e\n", &rmass_inverse[i]);
  // the real exactly diagonal mass matrix is read (not its inverse)
  // therefore invert it here once and for all
    rmass_inverse[i] = 1.f / rmass_inverse[i];
  }
  fclose(IIN);


  // read the derivation matrices from external file
  #if 0
   printf("reading file DATABASES_FOR_SOLVER/matrices.dat\n");
   if((IIN=fopen("DATABASES_FOR_SOLVER/matrices.dat","r"))==NULL) {
           fprintf(stderr,"Cannot open file DATABASES_FOR_SOLVER/matrices.dat, exiting...\n");
           exit(1);
         }
  #else
  #if defined(CONFIG_VERBOSE)
   printf("reading file ./DB/matrices.dat\n");
  #endif
     if((IIN=fopen("../DB/matrices.dat","r"))==NULL) {
           fprintf(stderr,"Cannot open file DATABASES_FOR_SOLVER/matrices.dat, exiting...\n");
           exit(1);
         }
  #endif

   for (i=0; i < NGLLY * NGLLX; i++){
       //parte chata de conversão de matriz ij para ji
       pos_mat = i/5;
       if (i % 5 != 0){
          pos_mat += 5*auxmat;
          auxmat++;
       }
       else
         auxmat = 1;
       fscanf(IIN, "%e\n", &tag_h[i].hprime_xx);
       fscanf(IIN, "%e\n", &tag_h[i].hprimewgll_xx);
       // compute the transpose matrices
       tag_h[pos_mat].hprime_xxT = tag_h[i].hprime_xx;
       tag_h[pos_mat].hprimewgll_xxT = tag_h[i].hprimewgll_xx;
       fscanf(IIN, "%e\n", &tag_h[i].wgllwgll_yz);
       fscanf(IIN, "%e\n", &tag_h[i].wgllwgll_xz);
       fscanf(IIN, "%e\n", &tag_h[i].wgllwgll_xy);
  }
   fclose(IIN);



  for (i=0;i<NGLOB;i++) {
       var[i].displx = 0.f; // VERYSMALLVAL;
       var[i].disply = 0.f; // VERYSMALLVAL;
       var[i].displz = 0.f; // VERYSMALLVAL;

       var[i].velocx = 0.f;
       var[i].velocy = 0.f;
       var[i].velocz = 0.f;

       var[i].accelx = 0.f;
       var[i].accely = 0.f;
       var[i].accelz = 0.f;
     }

     #if defined(CONFIG_VERBOSE)
      printf("\nstarting the time loop\n\n");
     #endif

      timeloop_begin = clock();

       t_start = usecs ();


// start of the time loop (which must remain serial obviously)
  for (it = 1; it <= NSTEP; it++) {

    if((it % NTSTEP_BETWEEN_OUTPUT_INFO) == 0 || it == 5 || it == NSTEP) {

      Usolidnorm = -1.f;

      for (iglob = 0; iglob < NGLOB; iglob++) {
        current_value = sqrtf(var[iglob].displx*var[iglob].displx + var[iglob].disply*var[iglob].disply + var[iglob].displz*var[iglob].displz);
        if(current_value > Usolidnorm) { Usolidnorm = current_value; }
      }

      #if defined(CONFIG_VERBOSE)
           printf("\nTime step # %d out of %d\n",it,NSTEP);
           printf("Max norm displacement vector U in the solid (m) = %.8g\n",Usolidnorm);
           timeloop_total = ((clock()-timeloop_begin)/(float)CLOCKS_PER_SEC);
           printf("Total elapsed time so far: %f\n",timeloop_total);
           if (it >= 100) { printf("Average elapsed time per time step: %f\n",timeloop_total/(float)(it-1)); }
      #endif

      // write a time stamp file
      #if 0
           sprintf(prname,"timestamp_%07d.txt",it);
           if((IIN = fopen(prname,"w")) == NULL) {
                   fprintf(stderr,"Cannot create time stamp file, exiting...\n");
                   exit(1);
                 }
           fprintf(IIN,"Time step # %d out of %d\n",it,NSTEP);
           fprintf(IIN,"Max norm displacement vector U in the solid (m) = %.8g\n",Usolidnorm);
           fprintf(IIN,"Total elapsed time so far: %f\n",timeloop_total);
           if (it >= 100) { fprintf(IIN,"Average elapsed time per time step: %f\n",timeloop_total/(float)(it-1)); }
           fprintf(IIN,"\n");
           fclose(IIN);
      #endif

      // check stability of the code, exit if unstable
      if(Usolidnorm > STABILITY_THRESHOLD || Usolidnorm < 0) {
        fprintf(stderr,"code became unstable and blew up\n");
        exit(1);
      }
    }
    timer_begin(0);
    for (i=0;i<NGLOB;i++) {
      var[i].displx += deltat*var[i].velocx + deltatsqover2*var[i].accelx;
      var[i].disply += deltat*var[i].velocy + deltatsqover2*var[i].accely;
      var[i].displz += deltat*var[i].velocz + deltatsqover2*var[i].accelz;

      var[i].velocx += deltatover2*var[i].accelx;
      var[i].velocy += deltatover2*var[i].accely;
      var[i].velocz += deltatover2*var[i].accelz;

      var[i].accelx = 0.f;
      var[i].accely = 0.f;
      var[i].accelz = 0.f;
    }
    timer_end(0);


    for (ispec=0;ispec<NSPEC;ispec++) {

    timer_begin(1);
      for (i=0; i < NGLLX * NGLLY * NGLLZ; i++){
          iglob = ibool[ispec][i];
          tag_u[i].dummyx_loc = var[iglob].displx;
          tag_u[i].dummyy_loc = var[iglob].displx;
          tag_u[i].dummyz_loc = var[iglob].displx;
      }
    timer_end(1);


    timer_begin(2);
      for (j=0;j<NGLL2;j++) {
        for (i=0;i<NGLLX;i++) {
          utemp[j*5+i].tempx1_2D_25_5 = tag_h[i].hprime_xx*tag_u[j*5 + 0].dummyx_loc_2D_25_5 +
                                         tag_h[i+5].hprime_xx*tag_u[j*5 + 1].dummyx_loc_2D_25_5 +
                                         tag_h[i+10].hprime_xx*tag_u[j*5 + 2].dummyx_loc_2D_25_5 +
                                         tag_h[i+15].hprime_xx*tag_u[j*5 + 3].dummyx_loc_2D_25_5 +
                                         tag_h[i+20].hprime_xx*tag_u[j*5 + 4].dummyx_loc_2D_25_5;

          utemp[j*5+i].tempy1_2D_25_5 = tag_h[i+0].hprime_xx*tag_u[j*5].dummyy_loc_2D_25_5 +
                                         tag_h[i+5].hprime_xx*tag_u[j*5+1].dummyy_loc_2D_25_5 +
                                         tag_h[i+10].hprime_xx*tag_u[j*5+2].dummyy_loc_2D_25_5+
                                         tag_h[i+15].hprime_xx*tag_u[j*5+3].dummyy_loc_2D_25_5 +
                                         tag_h[i+20].hprime_xx*tag_u[j*5+4].dummyy_loc_2D_25_5;

          utemp[j*5+i].tempz1_2D_25_5 = tag_h[i+0].hprime_xx*tag_u[j*5].dummyz_loc_2D_25_5 +
                                         tag_h[i+5].hprime_xx*tag_u[j*5+1].dummyz_loc_2D_25_5 +
                                         tag_h[i+10].hprime_xx*tag_u[j*5+2].dummyz_loc_2D_25_5 +
                                         tag_h[i+15].hprime_xx*tag_u[j*5+3].dummyz_loc_2D_25_5 +
                                         tag_h[i+20].hprime_xx*tag_u[j*5+4].dummyz_loc_2D_25_5;
        }
      }
    timer_end(2);

    timer_begin(3);
      for (k=0;k<NGLLZ;k++) {
        for (j=0;j<NGLLX;j++) {
          for (i=0;i<NGLLX;i++) {
            utemp2[k*25 + j*5 +i].tempx2 = tag_u[k*25 + i].dummyx_loc*tag_h[j*5 + 0].hprime_xxT +
                              tag_u[k*25 + 5 + i].dummyx_loc*tag_h[j*5 + 1].hprime_xxT +
                              tag_u[k*25 + 10 + i].dummyx_loc*tag_h[j*5 + 2].hprime_xxT +
                              tag_u[k*25 + 15 + i].dummyx_loc*tag_h[j*5 + 3].hprime_xxT +
                              tag_u[k*25 + 20 + i].dummyx_loc*tag_h[j*5 + 4].hprime_xxT;

            utemp2[k*25 + j*5 +i].tempy2 = tag_u[k*25 + i].dummyy_loc*tag_h[j*5 + 0].hprime_xxT +
                              tag_u[k*25 + 5 + i].dummyy_loc * tag_h[j*5 + 1].hprime_xxT +
                              tag_u[k*25 + 10 + i].dummyy_loc*tag_h[j*5 + 2].hprime_xxT +
                              tag_u[k*25 + 15 + i].dummyy_loc*tag_h[j*5 + 3].hprime_xxT +
                              tag_u[k*25 + 20 + i].dummyy_loc*tag_h[j*5 + 4].hprime_xxT;

            utemp2[k*25 + j*5 +i].tempz2 = tag_u[k*25 + i].dummyz_loc*tag_h[j*5 + 0].hprime_xxT +
                              tag_u[k*25 + 5 + i].dummyz_loc*tag_h[j*5 + 1].hprime_xxT +
                              tag_u[k*25 + 10 + i].dummyz_loc*tag_h[j*5 + 2].hprime_xxT +
                              tag_u[k*25 + 15 + i].dummyz_loc*tag_h[j*5 + 3].hprime_xxT +
                              tag_u[k*25 + 20 + i].dummyz_loc*tag_h[j*5 + 4].hprime_xxT;
          }
        }
      }
    timer_end(3);


    timer_begin(4);
        for (j=0;j<NGLLX;j++) {
          for (i=0;i<NGLL2;i++) {
            utemp3[j*25+i].tempx3_2D_5_25 = tag_u[i].dummyx_loc_2D_5_25*tag_h[j*5 + 0].hprime_xxT +
                                           tag_u[25+i].dummyx_loc_2D_5_25*tag_h[j*5 + 1].hprime_xxT +
                                           tag_u[50+i].dummyx_loc_2D_5_25*tag_h[j*5 + 2].hprime_xxT +
                                           tag_u[75+i].dummyx_loc_2D_5_25*tag_h[j*5 + 3].hprime_xxT +
                                           tag_u[100+i].dummyx_loc_2D_5_25*tag_h[j*5 + 4].hprime_xxT;

            utemp3[j*25+i].tempy3_2D_5_25 = tag_u[i].dummyy_loc_2D_5_25*tag_h[j*5].hprime_xxT +
                                           tag_u[25+i].dummyy_loc_2D_5_25*tag_h[j*5 + 1].hprime_xxT +
                                           tag_u[50+i].dummyy_loc_2D_5_25*tag_h[j*5 + 2].hprime_xxT +
                                           tag_u[75+i].dummyy_loc_2D_5_25*tag_h[j*5 + 3].hprime_xxT +
                                           tag_u[100+i].dummyy_loc_2D_5_25*tag_h[j*5 + 4].hprime_xxT;

            utemp3[j*25+i].tempz3_2D_5_25 = tag_u[i].dummyz_loc_2D_5_25*tag_h[j*5].hprime_xxT +
                                           tag_u[25+i].dummyz_loc_2D_5_25*tag_h[j*5+1].hprime_xxT +
                                           tag_u[50+i].dummyz_loc_2D_5_25*tag_h[j*5+2].hprime_xxT +
                                           tag_u[75+i].dummyz_loc_2D_5_25*tag_h[j*5+3].hprime_xxT +
                                           tag_u[100+i].dummyz_loc_2D_5_25*tag_h[j*5+4].hprime_xxT;
          }
        }
    timer_end(4);

    timer_begin(5);
    for (i=0; i < NGLLZ*NGLLY*NGLLX; i++) {
       // compute derivatives of ux, uy and uz with respect to x, y and z
              xixl = tag_nspec[ispec][i].xix;
              xiyl = tag_nspec[ispec][i].xiy;
              xizl = tag_nspec[ispec][i].xiz;
              etaxl = tag_nspec[ispec][i].etax;
              etazl = tag_nspec[ispec][i].etaz;
              gammaxl = tag_nspec[ispec][i].gammax;
              gammayl = tag_nspec[ispec][i].gammay;
              etayl = tag_nspec[ispec][i].etay;
              gammazl = tag_nspec[ispec][i].gammaz;
              jacobianl = 1.f / (xixl*(etayl*gammazl-etazl*gammayl)-xiyl*(etaxl*gammazl-etazl*gammaxl)+xizl*(etaxl*gammayl-etayl*gammaxl));

              duxdxl = xixl*utemp[i].tempx1 + etaxl*utemp2[i].tempx2 + gammaxl*utemp3[i].tempx3;
              duxdyl = xiyl*utemp[i].tempx1 + etayl*utemp2[i].tempx2 + gammayl*utemp3[i].tempx3;
              duxdzl = xizl*utemp[i].tempx1 + etazl*utemp2[i].tempx2 + gammazl*utemp3[i].tempx3;

              duydxl = xixl*utemp[i].tempy1 + etaxl*utemp2[i].tempy2 + gammaxl*utemp3[i].tempy3;
              duydyl = xiyl*utemp[i].tempy1 + etayl*utemp2[i].tempy2 + gammayl*utemp3[i].tempy3;
              duydzl = xizl*utemp[i].tempy1 + etazl*utemp2[i].tempy2 + gammazl*utemp3[i].tempy3;

              duzdxl = xixl*utemp[i].tempz1 + etaxl*utemp2[i].tempz2 + gammaxl*utemp3[i].tempz3;
              duzdyl = xiyl*utemp[i].tempz1 + etayl*utemp2[i].tempz2 + gammayl*utemp3[i].tempz3;
              duzdzl = xizl*utemp[i].tempz1 + etazl*utemp2[i].tempz2 + gammazl*utemp3[i].tempz3;

       // precompute some sums to save CPU time
              duxdxl_plus_duydyl = duxdxl + duydyl;
              duxdxl_plus_duzdzl = duxdxl + duzdzl;
              duydyl_plus_duzdzl = duydyl + duzdzl;
              duxdyl_plus_duydxl = duxdyl + duydxl;
              duzdxl_plus_duxdzl = duzdxl + duxdzl;
              duzdyl_plus_duydzl = duzdyl + duydzl;

       // compute isotropic elements
              kappal = tag_nspec[ispec][i].kappav;
              mul = tag_nspec[ispec][i].muv;

              lambdalplus2mul = kappal + 1.33333333333333333333f * mul;  // 4./3. = 1.3333333
              lambdal = lambdalplus2mul - 2.f*mul;

       // compute stress sigma
              sigma_xx = lambdalplus2mul*duxdxl + lambdal*duydyl_plus_duzdzl;
              sigma_yy = lambdalplus2mul*duydyl + lambdal*duxdxl_plus_duzdzl;
              sigma_zz = lambdalplus2mul*duzdzl + lambdal*duxdxl_plus_duydyl;

              sigma_xy = mul*duxdyl_plus_duydxl;
              sigma_xz = mul*duzdxl_plus_duxdzl;
              sigma_yz = mul*duzdyl_plus_duydzl;

       // form dot product with test vector
          utemp[i].tempx1 = jacobianl * (sigma_xx*xixl + sigma_xy*xiyl + sigma_xz*xizl);
          utemp[i].tempy1 = jacobianl * (sigma_xy*xixl + sigma_yy*xiyl + sigma_yz*xizl);
          utemp[i].tempz1 = jacobianl * (sigma_xz*xixl + sigma_yz*xiyl + sigma_zz*xizl);

          utemp2[i].tempx2 = jacobianl * (sigma_xx*etaxl + sigma_xy*etayl + sigma_xz*etazl);
          utemp2[i].tempy2 = jacobianl * (sigma_xy*etaxl + sigma_yy*etayl + sigma_yz*etazl);
          utemp2[i].tempz2 = jacobianl * (sigma_xz*etaxl + sigma_yz*etayl + sigma_zz*etazl);

          utemp3[i].tempx3 = jacobianl * (sigma_xx*gammaxl + sigma_xy*gammayl + sigma_xz*gammazl);
          utemp3[i].tempy3 = jacobianl * (sigma_xy*gammaxl + sigma_yy*gammayl + sigma_yz*gammazl);
          utemp3[i].tempz3 = jacobianl * (sigma_xz*gammaxl + sigma_yz*gammayl + sigma_zz*gammazl);

      }
    timer_end(5);

    timer_begin(6);
      for (j=0;j<NGLL2;j++) {
        for (i=0;i<NGLLX;i++) {
          tag_new[j*5+i].newtempx1_2D_25_5 = tag_h[i].hprimewgll_xxT*utemp[j*5].tempx1_2D_25_5 +
                                               tag_h[5+i].hprimewgll_xxT*utemp[j*5+1].tempx1_2D_25_5 +
                                               tag_h[10+i].hprimewgll_xxT*utemp[j*5+2].tempx1_2D_25_5 +
                                               tag_h[15+i].hprimewgll_xxT*utemp[j*5+3].tempx1_2D_25_5 +
                                               tag_h[20+i].hprimewgll_xxT*utemp[j*5+4].tempx1_2D_25_5;

          tag_new[j*5+i].newtempy1_2D_25_5 = tag_h[i].hprimewgll_xxT*utemp[j*5].tempy1_2D_25_5 +
                                               tag_h[i+5].hprimewgll_xxT*utemp[j*5+1].tempy1_2D_25_5 +
                                               tag_h[i+10].hprimewgll_xxT*utemp[j*5+2].tempy1_2D_25_5 +
                                               tag_h[i+15].hprimewgll_xxT*utemp[j*5+3].tempy1_2D_25_5 +
                                               tag_h[i+20].hprimewgll_xxT*utemp[j*5+4].tempy1_2D_25_5;

          tag_new[j*5+i].newtempz1_2D_25_5 = tag_h[i].hprimewgll_xxT*utemp[j*5].tempz1_2D_25_5 +
                                               tag_h[i+5].hprimewgll_xxT*utemp[j*5+1].tempz1_2D_25_5 +
                                               tag_h[i+10].hprimewgll_xxT*utemp[j*5+2].tempz1_2D_25_5 +
                                               tag_h[i+15].hprimewgll_xxT*utemp[j*5+3].tempz1_2D_25_5 +
                                               tag_h[i+20].hprimewgll_xxT*utemp[j*5+4].tempz1_2D_25_5;
        }
      }
    timer_end(6);

//estou aqui

    timer_begin(7);
      for (k=0;k<NGLLZ;k++) {
        for (j=0;j<NGLLX;j++) {
          for (i=0;i<NGLLX;i++) {
            aux = k*25+j*5+i;
            tag_new2[aux].newtempx2 = utemp2[k*25+i].tempx2*tag_h[j].hprimewgll_xx +
                                 utemp2[k*25+i+5].tempx2*tag_h[j+1].hprimewgll_xx +
                                 utemp2[k*25+i+10].tempx2*tag_h[j+2].hprimewgll_xx +
                                 utemp2[k*25+i+15].tempx2*tag_h[j+3].hprimewgll_xx +
                                 utemp2[k*25+i+20].tempx2*tag_h[j+4].hprimewgll_xx;

            tag_new2[aux].newtempy2 = utemp2[k*25+i].tempy2 * tag_h[j].hprimewgll_xx +
                                 utemp2[k*25+i+5].tempy2*tag_h[j+1].hprimewgll_xx +
                                 utemp2[k*25+i+10].tempy2*tag_h[j+2].hprimewgll_xx +
                                 utemp2[k*25+i+15].tempy2*tag_h[j+3].hprimewgll_xx +
                                 utemp2[k*25+i+20].tempy2*tag_h[j+4].hprimewgll_xx;

            tag_new2[aux].newtempz2 = utemp2[k*25+i].tempz2*tag_h[j].hprimewgll_xx +
                                 utemp2[k*25+i+5].tempz2*tag_h[j+1].hprimewgll_xx +
                                 utemp2[k*25+i+10].tempz2*tag_h[j+2].hprimewgll_xx +
                                 utemp2[k*25+i+15].tempz2*tag_h[j+3].hprimewgll_xx +
                                 utemp2[k*25+i+20].tempz2*tag_h[j+4].hprimewgll_xx;
          }
        }
      }
    timer_end(7);

    timer_begin(8);
        for (j=0;j<NGLLX;j++) {
          for (i=0;i<NGLL2;i++) {
            aux = j*25+i;
            tag_new3[aux].newtempx3_2D_5_25 = utemp3[i].tempx3_2D_5_25*tag_h[j*25].hprimewgll_xx+
                                                 utemp3[i+25].tempx3_2D_5_25*tag_h[j*25+1].hprimewgll_xx +
                                                 utemp3[i+50].tempx3_2D_5_25*tag_h[j*25+2].hprimewgll_xx+
                                                 utemp3[i+75].tempx3_2D_5_25*tag_h[j*25+3].hprimewgll_xx +
                                                 utemp3[i+100].tempx3_2D_5_25*tag_h[j*25+4].hprimewgll_xx;

            tag_new3[aux].newtempy3_2D_5_25 = utemp3[i].tempy3_2D_5_25*tag_h[j*25].hprimewgll_xx +
                                                 utemp3[i+25].tempy3_2D_5_25*tag_h[j*25+1].hprimewgll_xx +
                                                 utemp3[i+50].tempy3_2D_5_25*tag_h[j*25+2].hprimewgll_xx +
                                                 utemp3[i+75].tempy3_2D_5_25*tag_h[j*25+3].hprimewgll_xx +
                                                 utemp3[i+100].tempy3_2D_5_25*tag_h[j*25+4].hprimewgll_xx;

            tag_new3[aux].newtempz3_2D_5_25 = utemp3[i].tempz3_2D_5_25*tag_h[j*25].hprimewgll_xx +
                                                 utemp3[i+25].tempz3_2D_5_25*tag_h[j*25+1].hprimewgll_xx +
                                                 utemp3[i+50].tempz3_2D_5_25*tag_h[j*25+2].hprimewgll_xx +
                                                 utemp3[i+75].tempz3_2D_5_25*tag_h[j*25+3].hprimewgll_xx +
                                                 utemp3[i+100].tempz3_2D_5_25*tag_h[j*25+4].hprimewgll_xx;
          }
        }
    timer_end(8);

    timer_begin(9);
     for (k=0;k<NGLLZ;k++) {
       for (j=0;j<NGLLY;j++) {
          for (i=0;i<NGLLX;i++) {
              aux = k*25+j*5+i;
              aux2 = k*5+j;
     // sum contributions from each element to the global mesh using indirect addressing
              iglob = ibool[ispec][aux];
              var[iglob].accelx -= (tag_h[aux2].wgllwgll_yz * tag_new[aux].newtempx1 + tag_h[k*5+i].wgllwgll_xz*tag_new2[aux].newtempx2 + tag_h[j*5+i].wgllwgll_xy*tag_new3[aux].newtempx3);
              var[iglob].accely -= (tag_h[aux2].wgllwgll_yz * tag_new[aux].newtempy1 + tag_h[k*5+i].wgllwgll_xz*tag_new2[aux].newtempy2 + tag_h[j*5+i].wgllwgll_xy*tag_new3[aux].newtempy3);
              var[iglob].accelz -= (tag_h[aux2].wgllwgll_yz * tag_new[aux].newtempz1 + tag_h[k*5+i].wgllwgll_xz*tag_new2[aux].newtempz2 + tag_h[j*5+i].wgllwgll_xy*tag_new3[aux].newtempz3);
            }
          }
        }
    timer_end(9);
 }  // end of main loop on all the elements

// big loop over all the global points (not elements) in the mesh to update
// the acceleration and velocity vectors
    timer_begin(10);
 for (i=0;i<NGLOB;i++) {
   var[i].accelx *= rmass_inverse[i];
   var[i].accely *= rmass_inverse[i];
   var[i].accelz *= rmass_inverse[i];
 }
    timer_end(10);

 // add the earthquake source at a given grid point
 // this is negligible and is intrinsically serial because it is done by only
 // one grid point out of several millions typically
 // we subtract one to the element number of the source because arrays start at 0 in C
 // compute current time
  time = (it-1)*deltat;
  var[ibool[NSPEC_SOURCE-1][31]].accelz += 1.e4f * (1.f - 2.f*a*(time-t0)*(time-t0)) * expf(-a*(time-t0)*(time-t0)) / rho;

    timer_begin(11);
  for (i=0;i<NGLOB;i++) {
    var[i].velocx += deltatover2*var[i].accelx;
    var[i].velocy += deltatover2*var[i].accely;
    var[i].velocz += deltatover2*var[i].accelz;
  }
    timer_end(11);

 // record a seismogram to check that the simulation went well
 // we subtract one to the element number of the receiver because arrays start at 0 in C
    seismogram[it-1] = var[ibool[NSPEC_STATION-1][31]].displz;

  } // end of the serial time loop
   t_end = usecs ();
  report(12);

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
   printf ("elapsed time: %f seconds\n", (float) (t_end - t_start) / 1000000.f);


 // save the seismogram at the end of the run
  char filename[50];
  sprintf (filename, "seismogram_aos_serial_%d.txt", getpid ());
 if((IIN = fopen(filename,"w")) == NULL) {
         fprintf(stderr,"Cannot create file %s, exiting...\n", filename);
          exit(1);
        }
  for (it=0;it<NSTEP;it++)
  {  fprintf(IIN,"%e %e\n",it*deltat,seismogram[it]);
  }
  fclose(IIN);
 #endif

 #if defined(CONFIG_VERBOSE)
  printf("\nEnd of the program\n\n");
 #endif

  }

 long usecs () {
   struct timeval t;

   gettimeofday (&t, NULL);
   return t.tv_sec * 1000000 + t.tv_usec;

}
