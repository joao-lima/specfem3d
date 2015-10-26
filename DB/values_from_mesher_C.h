 #define NSPEC         2160
 #define NGLOB       147645
// #define NSTEP         6000
// #define NSTEP         600
 #define NSTEP         20
#define deltat   0.2000000000E+00f
 #define NGLOB2DMAX_XMIN_XMAX         4294
 #define NGLOB2DMAX_YMIN_YMAX         4294
 #define NPROC_XI            4
 #define NPROC_ETA            4
 
 // element and MPI slice number of the source and the station
 // after permutation of the elements by Cuthill-McKee in the mesher
 // (before permutation they are 10 and NSPEC - 10)
 #define RANK_SOURCE 0
 #define NSPEC_SOURCE          830
 
 #define RANK_STATION 0
 #define NSPEC_STATION          322
