/************************************************************************************************************
Author: Sultan Hassan
Module: very simple RT with GPU
*************************************************************************************************************/

#ifdef _OMPTHREAD_
#include <omp.h>
#endif
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>




#define PI 3.1415926535897932385E0
#define bfactor 1.1  /* value by which to divide bubble size R */
#define xHlim 0.999 /* cutoff limit for bubble calculation */
#define CM_PER_MPC 3.0857e24
#define CM_PER_PC 3.086e+18
#define MASSFRAC 0.76
#define sigma_L 6.3e-18

double get_wall_time();
double get_cpu_time();






__global__ void gamma_cycle(long int num_cells, double cell_size, double lambda  ,double * reion, double * gamma, double * NH, double * xHI){
  double    radius, di, dj, dp; 
  int ind1  = blockIdx.x * blockDim.x + threadIdx.x; // index 1 goes from 0 to the total number of cells// this is for the sinks//
  int ind2  = blockIdx.y * blockDim.y + threadIdx.y; // same for index 2 // this is for the source//

  if (  ind1 < num_cells*num_cells*num_cells && ind2 < num_cells*num_cells*num_cells){


    if(reion[ind2] == 0.0) return; // skip if no photons from the source cell 


    // get the i,j,k from the global N3 for the source and sink to compute the distance//

     di = ind1  / ( num_cells * num_cells)   - ind2 / (num_cells * num_cells) ;       
     dj = (ind1 / num_cells) % num_cells     - (ind2 / num_cells) % num_cells ;        
     dp = ind1 % num_cells - ind2 % num_cells;
      

     radius = sqrt(di*di + dj*dj + dp*dp)*cell_size; //cartesian distance


     if(radius == 0.0){ // if the source within the sink cell..//

       if(xHI[ind2]*NH[ind2] == 0.0) return; // skip if sink is in void//

       gamma[ind1] += reion[ind2]/(xHI[ind2]*NH[ind2]*cell_size*CM_PER_PC*cell_size*CM_PER_PC*cell_size*CM_PER_PC); // approximate gamma as rion/NHI
       
     }else{	

       gamma[ind1] += (reion[ind2]/(4.0*PI*radius*CM_PER_PC*radius*CM_PER_PC))*exp(-radius/lambda)*sigma_L;	// attenuate rion by the optical depth assuming mean free path, no ray-tracing just yet//

     }
  }
}

//expample for choosing the arrays ...-------
// number of cells = 16
// total is 16 * 16 * 16
// need to define
// dim3 grid(grid_size, grid_size, 1);
// dim3 block(block_size,  block_size, 1);
// but block_size in 3d shouldn't be more than 1024 threads
// maximum in 2d is 32 * 32 threads
// so total number of cells ( 16 * 16 * 16 ) / threads in each block dim (32)  = 128
// ----> grid size has to be 128
// dim3 grid(128, 128, 1);                                                                                                                                                           
// dim3 block(32, 32 , 1);   
// gamma_cycle<<<grid,block>>>(num_cells,  cell_size,  lambda  , reion_g,  gamma_g, NH_g, xHI_g);


int main(int argc, char *argv[]) {
  
  char fname[300];
  FILE *fid;
  DIR* dir;
  size_t elem;
  long int num_cells,all_cells;
  long int i;//,j,p, i1,j1,p1, di, dj, dp;
  
  int index,index1;
  double  radius, tau,  lambda;
  double *tau_box, *xHI;
  double *reion, *gamma, *NHI_old, *NHI_new, *NH, dt_add;
  double time_step, cell_size;
  double neutral, ifront, dt, add_gama, add_rec;
  double *gamma_g, *NH_g, *xHI_g, *reion_g, *tau_box_g;

  
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();
  
  
#ifdef _OMPTHREAD_
  omp_set_num_threads(24);
  printf("Using %d threads\n",global_nthreads);fflush(0);
#endif


  num_cells = 16;
  all_cells = num_cells*num_cells*num_cells;
  cell_size = 6.6*1e3/num_cells; //pc
  dt        = 0.01; // Myr


  // host memory.......

  // photoionization rate, gamma array//
  if(!(gamma=(double *) malloc(all_cells*sizeof(double)))) {
    printf("Problem1...\n");
    exit(1);
  }

  // ionization rate, photons per second, rion array//
  if(!(reion=(double *) malloc(all_cells*sizeof(double)))) {
    printf("Problem4...\n");
    exit(1);
  }

  // number density array // 
  if(!(NH=(double *) malloc(all_cells*sizeof(double)))) {
    printf("Problem16...\n");
    exit(1);
  }

  // neutral fraction array//
  if(!(xHI=(double *) malloc(all_cells*sizeof(double)))) {
    printf("Problem19...\n");
    exit(1);
  }
  // allocate device mem.....

  cudaMalloc((void**)&gamma_g, sizeof(double) *all_cells);
  cudaMalloc((void**)&NH_g, sizeof(double) *all_cells);
  cudaMalloc((void**)&xHI_g, sizeof(double) *all_cells);
  cudaMalloc((void**)&reion_g, sizeof(double) *all_cells); 
  



// set the initial vlaues for the uniform density (1e^-3), and neutral fraction (0.9988), following  Test 1 in Iliev et al (2006), https://arxiv.org/abs/astro-ph/0603199  

#ifdef _OMPTHREAD_
#pragma omp parallel for shared(all_cells,NH, xHI) private(i)
#endif
  for(i=0;i<all_cells;i++){
    xHI[i] = 0.9988;
    NH[i]  = 1.e-3;
  }
  neutral = 0.9988;
  // locate source at the corner which emits 5e48 s^-1 per second 

  reion[0] = 5e48; //s^-1


  // loop up to 500 Myr //
  for(time_step=0.;time_step<= 500.0;time_step+=dt){
    
    memset(gamma, 0.0, all_cells*sizeof(double)); // zero out the photo ionization rate (gamma) in the beginning of each time step, should be instantaneous, not cumulative! 
    lambda   = 2.2*cell_size*pow(neutral,-2./3.); // assume a global mean free path inversely related to the global neutral fraction, following MHR (2000), https://iopscience.iop.org/article/10.1086/308330/meta


    // transfer data from host to device
    cudaMemcpy(gamma_g, gamma, sizeof(double) *all_cells, cudaMemcpyHostToDevice);
    cudaMemcpy(reion_g, reion, sizeof(double) *all_cells, cudaMemcpyHostToDevice);
    cudaMemcpy(NH_g   , NH   , sizeof(double) *all_cells, cudaMemcpyHostToDevice);
    cudaMemcpy(xHI_g  , xHI  , sizeof(double) *all_cells, cudaMemcpyHostToDevice);

    int block_size = 32;
    int grid_size  = 128;

    dim3 grid(grid_size, grid_size, 1);
    dim3 block(block_size,  block_size, 1);


    gamma_cycle<<<grid,block>>>(num_cells,  cell_size,  lambda  , reion_g,  gamma_g, NH_g, xHI_g);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));


    // transfer data back to host ....
    cudaMemcpy(gamma, gamma_g,  sizeof(double) *all_cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(reion, reion_g,  sizeof(double) *all_cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(NH   , NH_g   ,  sizeof(double) *all_cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(xHI  , xHI_g  ,  sizeof(double) *all_cells, cudaMemcpyDeviceToHost);
    //cudaMemcpy(tau_box  , tau_box_g  ,  sizeof(double) *all_cells, cudaMemcpyDeviceToHost);

    neutral = 0.0;
    add_gama = 0.0;
    add_rec = 0.0;

    // solve the RT equation to update the neutral fraction array...//
#ifdef _OMPTHREAD_
#pragma omp parallel for  shared(all_cells, gamma, xHI, dt, NH) reduction(+: neutral, add_gama, add_rec) private(i) 
#endif
    for(i=0;i<all_cells;i++){
      xHI[i] = xHI[i]  +  (   2.59e-13*(1.0-xHI[i])*(1.0-xHI[i])*NH[i]  -  gamma[i]*xHI[i]    )*dt*3.153599e13;
      if(xHI[i] < 0.0 ) xHI[i] = 0.0;
      neutral += xHI[i];
      add_gama+=  gamma[i]*xHI[i];
      add_rec +=  2.59e-13*(1.0-xHI[i])*(1.0-xHI[i])*NH[i];
    }

    neutral /= all_cells;

    // find the position of the wave front, defined as the distance at 0.5 neutral fraction from the source in any direction, choose z-axis.
    for(i=0;i<num_cells;i++){
      if( xHI[i] >= 0.49){
	ifront = (double) i + ( (1.0)/(xHI[i+1]-xHI[i]) ) * (0.5 - xHI[i]);
	break;
      }
    }

    printf("%e %e %e %e %e %e\n", neutral,time_step, lambda, ifront*cell_size/5393.161517492427, add_gama, add_rec);fflush(0);

  } // end time cycle....

  // Time spent //
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();
  printf("#Wall Time = %f min\n",(wall1 - wall0)/60.0);
  printf("#CPU Time  = %f min\n", (cpu1  - cpu0)/60.0 );
  

  // free memory //

  cudaFree(NH_g);
  cudaFree(xHI_g);
  cudaFree(gamma_g);
  cudaFree(reion_g);
  

  free(NH);
  free(xHI);
  free(gamma);
  free(reion);
  
  exit(0);
}


double get_wall_time(){
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    //  Handle error
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
  return (double)clock() / CLOCKS_PER_SEC;
}
