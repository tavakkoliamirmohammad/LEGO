/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"
#include "lud_kernel.cu"

#define TIMING

#ifdef TIMING
//#include "timing.h"
#endif

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif

static int do_verify = 0;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {"runs", 1, NULL, 'r'},
  {0,0,0,0}
};

extern void
lud_cuda(float *d_m, int matrix_dim);

#ifdef TIMING
struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

/* add near top, after includes (stdlib.h already included) */
static int cmp_float(const void *a, const void *b) {
  float fa = *(const float*)a, fb = *(const float*)b;
  return (fa > fb) - (fa < fb);
}


int
main ( int argc, char *argv[] )
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *d_m, *mm;
  int runs = 10;
  int warmups = 25;
  stopwatch sw;
#ifdef TIMING
  float *times = NULL;    /* <— NEW: store per-run kernel times (ms) */
#endif

  while ((opt = getopt_long(argc, argv, "::vs:i:r:w:", 
                            long_options, &option_index)) != -1 ) {
    switch(opt){
    case 'i':
      input_file = optarg;
      break;
    case 'v':
      do_verify = 1;
      break;
    case 's':
      matrix_dim = atoi(optarg);
      printf("Generate input matrix internally, size =%d\n", matrix_dim);
      // fprintf(stderr, "Currently not supported, use -i instead\n");
      // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      // exit(EXIT_FAILURE);
      break;
    case 'r':
      runs = atoi(optarg);
      break;
    case 'w':
      warmups = atoi(optarg);
      break;
    case '?':
      fprintf(stderr, "invalid option\n");
      break;
    case ':':
      fprintf(stderr, "missing argument\n");
      break;
    default:
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
	      argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  
  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  } 
  else if (matrix_dim) {
    // printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }


  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

//  if (do_verify){
    // printf("Before LUD\n");
    // print_matrix(m, matrix_dim);
    // matrix_duplicate(m, &mm, matrix_dim);
    create_matrix(&mm, matrix_dim);
//  }

  cudaSetDevice(0);

#ifdef TIMING
  times = (float*)malloc(sizeof(float) * runs);
  if (!times) { perror("malloc"); exit(EXIT_FAILURE); }
#endif

  for (int run = 0; run < warmups; run++) {

    cudaMalloc((void**)&d_m, 
             matrix_dim*matrix_dim*sizeof(float));
    cudaDeviceSynchronize();

    /* beginning of timing point */
    stopwatch_start(&sw);
    cudaMemcpy(d_m, mm, matrix_dim*matrix_dim*sizeof(float), 
	           cudaMemcpyHostToDevice);

    #ifdef  TIMING
      kernel_time = 0;
      gettimeofday(&tv_kernel_start, NULL);
    #endif

      lud_cuda(d_m, matrix_dim);
      cudaDeviceSynchronize();
    #ifdef  TIMING
      gettimeofday(&tv_kernel_end, NULL);
      tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
      kernel_time = (tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0);
    #endif

      cudaMemcpy(m, d_m, matrix_dim*matrix_dim*sizeof(float), 
	         cudaMemcpyDeviceToHost);

      /* end of timing point */
      stopwatch_stop(&sw);
      //printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));

      cudaFree(d_m);
  }

  for (int run = 0; run < runs; run++) {

    cudaMalloc((void**)&d_m, 
             matrix_dim*matrix_dim*sizeof(float));
    cudaDeviceSynchronize();

    /* beginning of timing point */
    stopwatch_start(&sw);
    cudaMemcpy(d_m, mm, matrix_dim*matrix_dim*sizeof(float), 
	           cudaMemcpyHostToDevice);

    #ifdef  TIMING
      kernel_time = 0;
      gettimeofday(&tv_kernel_start, NULL);
    #endif

      lud_cuda(d_m, matrix_dim);
      cudaDeviceSynchronize();
    #ifdef  TIMING
      gettimeofday(&tv_kernel_end, NULL);
      tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
      kernel_time = (tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0);
    #endif

      cudaMemcpy(m, d_m, matrix_dim*matrix_dim*sizeof(float), 
	         cudaMemcpyDeviceToHost);

      /* end of timing point */
      stopwatch_stop(&sw);
      //printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));

      cudaFree(d_m);

#ifdef TIMING
    // printf("Exec: %f\n", kernel_time);
    times[run] = kernel_time;   /* <— NEW: record this run */
#endif
  }

  #ifdef TIMING
  if (runs > 0) {
    float sum = 0.0f, minv = times[0], maxv = times[0];
    for (int i = 0; i < runs; ++i) {
      float v = times[i];
      sum += v;
      if (v < minv) minv = v;
      if (v > maxv) maxv = v;
    }
    float avg = sum / runs;

    /* median: sort a copy in-place */
    qsort(times, runs, sizeof(float), cmp_float);
    float median = (runs % 2)
        ? times[runs/2]
        : (times[runs/2 - 1] + times[runs/2]) * 0.5f;

    printf("Kernel Exec Time (ms) over %d runs -> min=%.3f  max=%.3f  avg=%.3f  median=%.3f\n",
           runs, minv, maxv, avg, median);
  }
  free(times);
#endif

  if (do_verify){
    printf("After LUD\n");
    // print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim); 
    free(mm);
  }

  free(m);

  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
