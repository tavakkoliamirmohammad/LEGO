#include <cuda.h>
#include <stdio.h>

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif


__global__ void 
lud_diagonal(float *m, int matrix_dim, int offset)
{
  int i,j;
  __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

  int array_offset = offset*matrix_dim+offset;
  for(i=0; i < BLOCK_SIZE; i++){
    shadow[i][threadIdx.x]=m[array_offset+threadIdx.x];
    array_offset += matrix_dim;
  }
  __syncthreads();
  for(i=0; i < BLOCK_SIZE-1; i++) {

    if (threadIdx.x>i){
      for(j=0; j < i; j++)
        shadow[threadIdx.x][i] -= shadow[threadIdx.x][j]*shadow[j][i];
      shadow[threadIdx.x][i] /= shadow[i][i];
    }

    __syncthreads();
    if (threadIdx.x>i){

      for(j=0; j < i+1; j++)
        shadow[i+1][threadIdx.x] -= shadow[i+1][j]*shadow[j][threadIdx.x];
    }
    __syncthreads();
  }

  /* 
     The first row is not modified, it
     is no need to write it back to the
     global memory

   */
  array_offset = (offset+1)*matrix_dim+offset;
  for(i=1; i < BLOCK_SIZE; i++){
    m[array_offset+threadIdx.x]=shadow[i][threadIdx.x];
    array_offset += matrix_dim;
  }
}

__global__ void
lud_perimeter(float *m, int matrix_dim, int offset)
{
  __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i,j, array_offset;
  int idx;

  if (threadIdx.x < BLOCK_SIZE) {
    idx = threadIdx.x;
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i][idx]=m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

  } else {
    idx = threadIdx.x-BLOCK_SIZE;
    
    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i][idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }
  
  }
  __syncthreads();

/* this version works ok on hardware, but not gpgpusim
 **************************************************************
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    }

    
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }

    __syncthreads();
    
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }
***************************************************************/
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }
  }

  __syncthreads();
    
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }

}

__global__ void
lud_internal(float *m, int matrix_dim, int offset)
{
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i;
  float sum;

  int global_row_id = offset + (blockIdx.y+1)*BLOCK_SIZE;
  int global_col_id = offset + (blockIdx.x+1)*BLOCK_SIZE;

  peri_row[threadIdx.y][threadIdx.x] = m[(offset+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x];
  peri_col[threadIdx.y][threadIdx.x] = m[(global_row_id+threadIdx.y)*matrix_dim+offset+threadIdx.x];

  __syncthreads();

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
  m[(global_row_id+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x] -= sum;


}

/**
 * This code is not part of Rodinia. It was written by Cosmin
 *   as mid-point to figure out the layout and for comparison.
 */
template<int R>
__global__ void
lud_internal_reg_tiled(float *m, int matrix_dim, int offset)
{
  int T = BLOCK_SIZE / R;
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i, j, k;
  float sum[R][R] = {0};

  int global_row_id = offset + (blockIdx.y+1)*BLOCK_SIZE;
  int global_col_id = offset + (blockIdx.x+1)*BLOCK_SIZE;

  for(i=0; i<R; i++) {
    for(j=0; j<R; j++) {
      int tidy = threadIdx.y*R+i;
      int tidx = threadIdx.x*R+j;
      peri_row[tidy][tidx] = m[(offset+tidy)*matrix_dim+global_col_id+tidx];
      peri_col[tidy][tidx] = m[(global_row_id+tidy)*matrix_dim+offset+tidx];
    }
  }
  __syncthreads();

  //#pragma unroll
  for (k=0; k < BLOCK_SIZE; k++) {
    for(i=0; i<R; i++) {
      for(j=0; j<R; j++) {
         int tidy = threadIdx.y*R+i;
         int tidx = threadIdx.x*R+j;
         sum[i][j] += peri_col[tidy][k] * peri_row[k][tidx];
      }
    }
  }
  
  for(i=0; i<R; i++) {
    for(j=0; j<R; j++) {
      int tidy = threadIdx.y*R+i;
      int tidx = threadIdx.x*R+j;
      m[(global_row_id+tidy)*matrix_dim+global_col_id+tidx] -= sum[i][j];
    }
  }
}

/**
 * This is the ideally simplified `inv` function corresponding to the
 *   simple LEGO layout: 
 *       GroupBy( [R][T][R][T], OrderBy( RegP([R,R,T,T], [1,3,2,4]) ) )
 * Semantically the flat index to be unfolded is
 *       ` (ii*R+jj)*T*T + threadIdx.x `
 * Hopefully, it can be simplified to multi-dimension index:
 *       ` [ii, jj, threadIdx.x/T, threadIdx.x%T] ` 
 */
// template<int T, int R>
// __device__ __host__ inline
// void invOpt(int ii, int jj, int tid, int& ty, int& i, int& tx, int& j) {
//   i  = ii;
//   j  = jj;
//   ty = tid / T;
//   tx = tid % T;
// }

template<int T, int R>
__device__ __host__ inline
void invOpt(int ii, int jj, int tid, int& ty, int& i, int& tx, int& j) {
  i  = {{ i }};
  j  = {{ j }};
  ty = {{ tidx }};
  tx = {{ tidy}};
}

/**
 * We aim to perform thread coarsening by expressing a layout
 *   on the structure of virtual threads in the block.
 * For this purpose, we use a one-dimensional (flat) CUDA block,
 *   and we use the inverse (inv) function of LEGO to unfold it
 *   into a logical-view thread structure of shape [R][T][R][T],
 *   where `BLOCK_SIZE = R * T` was the original size of CUDA block
 *   on both x and y dimensions. T is semantically the shrunk
 *   (parallel) block size in both dimensions and R is the
 *   sequentialization factor in both dimensions (register tile).
 *
 * `inv` is semantically called with an expression such as
 *       `(ii*R + j) * T * T + threadIdx.x`
 *    and should ideally produce:
 *       `(ii, threadIdx.x / T, jj, threadIdx.x % T)`
 * To estimate performance, the code uses the function `invOpt`,
 *   defined above as the ideal implementation of `inv`.
 *
 * This is a somewhat contrived case, as it involves explicitly
 *   wrapping each block in the original code into two additional
 *   loops of count R and indices `ii` and `jj`.
 *
 * However, if we are desperate and LEGO can perform the simplification
 *   described above, we may go for it to strengthen the paper.
 */
template<int R>
__global__ void
lud_internal_lego(float *m, int matrix_dim, int offset)
{
  const int T = BLOCK_SIZE / R;
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int k, ii, jj;
  float sum[R][R] = {0};

  int global_row_id = offset + (blockIdx.y+1)*BLOCK_SIZE;
  int global_col_id = offset + (blockIdx.x+1)*BLOCK_SIZE;

  for(ii = 0; ii < R; ii++) {
    for(jj = 0; jj < R; jj++) {
      int ty, i, tx, j;
      invOpt<T,R>(ii, jj, threadIdx.x, ty, i, tx, j);
      int tidy = i*T + ty;
      int tidx = j*T + tx;
      //
      peri_row[tidy][tidx] = m[(offset+tidy)*matrix_dim+global_col_id+tidx];
      peri_col[tidy][tidx] = m[(global_row_id+tidy)*matrix_dim+offset+tidx];
    }
  }
  __syncthreads();

  //#pragma unroll
  for (k=0; k < BLOCK_SIZE; k++) {
    for(ii = 0; ii < R; ii++) {
      for(jj = 0; jj < R; jj++) {
         int ty, i, tx, j;
         invOpt<T,R>(ii, jj, threadIdx.x, ty, i, tx, j);
         int tidy = i*T + ty;
         int tidx = j*T + tx;
         sum[i][j] += peri_col[tidy][k] * peri_row[k][tidx];
      }
    }
  }
  
  for(ii = 0; ii < R; ii++) {
    for(jj = 0; jj < R; jj++) {
      int ty, i, tx, j;
      invOpt<T,R>(ii, jj, threadIdx.x, ty, i, tx, j);
      int tidy = i*T + ty;
      int tidx = j*T + tx;
      m[(global_row_id+tidy)*matrix_dim+global_col_id+tidx] -= sum[i][j];
    }
  }
}

///////////////////////////////////////////////////////////////////////////
/// HOST CODE
///////////////////////////////////////////////////////////////////////////

void lud_cuda(float *m, int matrix_dim)
{
  int i=0;
  float *m_debug = (float*)malloc(matrix_dim*matrix_dim*sizeof(float));

  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
      lud_diagonal<<<1, BLOCK_SIZE>>>(m, matrix_dim, i);
      lud_perimeter<<<(matrix_dim-i)/BLOCK_SIZE-1, BLOCK_SIZE*2>>>(m, matrix_dim, i);
      dim3 dimGrid((matrix_dim-i)/BLOCK_SIZE-1, (matrix_dim-i)/BLOCK_SIZE-1);
#ifndef LEGO
      {
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        lud_internal<<<dimGrid, dimBlock>>>(m, matrix_dim, i);
      }
#endif
#ifdef LEGO_R16
    {
        lud_internal_lego<16><<<dimGrid, (BLOCK_SIZE/16) * (BLOCK_SIZE/16)>>>(
            m, matrix_dim, i
        );
    }
#elif defined(LEGO_R8)
    {
        lud_internal_lego<8><<<dimGrid, (BLOCK_SIZE/8) * (BLOCK_SIZE/8)>>>(
            m, matrix_dim, i
        );
    }
#elif defined(LEGO_R4)
    {
        lud_internal_lego<4><<<dimGrid, (BLOCK_SIZE/4) * (BLOCK_SIZE/4)>>>(
            m, matrix_dim, i
        );
    }
#elif defined(LEGO_R2)
    {
        lud_internal_lego<2><<<dimGrid, (BLOCK_SIZE/2) * (BLOCK_SIZE/2)>>>(
            m, matrix_dim, i
        );
    }
#endif
  }
  lud_diagonal<<<1,BLOCK_SIZE>>>(m, matrix_dim, i);
}

