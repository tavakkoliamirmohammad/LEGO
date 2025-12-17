
#include "needle.h"
#include <stdio.h>

#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

__host__ __device__ int 
antiDiagPermLEGO(int n, int i, int j) {
  return ((i + j - n <= -1) ? (
   i*j + i + ((i*i + i + j*j + j)/(2))
)
: (
   -i*j + 2*i*n + i + 2*j*n - n*n + 2*n - ((i*i + 3*i + j*j + 3*j)/(2)) - 1
));
}

__host__ __device__ int 
antiDiagPermBase(int n, int i, int j) {
  int flat_ind;
  int antidiag = i + j + 1;
  
  if(antidiag <= n) {
      int gauss_sum = antidiag * (antidiag - 1) / 2;
      flat_ind = gauss_sum + i;
  } else {
      int antidiag_rev = 2*n - antidiag;
      int gauss_sum = antidiag_rev * (antidiag_rev - 1) / 2;
      flat_ind = (n*n - n) + j - gauss_sum;
  }

  return flat_ind;
}

__host__ __device__ int 
antiDiagPerm(int n, int i, int j) {
  return antiDiagPermLEGO(n, i, j);
}

struct Item {
    int* buff;
    int  i;
    __device__ Item(int* buf) { buff = buf; i = 0; }
    __device__ int& operator[](int j) {
        return buff[ antiDiagPerm(BLOCK_SIZE+1, i, j) ];        
    }
};

struct DelayedArray {
    Item* item;
    __device__ DelayedArray(Item* item) { 
        this->item = item; 
    }
    __device__ Item& operator[](int ind) {
        item->i = ind;
        return *item;
    }
};


__device__ __host__ int 
maximum( int a,
		 int b,
		 int c){

int k;
if( a <= b )
k = b;
else 
k = a;

if( k <=c )
return(c);
else
return(k);

}

//#define OPTIM 1

__global__ void
needle_cuda_shared_1(  int* referrence,
			  int* matrix_cuda, 
			  int cols,
			  int penalty,
			  int i,
			  int block_width) 
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx;
  int b_index_y = i - 1 - bx;

  int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
  int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

#if OPTIM == 1
  __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+2];
#elif OPTIM == 0
  __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
#else
  __shared__  int temp0[BLOCK_SIZE+1][BLOCK_SIZE+1];
  Item item( &(temp0[0][0]) );
  DelayedArray temp( &item );
#endif
  __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE];

   if (tx == 0)
		  temp[tx][0] = matrix_cuda[index_nw];


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  ref[ty][tx] = referrence[index + cols * ty];

  __syncthreads();

  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  __syncthreads();

  temp[0][tx + 1] = matrix_cuda[index_n];
  
  __syncthreads();
  

  for( int m = 0 ; m < BLOCK_SIZE ; m++){
   
	  if ( tx <= m ){

		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;

          temp[t_index_y][t_index_x] = 
          	maximum( temp[t_index_y-1][t_index_x-1] + 
          	            ref[t_index_y-1][t_index_x-1],
		                 temp[t_index_y][t_index_x-1]  - penalty, 
										 temp[t_index_y-1][t_index_x]  - penalty
									 );

		  
	  
	  }

	  __syncthreads();
  
    }

 for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){

		  int t_index_x =  tx + BLOCK_SIZE - m ;
		  int t_index_y =  BLOCK_SIZE - tx;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);
	   
	  }

	  __syncthreads();
  }

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

}


__global__ void
needle_cuda_shared_2(  int* referrence,
			  int* matrix_cuda, 
			 
			  int cols,
			  int penalty,
			  int i,
			  int block_width) 
{

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx + block_width - i  ;
  int b_index_y = block_width - bx -1;

  int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
  int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

#if OPTIM == 1
  __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+2];
#elif OPTIM == 0
  __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
#else
  __shared__  int temp0[BLOCK_SIZE+1][BLOCK_SIZE+1];
  Item item( &(temp0[0][0]) );
  DelayedArray temp( &item );
#endif

  __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE];

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  ref[ty][tx] = referrence[index + cols * ty];

  __syncthreads();

   if (tx == 0)
		  temp[tx][0] = matrix_cuda[index_nw];
 
 
  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  __syncthreads();

  temp[0][tx + 1] = matrix_cuda[index_n];
  
  __syncthreads();
  

  for( int m = 0 ; m < BLOCK_SIZE ; m++){
   
	  if ( tx <= m ){

		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);	  
	  
	  }

	  __syncthreads();
  
    }


 for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){

		  int t_index_x =  tx + BLOCK_SIZE - m ;
		  int t_index_y =  BLOCK_SIZE - tx;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);


	  }

	  __syncthreads();
  }


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

}
