
/*
 * Created by:
 * Alysha McCullough
 * Jared Adams
 * Anna Mikhailenko
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCKSIZE 8 // Number of threads in each thread block

// CUDA kernel. Each thread takes care of one element of a
__global__ void adj_diff( float *result, float *input, int n )
{
    int tx = threadIdx.x;
// allocate a __shared__ array, one element per thread
    __shared__ int s_data[BLOCKSIZE];
// each thread reads one element to s_data
    unsigned int i = blockDim.x * blockIdx.x + tx;
    if( i < n ){
        s_data[tx] = input[i];
    }
// avoid race condition: ensure all loads
// complete before continuing
    __syncthreads();
//continued on next slide
//...
//...
    if(tx >= 0 && tx < BLOCKSIZE-1)
        result[i] = s_data[tx+1] - s_data[tx];
    else if(i > 0 && i < n-1)
    {
// handle thread block boundary
        result[i] = input[i+1] - s_data[tx];
    }
}

int main( int argc, char* argv[] )
{
    // Size of vectors
    int i;
    float input[] = {4, 5, 6, 7, 19, 10, 0, 4, 2, 3, 1, 7, 9, 11, 45, 23, 99, 29};
    int n = sizeof(input) / sizeof(float); //careful, this usage only works with statically allocated arrays, NOT dynamic arrays


    // Host input vectors
    float *h_in = input;
    //Host output vector
    float *h_out = (float *) malloc((n - 1) * sizeof(float));

    // Device input vectors
    float *d_in;;
    //Device output vector
    float *d_out;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(float);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes - sizeof(float));

    // Copy host data to device
    cudaMemcpy( d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // TODO: setup the blocksize and gridsize and launch the kernel below.

    // Number of threads in each thread block
    dim3 dimBlock(BLOCKSIZE, 1, 1);

    // Number of thread blocks in grid
    //bignum gridSize = (bignum) ceil((float) N / 2.0 / blockSize);
    int gridSize = (int ) ceil((float) n/ BLOCKSIZE);
    dim3 dimGrid(gridSize, 1, 1);



    // Execute the kernel
    //calcPrime<<<gridSize, blockSize>>>(d_results, N+1);
    adj_diff<<<dimGrid, dimBlock, BLOCKSIZE>>>(d_out, d_in, n);


    // Copy array back to host
    cudaMemcpy( h_out, d_out, bytes - sizeof(float), cudaMemcpyDeviceToHost );

    // Show the result
    printf("The original array is: ");
    for(i = 0; i < n; i ++)
        printf("%4.0f,", h_in[i] );

    printf("\n\nThe diff     array is: ");
    for(i = 0; i < n - 1; i++)
        printf("%4.0f,", h_out[i] );
    puts("");

    // Release device memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Release host memory
    free(h_out);

    return 0;
}

