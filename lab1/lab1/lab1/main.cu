//
// Created by Alysha McCullough, Anna Mikhailenko, Jared Adams and on 1/20/23.
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "timing.h"



typedef unsigned long long bignum;


__host__ void getPrimeNumbers(int * arr, bignum size);
__host__ __device__ int isPrime(bignum x);
__host__ void printArray(int * a, int len);
int arrSum (int* result, bignum length);



// CUDA kernel. Each thread takes care of one element of arr
__global__ void calcPrime( int *arr, unsigned long long size)
{
    // Get our global thread ID
    unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num = (2*(id+1)) - 1;

    // Make sure we do not go out of bounds
    if (num < size) {
        if(num > 2) {
            arr[num] = isPrime(num);
        }

    }

}



__host__ void getPrimeNumbers(int *arr, bignum size){

    int nextOdd;
    bignum i = 1;

    for(; ((2*i)-1) < size; i++){
        nextOdd = (2*i) -1;
        arr[nextOdd] = isPrime(nextOdd);
    }
}

__host__ __device__ int isPrime(bignum x){
    if(x==1){return 0;}
    if (x%2 == 0 && x > 2) {return 0;}

    bignum i;
    bignum lim = (bignum) sqrt((float) x) + 1;

    for(i=2; i<lim; i++){
        if ( x % i == 0)
            return 0;
    }


    return 1;
}

__host__ void printArray(int *a, int len){

    int i;

    for(i=0; i<len; i++){
        //printf("i: %d \n", i);
        if (i % 5 == 0) {puts("");}
        printf("%d. %d, ", i, a[i]);

    }

}
int arrSum (int* result, bignum length)
{
    int i, s = 0;
    for( i = 0; i < length; i ++ )
        s += result[i];

    return s;
}


int main (int argc, const char * argv[]) {

    int MAXBLOCKSIZE = 1024;

    if(argc < 2)
    {
        printf("Usage: prime upbound\n");
        exit(-1);
    }

    //Convert input into a bignum(unsigned long long) int
    bignum N = (bignum) atoi(argv[1]);
    int blockSize = (int ) atoi(argv[2]);
    bignum gridSize = (bignum) ceil((float) N / 2.0 / blockSize);


    if(N <= 0)
    {
        printf("Usage: prime upbound, you input invalid upbound number!\n");
        exit(-1);
    }
    if(blockSize <= 0 || blockSize > MAXBLOCKSIZE){
        printf("Usage: upper-bound block-size, your block-size number is invalid!\n");
        exit(-1);
    }

    double now, then, scost, pcost;
    int *h_results, *d_results, sum;
    bignum arrSize = ((N+1)*sizeof(int ));

    //Initialize CPU memory
    h_results = (int *) calloc(N+1, sizeof(int));



    printf("Find all prime numbers in the range of 0 to %llu... \n\n", N);
    then = currentTime();

    //allocate GPU memory
    cudaError_t s = cudaMalloc((void **) &d_results, arrSize);
    s = cudaMemset(d_results, 0, arrSize);

    if (s != cudaSuccess) { printf("Malloc Error!!!!!\n");}

    calcPrime<<<gridSize, blockSize>>>(d_results, N+1);


    s = cudaMemcpy(h_results, d_results, arrSize, cudaMemcpyDeviceToHost);

    if (s != cudaSuccess) { printf("Memcpy_dh Error!!!!!\n");}

    //to avoid parallel condition later
    h_results[0] = 0;
    h_results[2] = 1;

    //printArray(h_results, N+1);
    //printf("\n\n");

    //parallel results
    now = currentTime();
    pcost = now - then;
    sum = arrSum(h_results, N + 1);
    printf("GPU code execution time in seconds is: %lf\n", pcost);
    printf("Total number of primes found in the GPU in that range is: %d.\n\n", sum);


    //Resetting array to 0 at 3 and onward
    for (int z = 3; z < N; z++) {
        h_results[z] = 0;

    }

    then = currentTime();
    getPrimeNumbers(h_results, N);
    now = currentTime();
    scost = now - then;
    printf("%%%%%% Serial code (CPU) executiontime in second is %lf\n", scost);
    printf("Total number of primes in that range is: %d.\n\n", arrSum(h_results, N + 1));

    printf("%%%%%% The speedup(SerialTimeCost / GPUTimeCost) is %lf\n", scost / pcost);





    cudaFree(d_results);
    free(h_results);

    return 0;

}





