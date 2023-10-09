#include <stdio.h>
#include "kernel1.h"


extern  __shared__  float sdata[];

////////////////////////////////////////////////////////////////////////////////
//! Weighted Jacobi Iteration
//! @param g_dataA  input data in global memory
//! @param g_dataB  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void k1( float* g_dataA, float* g_dataB, int floatpitch, int width)
{
    extern __shared__ float s_data[];
    //Write this kernel to achieve the same output as the provided k0, but you will have to use
    // shared memory.

    // global thread(data) row index
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    i = i + 1; //because the edge of the data is not processed

    // global thread(data) column index
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    j = j + 1; //because the edge of the data is not processed

    int threadID = threadIdx.x;

    int s_rowwidth = blockDim.x + 2;

    // Index's
    int g_ind_0 = (i-1) * floatpitch +  j;
    int g_ind_1 = i * floatpitch + j;
    int g_ind_2 = (i+1) * floatpitch +  j;

    // -- Shared
    int s_ind_0 = threadID + 1 + (s_rowwidth * 0);
    int s_ind_1 = threadID + 1 + (s_rowwidth * 1);
    int s_ind_2 = threadID + 1 + (s_rowwidth * 2);
    int s_index_result = threadID + 1 + (s_rowwidth * 3);

    //Check the boundary.
    if( i >= width - 1|| j >= width || i < 1 || j < 1 ) {
    }
    else {
        // Copy 3 row for this col
        s_data[s_ind_0] = g_dataA[g_ind_0];
        s_data[s_ind_1] = g_dataA[g_ind_1];
        s_data[s_ind_2] = g_dataA[g_ind_2];
        // If tid = 0 run top copys 3 for col before
        if (threadID == 0) {
            s_data[threadID + (s_rowwidth * 0)] = g_dataA[(i-1) * floatpitch +  (j-1)];
            s_data[threadID + (s_rowwidth * 1)] = g_dataA[i * floatpitch +  (j-1)];
            s_data[threadID + (s_rowwidth * 2)] = g_dataA[(i+1) * floatpitch +  (j-1)];
        }
            // If tid = rowwidth run top copys 3 for col after
        else if (threadID == blockDim.x -1) {
            s_data[threadID + 2 + (s_rowwidth * 0)] = g_dataA[(i-1) * floatpitch +  (j)];
            s_data[threadID + 2 + (s_rowwidth * 1)] = g_dataA[i * floatpitch +  (j)];
            s_data[threadID + 2 + (s_rowwidth * 2)] = g_dataA[(i+1) * floatpitch +  (j)];
        }
    }
    __syncthreads();

    if( i >= width - 1|| j >= width - 1 || i < 1 || j < 1 ) {
        // Do Nothing
    } else {
        int s_i = 1;
        int s_j = threadID + 1;

        //Calculate our cell's result using a LOCAL variable, then write that variable to the result
        s_data[s_index_result] = (
                                         0.2f * s_data[s_i * s_rowwidth + s_j] +               //itself
                                         0.1f * s_data[(s_i-1) * s_rowwidth +  s_j   ] +       //N
                                         0.1f * s_data[(s_i-1) * s_rowwidth + (s_j+1)] +       //NE
                                         0.1f * s_data[ s_i    * s_rowwidth + (s_j+1)] +       //E
                                         0.1f * s_data[(s_i+1) * s_rowwidth + (s_j+1)] +       //SE
                                         0.1f * s_data[(s_i+1) * s_rowwidth +  s_j   ] +       //S
                                         0.1f * s_data[(s_i+1) * s_rowwidth + (s_j-1)] +       //SW
                                         0.1f * s_data[ s_i    * s_rowwidth + (s_j-1)] +       //W
                                         0.1f * s_data[(s_i-1) * s_rowwidth + (s_j-1)]         //NW
                                 ) * 0.95f;
    }

    __syncthreads();

    if( i >= width - 1|| j >= width - 1 || i < 1 || j < 1 ) {
        return;
    }

    g_dataB[i * floatpitch + j] = s_data[s_index_result];

}
