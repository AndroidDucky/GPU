
//  Created by Alysha McCullough, Anna Mikhailenko, Jared Adams


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "pgmUtility.cuh"
__host__ __device__ float distance( int p1[], int p2[] );
// Implement or define each function prototypes listed in pgmUtility.h file.
// NOTE: Please follow the instructions stated in the write-up regarding the interface of the functions.
// NOTE: You might have to change the name of this file into pgmUtility.cu if needed.


__global__ void circle(int *pix, int dimx, int dimy, int centerX, int centerY, int radius){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //int idx = blockIdx.y * blockDim.x + blockIdx.x;
    int idx = y * dimx + x;

    if((x < dimx && y < dimy) ){
        //int pt1[] = {x,y};
        //int pt2[] = {centerX,centerY};

        if(/*distance(pt1,pt2)*/(x - centerX)*(x - centerX) + (y - centerY)*(y - centerY) < radius*radius){
            //idx = blockIdx.y * blockDim.x + blockIdx.x;
            pix[idx] = 0;
        }
    }
}

int pgmDrawCircle( int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header ){
    int dimx = numCols;
    int dimy = numRows;

    dim3 grid, block;
    int num_bytes = dimx*dimy*sizeof(int);
    int *d_a;
    cudaMalloc((void **)&d_a, num_bytes);

    if(0 ==d_a) {
        printf("error");
        return 0;
    }
    cudaMemset(d_a,0,num_bytes);


    cudaMemcpy( d_a, pixels, num_bytes, cudaMemcpyHostToDevice);
    block.x = 4;
    block.y = 4;//change the 4 threads to more

    grid.x = ceil((float ) dimx / block.x);
    grid.y = ceil((float) dimy / block.y);



    //printf("dimx: %d, dimy: %d, centerCol: %d, centerRow: %d, radius: %d\n", dimx, dimy, centerCol, centerRow, radius);
    circle<<<grid, block>>>(d_a, dimx, dimy, centerCol, centerRow, radius);
    cudaMemcpy(pixels,d_a, num_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    return 0;
}

__global__ void edgeKernel(int *d_a, int dimx, int dimy,int edge )
{
    int ix   = blockIdx.x*blockDim.x + threadIdx.x;
    int iy   = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*dimx + ix;
    if(ix<edge||ix>=(dimx-edge)||iy<edge||iy>=(dimy-edge)){
        d_a[idx]=0;
    }
}


__global__ void lineKernel(int *pixels, int dimx, int dimy, int yStart, int xStart, int yEnd,  int xEnd )
{
    int ix   = blockIdx.x*blockDim.x + threadIdx.x;
    int iy   = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*dimx + ix;


    if(ix < dimx && iy < dimy){

        //distances
        float startToEndLine= sqrtf((xEnd-xStart)*(xEnd-xStart)+(yEnd-yStart)*(yEnd-yStart));
        float startToCenterLine = sqrtf((ix-xStart)*(ix-xStart)+(iy-yStart)*(iy-yStart));
        float centerToEndLine = sqrtf((xEnd-ix)*(xEnd-ix)+(yEnd-iy)*(yEnd-iy));
        if(fabs(startToEndLine-(startToCenterLine+centerToEndLine))<=.0009f){
            pixels[idx]=0;
        }
    }
}


int * pgmRead( char **header, int *numRows, int *numCols, FILE *in )
{
    int i;


    // read in header of the image first
    for( i = 0; i < rowsInHeader; i ++)
    {
        if ( header[i] == NULL )
        {
            return NULL;
        }
        if( fgets( header[i], maxSizeHeadRow, in ) == NULL )
        {
            return NULL;
        }
    }
    // extract rows of pixels and columns of pixels
    sscanf( header[rowsInHeader - 2], "%d %d", numCols, numRows );  // in pgm the first number is # of cols

    // Now we can intialize the pixel of 2D array, allocating memory
    int *pixels = (int*) malloc((*numRows)*(*numCols)*sizeof(int));


    // read in all pixels into the pixels array.
    for( i = 0; i < (*numRows)*(*numCols); i ++ )
        if ( fscanf(in, "%d ",  &pixels[i]) < 0 ) {
            return NULL;
        }

    return pixels;
}

int pgmWrite( const char **header, const int *pixels, int numRows, int numCols, FILE *out )
{
    int i, j;

    // write the header
    for ( i = 0; i < rowsInHeader; i ++ )
    {
        fprintf(out, "%s", *( header + i ) );
    }

    // write the pixels
    for( i = 0; i < numRows*numCols; i ++ )
    {
        if ( j < numCols - 1 )
            fprintf(out, "%d ", pixels[i]);
        else
            fprintf(out, "%d\n", pixels[i]);
    }
    return 0;
}
int pgmDrawLine( int *pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col ){
    int dimx = numCols;
    int dimy = numRows;
    dim3 grid, block;
    int num_bytes = dimx*dimy*sizeof(int);
    int *d_a;
    cudaMalloc((void **)&d_a, num_bytes);

    if(0 == d_a) {
        printf("error");
        return 0;
    }
    cudaMemset(d_a,0,num_bytes);
    cudaMemcpy( d_a, pixels, num_bytes, cudaMemcpyHostToDevice);
    block.x = 4;
    block.y = 4;
    grid.x  = ceil( (float)dimx / block.x );
    grid.y  = ceil( (float)dimy / block.y );

    
    //x:COL    y:ROW
    //      dimx        dimy     p1row       p1col       p2row     p2col
    //printf("dimx : %d, dimy: %d, yStart: %d, xStart: %d, yEnd: %d, xEnd: %d \n", dimx, dimy, p1row, p1col, p2row, p2col);
    
    lineKernel<<<grid, block>>>( d_a, dimx, dimy, p1row, p1col, p2row, p2col);

    cudaMemcpy(pixels,d_a, num_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    return 0;

}

int pgmDrawEdge( int *pixels, int numRows, int numCols, int edgeWidth, char **header ){
    int dimx = numCols;
    int dimy = numRows;
    dim3 grid, block;
    //distance( int p1[], int p2[] );
    int num_bytes = dimx*dimy*sizeof(int);
    int *d_a;
    cudaMalloc((void **)&d_a, num_bytes);

    if(0 ==d_a) {
        printf("error");
        return 0;
    }
    cudaMemset(d_a,0,num_bytes);

    cudaMemcpy( d_a, pixels, num_bytes, cudaMemcpyHostToDevice);
    block.x = 4;
    block.y = 4;
    grid.x  = ceil( (float)dimx / block.x );
    grid.y  = ceil( (float)dimy / block.y );

    edgeKernel<<<grid, block>>>( d_a, dimx, dimy, edgeWidth);

    cudaMemcpy(pixels,d_a, num_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    return 0;
}



__host__ __device__ float distance( int p1[], int p2[] )
{
    return sqrtf((p1[0] - p2[0])*(p1[0] - p2[0]))+ ((p1[1] - p2[1])*(p1[1] - p2[1]));
}







