//
// Created by Alysha McCullough, Anna Mikhailenko, Jared Adams
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "pgmUtility.cuh"


void usage();
// int pgmDrawCircleCPU( int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header );
// int pgmDrawEdgeCPU( int *pixels, int numRows, int numCols, int edgeWidth, char **header );
// int pgmDrawLineCPU( int *pixels, int numRows, int numCols, char **header, int yStart, int xStart, int yEnd, int xEnd );
// double currentTime();



int main(int argc, char *argv[]){

    FILE * fp = NULL;
    FILE * out = NULL;

    char ** header = (char**) malloc( sizeof(char *) * rowsInHeader);
    int i;
    int * pixels = NULL;
    for(i = 0; i < 4; i++){
        header[i] = (char *) malloc (sizeof(char) * maxSizeHeadRow);
    }
    int numRows, numCols;
    int p1y = 0;
    int p1x = 0;
    int p2y = 0;
    int p2x = 0;

    int m, n, l, x, ch;
    //double then, now, time, time2, difference;
    int edgeWidth, circleCenterRow, circleCenterCol, radius;
    char originalImageName[100], newImageFileName[100];
    if(argc != 5 && argc != 7 && argc != 8){
        usage();
        return 1;
    }
    else
    {
        l = strlen( argv[1]);
        
        if(l != 2){
            usage();
            return 1;
        }
        ch = (int)argv[1][1];
        if(ch < 97) {
            ch = ch + 32;
        }
        switch( ch )
        {
            case 'c':
                if(argc != 7){
                    usage();
                    break;
                }
                circleCenterRow = atoi(argv[2]);
                circleCenterCol = atoi(argv[3]);
                radius = atoi(argv[4]);
                strcpy(originalImageName, argv[5]);
                strcpy(newImageFileName, argv[6]);

                fp = fopen(originalImageName, "r");
                if(fp == NULL){
                    usage();
                    return 1;
                }
                out = fopen(newImageFileName, "w");
                if(out == NULL){
                    usage();
                    fclose(fp);
                    return 1;
                }
                pixels = pgmRead(header, &numRows, &numCols, fp);


                //then = currentTime();
                pgmDrawCircle(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header );
                //now = currentTime();
                //time = now - then;

                // then = currentTime();
                // pgmDrawCircleCPU(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header );
                // now = currentTime();
                // time2 = now - then;

                //difference = time - time2;

                pgmWrite((const char **)header, (const int *)pixels, numRows, numCols, out );
                //printf(" Circle: \n Gpu time: %f \n CPU time: %f \n Diffenece: %f \n", time, time2, difference);
                break;

            case 'e':
                if(argc != 5){
                    usage();
                    break;
                }
                edgeWidth = atoi(argv[2]);
                strcpy(originalImageName, argv[3]);
                strcpy(newImageFileName, argv[4]);
                fp = fopen(originalImageName, "r");
                if(fp == NULL){
                    usage();
                    return 1;
                }
                out = fopen(newImageFileName, "w");
                if(out == NULL){
                    usage();
                    fclose(fp);
                    return 1;
                }

                pixels = pgmRead(header, &numRows, &numCols, fp);

                // then = currentTime();
                pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);
                // now = currentTime();
                // time = now - then;
                
                // then = currentTime();
                // pgmDrawEdgeCPU(pixels, numRows, numCols, edgeWidth, header);
                // now = currentTime();
                // time2 = now - then;

                // difference = time - time2;

                pgmWrite((const char **)header, (const int *)pixels, numRows, numCols, out );
                // printf(" Edge: \n Gpu time: %f \n CPU time: %f \n Diffenece: %f \n", time, time2, difference);
                break;

            case 'l':
                if(argc != 8){
                    usage();
                    break;
                }
                p1y = atoi(argv[2]);
                p1x = atoi(argv[3]);

                p2y = atoi(argv[4]);
                p2x = atoi(argv[5]);


                strcpy(originalImageName, argv[6]);
                strcpy(newImageFileName, argv[7]);

                fp = fopen(originalImageName, "r");
                if(fp == NULL){
                    usage();
                    return 1;
                }
                out = fopen(newImageFileName, "w");

                if(out == NULL){
                    usage();
                    fclose(fp);
                    return 1;
                }

                pixels = pgmRead(header, &numRows, &numCols, fp);

                // then = currentTime();
                pgmDrawLine(pixels, numRows, numCols, header, p1y, p1x, p2y, p2x);
                // now = currentTime();
                // time = now - then;
                
                // then = currentTime();
                // pgmDrawLineCPU(pixels, numRows, numCols, header, p1y, p1x, p2y, p2x);
                // now = currentTime();
                // time2 = now - then;

                // difference = time - time2;

                
                // printf(" Line: \n Gpu time: %f \n CPU time: %f \n Diffenece: %f \n", time, time2, difference);

                pgmWrite((const char **)header, (const int *)pixels, numRows, numCols, out );
                break;
        }
    }

    free(pixels);
    i = 0;
    for(;i < rowsInHeader; i++)
        free(header[i]);
    free(header);
    if(out != NULL)
        fclose(out);
    if(fp != NULL)
        fclose(fp);
    return 0;
}

void usage()
{
    printf("Usage:\n -e edgeWidth oldImageFile newImageFile\n -c circleCenterRow circleCenterCol radius oldImageFile newImageFile\n -l p1row p1col p2row p2col oldImageFile newImageFile\n");
}


// int pgmDrawCircleCPU( int *pixels, int numRows, int numCols, int centerRow,
//                    int centerCol, int radius, char **header )
// {
//     int i,j;

//     for(i = 0; i < numRows; i++){
//         for (j = 0; j < numCols; j++){
//             if((j <= numCols && i <= numRows) ){
//                 int pt1[] = {j,i};
//                 int pt2[] = {centerCol,centerRow};

//                 int dist = sqrt((pt1[0] - pt2[0])*(pt1[0] - pt2[0]))+ ((pt1[1] - pt2[1])*(pt1[1] - pt2[1]));

//                 if(dist < radius){
//                     //idx = blockIdx.y * blockDim.x + blockIdx.x;
//                     // M_1D[i * columnWidth + j]  
//                     pixels[i * numCols + j] = 0;
//                 }
//             }
//         }
//     }
//     return 0;
// }

// //---------------------------------------------------------------------------
// int pgmDrawEdgeCPU( int *pixels, int numRows, int numCols, int edgeWidth, char **header ) {
//     int i, j;

//     for (i = 0; i < numRows; i++) {
//         for (j = 0; j < numCols; j++) {
//             if (i < edgeWidth || i >= (numCols - edgeWidth) || j < edgeWidth || j >= (numRows - edgeWidth)) {
//                 pixels[i * numCols + j] = 0;
//             }
//         }
//     }
//     return 0;
// }

// //---------------------------------------------------------------------------

// int pgmDrawLineCPU( int *pixels, int numRows, int numCols, char **header,
//                  int yStart, int xStart, int yEnd, int xEnd ){
//     int i,j;
//     for(i = 0; i < numRows; i++) {
//         for (j = 0; j < numCols; j++) {
//             if(j < numCols && i < numCols){

//                 //distances
//                 float startToEndLine= sqrtf((xEnd-xStart)*(xEnd-xStart)+(yEnd-yStart)*(yEnd-yStart));
//                 float startToCenterLine = sqrtf((j-xStart)*(j-xStart)+(i-yStart)*(i-yStart));
//                 float centerToEndLine = sqrtf((xEnd-j)*(xEnd-j)+(yEnd-i)*(yEnd-i));
//                 if(abs(startToEndLine-(startToCenterLine+centerToEndLine))<=.0009f){
//                     pixels[i * numCols + j]=0;
//                 }
//             }
//         }
//     }
//     return 0;
// }

// double currentTime(){

//     struct timeval now;
//     gettimeofday(&now, NULL);

//     return now.tv_sec + now.tv_usec/1000000.0;
// }


