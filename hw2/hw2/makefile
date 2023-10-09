mamyPaint: main.o pgmUtility.o pgmProcess.o
	nvcc -arch=sm_52 -I. -o myPaint main.o pgmUtility.o pgmProcess.o

main.o: main.c
	g++ -c -I. -o main.o main.c

pgmUtility.o: pgmUtility.cu pgmUtility.cuh
	nvcc -arch=sm_52 -c pgmUtility.cu -I.

pgmProcess.o: pgmProcess.cu pgmProcess.cuh
	nvcc -arch=sm_52 -c pgmProcess.cu -I.

clean:
	rm -r *.o myPaint
