all:
	g++ -fopenmp src/OpenMP/horiz_vert_flip_soln.cpp
	nvcc src/Cuda/horizontal_flip.cu
omp:
	g++ -fopenmp src/OpenMP/horiz_vert_flip_soln.cpp
cuda:
	nvcc src/Cuda/horizontal_flip.cu
clean:
	$(RM) a.out
