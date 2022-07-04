omp:
	g++ -fopenmp src/OpenMP/horiz_vert_flip_soln.cpp
	nvcc src/Cuda/horizontal_flip.cu
clean:
	$(RM) a.out
