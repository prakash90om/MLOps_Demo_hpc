omp:
	g++ -fopenmp src/OpenMP/horiz_vert_flip_soln.cpp

clean:
	$(RM) a.out
