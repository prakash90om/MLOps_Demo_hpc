all:
	g++ -fopenmp horiz_vert_flip_soln.cpp

clean:
	$(RM) a.out
