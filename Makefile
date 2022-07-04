all:
	g++ -fopenmp src/OpenMP/horiz_vert_flip_soln.cpp
	nvcc src/Cuda/horizontal_flip.cu
omp:
	g++ -fopenmp src/OpenMP/horiz_vert_flip_soln.cpp
cuda:
	nvcc src/CUDA/image_rotate.cu
	nvcc src/CUDA/horizontal_flip.cu
	nvcc src/CUDA/image_rotate.cu
	nvcc src/CUDA/image_scaling_down.cu
	nvcc src/CUDA/image_scaling_up.cu
	nvcc src/lodepng/CudaImageFlipHorizontal.cu
	nvcc src/lodepng/CudaImageFlipVertical.cu

clean:
	$(RM) a.out
