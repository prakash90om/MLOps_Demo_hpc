# Build folder
BUILDDIR = ./build

all:	builddir
	g++ -fopenmp src/OpenMP/horiz_vert_flip_soln.cpp -o $(BUILDDIR)/horiz_vert_flip_soln.o
	nvcc src/CUDA/horizontal_flip.cu -o $(BUILDDIR)/horizontal_flip.o
	nvcc src/CUDA/image_rotate.cu -o $(BUILDDIR)/image_rotate.o
	nvcc src/CUDA/image_scaling_down.cu -o $(BUILDDIR)/image_scaling_down.o
	nvcc src/CUDA/image_scaling_up.cu -o $(BUILDDIR)/image_scaling_up.o
	g++ src/lodepng/ImageFlipVerticalHorizontal.cpp -o $(BUILDDIR)/ImageFlipVerticalHorizontal.o
	nvcc src/lodepng/CudaImageFlipHorizontal.cu -o $(BUILDDIR)/CudaImageFlipHorizontal.o
	nvcc src/lodepng/CudaImageFlipVertical.cu -o $(BUILDDIR)/CudaImageFlipVertical.o



omp:	builddir
	g++ -fopenmp src/OpenMP/horiz_vert_flip_soln.cpp -o $(BUILDDIR)/horiz_vert_flip_soln.o

cuda:	builddir
	nvcc src/CUDA/horizontal_flip.cu -o $(BUILDDIR)/horizontal_flip.o
	nvcc src/CUDA/image_rotate.cu -o $(BUILDDIR)/image_rotate.o
	nvcc src/CUDA/image_scaling_down.cu -o $(BUILDDIR)/image_scaling_down.o
	nvcc src/CUDA/image_scaling_up.cu -o $(BUILDDIR)/image_scaling_up.o

lodepng:	builddir
	g++ src/lodepng/ImageFlipVerticalHorizontal.cpp -o $(BUILDDIR)/ImageFlipVerticalHorizontal.o
	nvcc src/lodepng/CudaImageFlipHorizontal.cu -o $(BUILDDIR)/CudaImageFlipHorizontal.o
	nvcc src/lodepng/CudaImageFlipVertical.cu -o $(BUILDDIR)/CudaImageFlipVertical.o

clean:
	rm -rf $(BUILDDIR)

builddir:
	mkdir -p $(BUILDDIR)
