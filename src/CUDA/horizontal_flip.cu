#include <stdio.h>
#include <stdlib.h>
 #include <cuda.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../include/stb_image_write.h"


__global__ void horizontal_flip_image(unsigned char* horiz_flip_img,unsigned char* img,int height,int width)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;


	if(row < width && col < height )
	{	
		horiz_flip_img[3 * (col * width + row) + 0] = img[3 * (col * width + (width-1-row)) + 0];
		horiz_flip_img[3 * (col * width + row) + 1] = img[3 * (col * width + (width-1-row)) + 1];
		horiz_flip_img[3 * (col * width + row) + 2] = img[3 * (col * width + (width-1-row)) + 2];

	}
}


 
int main(void) {
	int width, height, channels;
	int original_channels;
	const char *fname = "../../res/high_res.jpg";
	stbi_info(fname, &width, &height, &channels);

	unsigned char *img = stbi_load(fname, &width, &height, &original_channels, channels);
	if(img == NULL) {
		printf("Error in loading the image\n");
		exit(1);
	}
	printf("Loaded image with a width of %dpx, a height of %dpx , %d Originals channels and %d channels\n", width, height, original_channels, channels);

	
	size_t img_size = width * height * channels;

	unsigned char *horiz_flip_img = (unsigned char*)malloc(img_size);
	if(horiz_flip_img == NULL) {
		printf("Unable to allocate memory for the rotate image.\n");
		exit(1);
	}	


	unsigned char *d_img =NULL;
	unsigned char *d_horiz_flip_img=NULL;
	
	cudaMalloc(( void **)&d_img, sizeof(unsigned char)*img_size);
	cudaMalloc(( void **)&d_horiz_flip_img, sizeof(unsigned char)*img_size);

	cudaMemcpy(d_img, img, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_horiz_flip_img, horiz_flip_img, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);
	
	int NUM_THREADS = 32;
	dim3 threadsPerBlock(NUM_THREADS, NUM_THREADS);

	dim3 blocksPerGrid(ceil(double(height/NUM_THREADS)), ceil(double(width/NUM_THREADS)));
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

	horizontal_flip_image<<<blocksPerGrid,threadsPerBlock>>>(d_horiz_flip_img,d_img,height,width);
	
	cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
  
    printf("Time taken - Horizontal Flip: %f\n",time_taken);
	cudaMemcpy(horiz_flip_img, d_horiz_flip_img, sizeof(unsigned char)*img_size, cudaMemcpyDeviceToHost);

	cudaFree(d_img);
	cudaFree(d_horiz_flip_img);
	stbi_write_jpg("augmented_images/horizontal_flipped_img.jpg", width, height, channels, horiz_flip_img, 100);

}
