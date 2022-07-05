#include <stdio.h>
#include <stdlib.h>
 #include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../include/stb_image_write.h"



__global__ void vertical_flip_image(unsigned char* vertical_flip_img,unsigned char* img,int height,int width)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < height && col < width )
	{	
		vertical_flip_img[3 * (row * width + col) + 0] = img[3 * ((height-1-row) * width + col) + 0];
		vertical_flip_img[3 * (row * width + col) + 1] = img[3 * ((height-1-row) * width + col) + 1];
		vertical_flip_img[3 * (row * width + col) + 2] = img[3 * ((height-1-row) * width + col) + 2];
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

	unsigned char *vertical_flip_img = (unsigned char*)malloc(img_size);
	if(vertical_flip_img == NULL) {
		printf("Unable to allocate memory for the rotate image.\n");
		exit(1);
	}	


	unsigned char *d_img =NULL;
	unsigned char *d_vertical_flip_img=NULL;
	
	cudaMalloc(( void **)&d_img, sizeof(unsigned char)*img_size);
	cudaMalloc(( void **)&d_vertical_flip_img, sizeof(unsigned char)*img_size);

	cudaMemcpy(d_img, img, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertical_flip_img, vertical_flip_img, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);
	
	int NUM_THREADS = 32;
	dim3 threadsPerBlock(NUM_THREADS, NUM_THREADS);

	dim3 blocksPerGrid(ceil(double(width/NUM_THREADS)), ceil(double(height/NUM_THREADS)));
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
	vertical_flip_image<<<blocksPerGrid,threadsPerBlock>>>(d_vertical_flip_img,d_img,height,width);
	
	cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
  
    printf("Time taken - Vertical Flip: %f\n",time_taken);
	cudaMemcpy(vertical_flip_img, d_vertical_flip_img, sizeof(unsigned char)*img_size, cudaMemcpyDeviceToHost);

	cudaFree(d_img);
	cudaFree(d_vertical_flip_img);
	stbi_write_jpg("augmented_images/vertical_flipped_img.jpg", width, height, channels, vertical_flip_img, 100);

}
