#include <stdio.h>
#include <stdlib.h>
 #include <cuda.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../include/stb_image_write.h"


__global__ void scale_image(unsigned char* scale_img,unsigned char* img,int height,int width,int scale_factor)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < height*scale_factor && col < width*scale_factor )
	{	
		scale_img[3 * (row * width * scale_factor + col) + 0] = img[3 * (row / scale_factor * width  + col / scale_factor) + 0];
		scale_img[3 * (row * width * scale_factor + col) + 1] = img[3 * (row / scale_factor * width  + col / scale_factor) + 1];
		scale_img[3 * (row * width * scale_factor + col) + 2] = img[3 *(row / scale_factor * width  + col / scale_factor) + 2];
	}
}
 
int main(void) {
	int width, height, channels;
	int original_channels;
	const char *fname = "high_res_images/high_res.jpg";
	stbi_info(fname, &width, &height, &channels);

	unsigned char *img = stbi_load(fname, &width, &height, &original_channels, channels);
	if(img == NULL) {
		printf("Error in loading the image\n");
		exit(1);
	}
	printf("Loaded image with a width of %dpx, a height of %dpx , %d Originals channels and %d channels\n", width, height, original_channels, channels);

	
	size_t img_size = width * height * channels;
    int scale_factor = 4;

	unsigned char *scale_img = (unsigned char*)malloc(img_size*scale_factor*scale_factor);
	if(scale_img == NULL) {
		printf("Unable to allocate memory for the rotate image.\n");
		exit(1);
	}	


	unsigned char *d_img =NULL;
	unsigned char *d_scale_img=NULL;
	
	cudaMalloc(( void **)&d_img, sizeof(unsigned char)*img_size);
	cudaMalloc(( void **)&d_scale_img, sizeof(unsigned char)*img_size*scale_factor*scale_factor);

	cudaMemcpy(d_img, img, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_scale_img, scale_img, sizeof(unsigned char)*img_size*scale_factor*scale_factor, cudaMemcpyHostToDevice);
	
	int NUM_THREADS = 4;
	dim3 threadsPerBlock(NUM_THREADS, NUM_THREADS);

	dim3 blocksPerGrid(ceil(double((width*scale_factor)/NUM_THREADS)), ceil(double((height*scale_factor)/NUM_THREADS)));
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
	scale_image<<<blocksPerGrid,threadsPerBlock>>>(d_scale_img,d_img,height,width,scale_factor);
	
	cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
  
    printf("Time taken - Image scaling up: %f\n",time_taken);
	cudaMemcpy(scale_img, d_scale_img, sizeof(unsigned char)*img_size*scale_factor*scale_factor, cudaMemcpyDeviceToHost);

	cudaFree(d_img);
	cudaFree(d_scale_img);
	stbi_write_jpg("augmented_images/image_scaling_up.jpg", width*scale_factor, height*scale_factor, channels, scale_img, 100);

}
