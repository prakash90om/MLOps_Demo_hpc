#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>



#define STB_IMAGE_IMPLEMENTATION
#include "../../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../include/stb_image_write.h"


__global__ void image_rotate(unsigned char* rotate_img,unsigned char* img,int height,int width)
{
		int col = threadIdx.x + blockDim.x * blockIdx.x;
		int row = threadIdx.y + blockDim.y * blockIdx.y;

    double degree =20.0;
    double radians = (degree * 3.1415926)/180.0;
    double sin_rad = sin(-radians);
    double cos_rad = cos(-radians);

    double row_center = 0.5*(height);
    double col_center = 0.5*(width);

    double a = row - row_center;
    double b = col - col_center ;

    int row_orig = (int)(a * cos_rad - b*sin_rad + row_center);
    int col_orig = (int)(a * sin_rad + b*cos_rad + col_center);

		if(row < height && col < width )
		{	
      if(row_orig >=0 && row_orig <height && col_orig>=0 && col_orig < width)
      {
        rotate_img[3 * (row * width + col) + 0] = img[3 * (row_orig * width + col_orig) + 0];
        rotate_img[3 * (row * width + col) + 1] = img[3 * (row_orig * width + col_orig) + 1];
        rotate_img[3 * (row * width + col) + 2] = img[3 * (row_orig * width + col_orig) + 2];
      }
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

	unsigned char *rotate_img = (unsigned char*)malloc(img_size);
	if(rotate_img == NULL) {
		printf("Unable to allocate memory for the rotate image.\n");
		exit(1);
	}	


	unsigned char *d_img =NULL;
	unsigned char *d_rotate_img=NULL;
	
  cudaMalloc(( void **)&d_img, sizeof(unsigned char)*img_size);
  cudaMalloc(( void **)&d_rotate_img, sizeof(unsigned char)*img_size);

  cudaMemcpy(d_img, img, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_rotate_img, rotate_img, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);
 
  int NUM_THREADS = 32;
  dim3 threadsPerBlock(NUM_THREADS, NUM_THREADS);

  dim3 blocksPerGrid(ceil(double(width/NUM_THREADS)), ceil(double(height/NUM_THREADS)));
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  image_rotate<<<blocksPerGrid,threadsPerBlock>>>(d_rotate_img,d_img,height,width);
  
  cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
  
    printf("Time taken - Image Rotate: %f\n",time_taken);
  cudaMemcpy(rotate_img, d_rotate_img, sizeof(unsigned char)*img_size, cudaMemcpyDeviceToHost);

  cudaFree(d_img);
  cudaFree(d_rotate_img);
  stbi_write_jpg("augmented_images/image_rotate.jpg", width, height, channels, rotate_img, 100);

}
