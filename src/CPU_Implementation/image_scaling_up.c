#include <stdio.h>
#include <stdlib.h>
#include <time.h>
 
#define STB_IMAGE_IMPLEMENTATION
#include "../../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../include/stb_image_write.h"

// ...

void Scale_image(unsigned char *scale_img,unsigned char *img,int height,int width,int scale_factor)
{
  	for (int i = 0; i < scale_factor * height; i++) {
			for (int k = 0; k < scale_factor * width; k++) {
					scale_img[3 * (i * width *scale_factor + k) + 0] = img[3 * (i/scale_factor * width + k/scale_factor) + 0];
					scale_img[3 * (i * width *scale_factor + k) + 1] = img[3 * (i/scale_factor * width + k/scale_factor) + 1];
					scale_img[3 * (i * width *scale_factor + k) + 2] =  img[3 *(i/scale_factor * width + k/scale_factor) + 2];
			}
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
    int scale_factor =2;
	unsigned char *scale_img = (unsigned char*)malloc(img_size*scale_factor*scale_factor);
	if(scale_img == NULL) {
		printf("Unable to allocate memory for the rotate image.\n");
		exit(1);
	}	

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    Scale_image(scale_img,img,height,width,scale_factor);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
  
    printf("Time taken - Image Scaling up:%f\n",time_taken);

	stbi_write_jpg("augmented_images/image_cpu.jpg", width*scale_factor, height*scale_factor, channels, scale_img, 100);

	// ...
}
