#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../include/stb_image_write.h"

// ...

void horizontal_flip_image(unsigned char *flip_img,unsigned char *img,int height,int width)
{
	for (int i = 0, j = width-1; i < width; i++, j--) {
		for (int k = 0, l = height-1; k < height; k++, l--) { 			
			flip_img[3 * (k * width + i) + 0] = img[3 * (k * width + j) + 0];
			flip_img[3 * (k * width + i) + 1] = img[3 * (k * width + j) + 1];
			flip_img[3 * (k * width + i) + 2] = img[3 * (k * width + j) + 2];
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

	unsigned char *flip_img = (unsigned char*)malloc(img_size);
	if(flip_img == NULL) {
		printf("Unable to allocate memory for the rotate image.\n");
		exit(1);
	}	
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    horizontal_flip_image(flip_img,img,height,width);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken;
    time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
  
    printf("Time taken - Horizontal Flip: %f\n",time_taken);
	
	stbi_write_jpg("augmented_images/horizontal_flipped_img.jpg", width, height, channels, flip_img, 100);
	// ...
}
