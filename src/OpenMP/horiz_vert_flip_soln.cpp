#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../include/stb_image_write.h"

// ...

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("Provide the Argument for num of thread.\n");
        exit(0);
    }

    int MAX = std::atoi(argv[1]);
	int width, height, channels;
	struct timeval t1, t2;
	int original_channels;
	const char *fname = "high_res.jpg";
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

	omp_set_num_threads(MAX);
	gettimeofday(&t1, 0);

	#pragma omp parallel for
	//Flip Horizontally the given image
	for (int i = 0; i < width; i++) {
		for (int k = 0; k < height; k++) {
			int j = width-1-i;
			int l = height-1-k;
			flip_img[3 * (k * width + i) + 0] = img[3 * (k * width + j) + 0];
			flip_img[3 * (k * width + i) + 1] = img[3 * (k * width + j) + 1];
			flip_img[3 * (k * width + i) + 2] = img[3 * (k * width + j) + 2];
		}
	}

	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Horizontal Flip time taken: %f milisec\n", time);

	stbi_write_jpg("horizontal_flipped_img.jpg", width, height, channels, flip_img, 100);

	gettimeofday(&t1,0);

	#pragma omp parallel for
	//Flip Vertically the given image
	for (int i = 0; i < height; i++) {
		for (int k = 0; k < width; k++) {
			int j = height-1-i;
			int l = width-1-k;
			flip_img[3 * (i * width + k) + 0] = img[3 * (j * width + k) + 0];
			flip_img[3 * (i * width + k) + 1] = img[3 * (j * width + k) + 1];
			flip_img[3 * (i * width + k) + 2] = img[3 * (j * width + k) + 2];
		}
	}

	gettimeofday(&t2, 0);
	time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("Verticaltal Flip time taken: %f milisec\n", time);

	stbi_write_jpg("vertical_flipped_img.jpg", width, height, channels, flip_img, 100);

	// ...
}
