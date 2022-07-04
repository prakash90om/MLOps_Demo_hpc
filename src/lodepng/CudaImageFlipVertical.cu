/////////////////////////////////////////////////////////
//  CUDA CPP Code to Flip an Image Vertically
//  Load the image in InputImage.png
/////////////////////////////////////////////////////////

#include "../../include/lodepng.cpp"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include <algorithm>
#include<vector>

using namespace std;
using namespace std::chrono;


int BLOCKDIM;
int GRIDDIM;

__global__ void flip_image_pixel (unsigned char* in_pixel,   unsigned char* out_pixel, int image_size){  
  
  int divby3 = (threadIdx.x + blockDim.x * blockIdx.x) % 3;

  if (divby3 == 0)
    out_pixel[threadIdx.x + blockDim.x * blockIdx.x] = in_pixel[image_size - (threadIdx.x + blockDim.x * blockIdx.x + 2)];
  else if (divby3 == 1)
    out_pixel[threadIdx.x + blockDim.x * blockIdx.x] = in_pixel[image_size - (threadIdx.x + blockDim.x * blockIdx.x + 1)];
  else
    out_pixel[threadIdx.x + blockDim.x * blockIdx.x] = in_pixel[image_size - (threadIdx.x + blockDim.x * blockIdx.x)];
}

// Driver code
int main()
{
    std::vector<unsigned char> in_image;
    unsigned int width, height;
    const char* input_file = "../../res/InputImage.png";
    const char* output_file = "../../res/GPUFlipVertical.png";
    const char* output_file_test = "Denarys_test.png";

    auto start_chrono = high_resolution_clock::now();

    // Load the data
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    unsigned long in_image_size = in_image.size();


    // Prepare the data
    //unsigned char* input_image = new unsigned char[(in_image_size*3)/4];
    //unsigned char* output_image = new unsigned char[(in_image_size*3)/4];
    unsigned char* input_image = new unsigned char[(in_image_size)];
    unsigned char* output_image = new unsigned char[(in_image_size)];
    
    unsigned char* host_in_img;
    unsigned char*  host_out_img;
    unsigned char* device_in_img;
    unsigned char*  device_out_img;

    // allocation on host
    host_in_img = (unsigned char *)malloc(sizeof(unsigned char)*in_image_size);
    host_out_img = (unsigned char *)malloc(sizeof(unsigned char)*in_image_size);

    int where = 0;
    for(int i = 0; i < in_image_size; ++i) {
        host_in_img[where] = in_image.at(i);
        where++;
    } 

    /*
    error = lodepng::encode(output_file_test, host_in_img, width, height);

    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    else cout <<"SUCCESS 2"<<endl;  */

    // allocation on device
    cudaMalloc(&device_in_img, sizeof(unsigned char)*in_image_size);
    cudaMalloc(&device_out_img, sizeof(unsigned char)*in_image_size);

    cudaMemcpy(device_in_img, host_in_img, sizeof(unsigned char)*in_image_size, cudaMemcpyHostToDevice);

    int NUM_THREADS = 32;
    dim3 threadsPerBlock(NUM_THREADS);

    dim3 blocksPerGrid(ceil(double(in_image_size)/(NUM_THREADS)));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    flip_image_pixel<<<blocksPerGrid,threadsPerBlock>>>(device_in_img,device_out_img,in_image_size);
    
    //Calculating Time for Flip alone
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(host_out_img, device_out_img, sizeof(unsigned char)*in_image_size, cudaMemcpyDeviceToHost);


    error = lodepng::encode(output_file, host_out_img, width, height);

    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Time taken to Flip the image on GPU (Computation time alone): "<<milliseconds<<"ms" <<endl;

    auto stop_chrono = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop_chrono - start_chrono);

}