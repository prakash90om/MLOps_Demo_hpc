/////////////////////////////////////////////////////////
//  CUDA CPP Code to Flip an Image Horizontally
//  Load the image in InputImage.png
/////////////////////////////////////////////////////////
#include "../../include/lodepng.cpp"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

using namespace std;

int BLOCKDIM;
int GRIDDIM;

__global__ void flip_image_pixel (unsigned char* in_pixel,   unsigned char* out_pixel, unsigned long image_size, unsigned long width){
    
  unsigned long i = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned long j = threadIdx.y + blockDim.y * blockIdx.y;

  if(j < (4*width) && i < (image_size/(4*width)))
  {
    out_pixel[(i*4*width)+j] =in_pixel[(i+1)*(4*width) - 1 - j];
  }

}

// Driver code
int main()
{
    std::vector<unsigned char> in_image;
    unsigned int width, height;
    const char* input_file = "../../res/InputImage.png";
    const char* output_file = "../../res/GPUHorizontalFlip.png";

    // Load the data
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    unsigned long in_image_size = in_image.size();

    //cout << height << " " << width << " " << in_image.size() << endl;


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

    //Converting the in_image char vector into unsigned char
    int where = 0;
    for(int i = 0; i < in_image_size; ++i) {
        //if (i % 4 != 0)
        //{
          host_in_img[where] = in_image.at(i);
          where++;
        //}
    } 

    // allocation on device
    cudaMalloc(&device_in_img, sizeof(unsigned char)*in_image_size);
    cudaMalloc(&device_out_img, sizeof(unsigned char)*in_image_size);

    cudaMemcpy(device_in_img, host_in_img, sizeof(unsigned char)*in_image_size, cudaMemcpyHostToDevice);

    int NUM_THREADS = 32;
    
    dim3 threadsPerBlock(NUM_THREADS,NUM_THREADS);

    dim3 blocksPerGrid(ceil(double(height)/(NUM_THREADS)),ceil(double(in_image_size/height)/(NUM_THREADS)));
    
    //Calculating Time for Flip alone
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);


    flip_image_pixel<<<blocksPerGrid,threadsPerBlock>>>(device_in_img,device_out_img,in_image_size,width);
    
    cudaDeviceSynchronize();
  
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(host_out_img, device_out_img, sizeof(unsigned char)*in_image_size, cudaMemcpyDeviceToHost);

    error = lodepng::encode(output_file, host_out_img, width, height);

    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
   
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Time taken to Flip the image on GPU (Computation time alone): "<<milliseconds<<"ms" <<endl;


}