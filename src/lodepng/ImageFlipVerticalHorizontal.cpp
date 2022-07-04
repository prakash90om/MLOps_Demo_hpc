/////////////////////////////////////////////////////////
//  CPP Code to Flip an Image Vertically
//  Load the image in InputImage.png
/////////////////////////////////////////////////////////

#include "../../include/lodepng.cpp"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <algorithm>
#include<vector>


using namespace std;
using namespace std::chrono;

int BLOCKDIM;
int GRIDDIM;

// Driver code
int main()
{
    std::vector<unsigned char> in_image;
    unsigned int width, height;
    const char* input_file = "../../res/InputImage.png";
    const char* output_file = "../../res/CPUVerticalFlip.png";
    const char* output_file_hori = "../../res/CPUHorizontalFlip.png";


    
    // Load the data
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;


    // Prepare the data
    //unsigned char* input_image = new unsigned char[(in_image.size()*3)/4];
    //unsigned char* output_image = new unsigned char[(in_image.size()*3)/4];
    unsigned char* input_image = new unsigned char[(in_image.size())];
    unsigned char* output_image = new unsigned char[(in_image.size())];
    unsigned char* output_image_hori = new unsigned char[(in_image.size())];


    //cout << "Image Size : " << in_image.size() <<endl;
    
    auto start = high_resolution_clock::now();

/////////////////////////////////
//      Vertical Flip
/////////////////////////////////

    //Computing time to flip image alone
    int where = 0;
    for(int i = 0; i < in_image.size(); ++i) {
           output_image[where] = in_image.at(in_image.size() - 1 - i);
           where++;
    }

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken to Vertical flip the image : " << duration.count() <<"us" << endl;

    
/*    for(int i = in_image.size()-1; i >=0 ; --i) {
           output_image[where - i] = input_image[i];
           //cout <<  output_image[where];
    } */

    error = lodepng::encode(output_file, output_image, width, height);
        
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
 
/////////////////////////////////
//      Horizontal Flip
/////////////////////////////////

    start = high_resolution_clock::now();

    where = 0;

    for (int i=0; i < in_image.size(); i++)
    {
        input_image[where] = in_image.at(i);
        where++;
    }

    where = 0;
    for(int i = 0; i < height; ++i) 
    {
        for (int j = 0; j < (in_image.size()/height); j++)
        {
           output_image_hori[where] =input_image[(i+1)*(in_image.size()/height) - 1 - j];
           //output_image_hori[where + 1] =input_image[(i+1)*(in_image.size()/height) - 2 - j];
           //output_image_hori[where + 1] =input_image[(i+1)*(in_image.size()/height) - 3 - j];

           where++;
        }
        //where += (in_image.size()/height)/2;
    }

    stop = high_resolution_clock::now();

    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken to Horizontal flip the image : " << duration.count() <<"us" << endl;

    
/*    for(int i = in_image.size()-1; i >=0 ; --i) {
           output_image[where - i] = input_image[i];
           //cout <<  output_image[where];
    } */

    error = lodepng::encode(output_file_hori, output_image_hori, width, height);
        
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

}
