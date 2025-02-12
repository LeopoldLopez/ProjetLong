#include <stdio.h>
#include <cuda_runtime.h>

int main (){
    int deviceCount; 
    cudaGetDeviceCount(&deviceCount);


    if (deviceCount==0){
        printf("No CUDA-Capable available device\n"); 
        return 0; 
    }
    printf("Number of available CUDA Devices : %d\n", deviceCount); 
    for (int i =0; i<deviceCount; ++i){
        cudaDeviceProp deviceProp; 
        cudaGetDeviceProperties(&deviceProp, i); 
        printf("\nDevice: %d\n", i); 
        printf("\tDevice Name: %s\n", deviceProp.name); 
        printf("\tCompute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("\tTotal Global Memory: %.2f\n", static_cast<float>(deviceProp.totalGlobalMem/(1024*1024*1024)));
        printf("\tNumber of multiprocessor: %d\n", deviceProp.multiProcessorCount); 
        printf("\tMaximum Threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("\tMaximum Threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor); 
    }

    return 0; 

}