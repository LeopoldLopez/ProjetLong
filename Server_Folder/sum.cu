#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sumKernel(int *args, int n, int *result) {
    extern __shared__ int sharedData[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Charger uniquement les indices valides
    if (i < n) sharedData[i] = args[i];
    else sharedData[i] = 0; // Assurez-vous que les threads hors limites ne modifient pas sharedData
    
    __syncthreads();

    if (i == 0) {
        for (int y = 1; y < n; y++) {
            sharedData[0] += sharedData[y];
        }
        atomicAdd(result, sharedData[0]);
    }
}



int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s size grid_size block_size num1 num2 [...]\n", argv[0]);
        return 1;
    }
    
    int nbArgs = atoi(argv[1]);
    int gridSize = atoi(argv[2]);
    int blockSize = atoi(argv[3]);
    
    int *h_args = (int *)malloc(nbArgs * sizeof(int));
    int h_result = 0;
    
    int *d_args, *d_result;
    
    for (int i = 0; i < nbArgs; i++) {
        h_args[i] = atoi(argv[i + 1 + 3]);
    }
    
    cudaMalloc((void **)&d_args, nbArgs * sizeof(int));
    cudaMalloc((void **)&d_result, sizeof(int));
    cudaMemcpy(d_args, h_args, nbArgs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);
    
    sumKernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_args, nbArgs, d_result);
    
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Sum: %d\n", h_result);
    
    cudaFree(d_args);
    cudaFree(d_result);
    free(h_args);
    return 0;
}

