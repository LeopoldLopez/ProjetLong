#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Function to measure execution time using gettimeofday()
double measureExecutionTimeGettimeofday(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
}

__global__ void sumKernel(int *args, int n, int *blockSums) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Charger les données en mémoire partagée
    sharedData[tid] = (globalIdx < n) ? args[globalIdx] : 0;
    __syncthreads();

    if (tid == 0) 
        for (int y = 1; y < n; y++) 
            sharedData[tid] += sharedData[y];

    // Seul le thread 0 stocke le résultat du bloc
    if (tid == 0) {
        blockSums[blockIdx.x] = sharedData[0];
    }
}

// Kernel pour additionner les sommes des blocs
__global__ void finalSumKernel(int *blockSums, int numBlocks, int *result) {

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (globalIdx == 0) {
        for (int i = 1; i < numBlocks; i++)
            blockSums[0] += blockSums[i];
        *result = blockSums[0];
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
    
    int *d_args, *d_blockSums, *d_result;
    
    for (int i = 0; i < nbArgs; i++) {
        h_args[i] = atoi(argv[i + 4]);
    }
    
    cudaMalloc((void **)&d_args, nbArgs * sizeof(int));
    cudaMalloc((void **)&d_blockSums, gridSize * sizeof(int)); // Stockage des sommes partielles
    cudaMalloc((void **)&d_result, sizeof(int));
    
    cudaMemcpy(d_args, h_args, nbArgs * sizeof(int), cudaMemcpyHostToDevice);
    
    
    struct timeval add_start_tv;
    struct timeval add_end_tv;
    gettimeofday(&add_start_tv, NULL);
    //Somme partielle dans chaque bloc
    sumKernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_args, nbArgs, d_blockSums);
    
    //Somme globale des blocs
    finalSumKernel<<<1, gridSize, gridSize * sizeof(int)>>>(d_blockSums, gridSize, d_result);
    
    gettimeofday(&add_end_tv, NULL);
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Sum: %d\n", h_result);
    
    cudaFree(d_args);
    cudaFree(d_blockSums);
    cudaFree(d_result);
    free(h_args);
    
    
    double sum_time_gettimeofday = measureExecutionTimeGettimeofday(add_start_tv, add_end_tv);
    
    
    printf("Sum_time: %f\n", sum_time_gettimeofday);
    
    return sum_time_gettimeofday;
}
