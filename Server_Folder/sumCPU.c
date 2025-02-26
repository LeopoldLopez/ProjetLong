#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Function to measure execution time using gettimeofday()
double measureExecutionTimeGettimeofday(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
}

int sumCPU(int argc, char *argv[]){
    int sum = 0;
    for (int i = 2; i < argc; i++) {
        sum += atoi(argv[i]);
    }
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s size num1 num2 [...]\n", argv[0]);
        return -1;
    }
    
    
    struct timeval add_start_tv;
    struct timeval add_end_tv;
    gettimeofday(&add_start_tv, NULL);
    
    sumCPU(argc, argv);
    
    gettimeofday(&add_end_tv, NULL);
    
    
    double sum_time_gettimeofday = measureExecutionTimeGettimeofday(add_start_tv, add_end_tv);
    
    
    printf("Sum_time: %f\n", sum_time_gettimeofday);
    
    return sum_time_gettimeofday;
}
