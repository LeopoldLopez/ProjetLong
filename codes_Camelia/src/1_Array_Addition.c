#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// Function to initialize the input arrays with random integers
void initializeArray(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100; // Generate random integers between 0 and 99
    }
}

// Function to perform sequential array addition
void addArrays(int *A, int *B, int *C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

// Function to measure execution time using clock()
double measureExecutionTime(clock_t start, clock_t end) {
    return (double)(end - start) / CLOCKS_PER_SEC;
}

// Function to measure execution time using gettimeofday()
double measureExecutionTimeGettimeofday(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int *A = (int *)malloc(size * sizeof(int));
    int *B = (int *)malloc(size * sizeof(int));
    int *C = (int *)malloc(size * sizeof(int));

    // Measure initialization time
    clock_t init_start = clock();
    struct timeval init_start_tv;
    gettimeofday(&init_start_tv, NULL);
    initializeArray(A, size);
    initializeArray(B, size);
    clock_t init_end = clock();
    struct timeval init_end_tv;
    gettimeofday(&init_end_tv, NULL);
    double init_time_clock = measureExecutionTime(init_start, init_end);
    double init_time_gettimeofday = measureExecutionTimeGettimeofday(init_start_tv, init_end_tv);

    // Measure addition time
    clock_t add_start = clock();
    struct timeval add_start_tv;
    gettimeofday(&add_start_tv, NULL);
    addArrays(A, B, C, size);
    clock_t add_end = clock();
    struct timeval add_end_tv;
    gettimeofday(&add_end_tv, NULL);
    double add_time_clock = measureExecutionTime(add_start, add_end);
    double add_time_gettimeofday = measureExecutionTimeGettimeofday(add_start_tv, add_end_tv);

    // Print results
    printf("Initialization Time (clock): %f seconds\n", init_time_clock);
    printf("Initialization Time (gettimeofday): %f seconds\n", init_time_gettimeofday);
    printf("Addition Time (clock): %f seconds\n", add_time_clock);
    printf("Addition Time (gettimeofday): %f seconds\n", add_time_gettimeofday);

    free(A);
    free(B);
    free(C);

    return 0;
}

