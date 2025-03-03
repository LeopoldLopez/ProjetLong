#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Function to measure execution time using gettimeofday()
double measureExecutionTimeGettimeofday(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
}

int sumCPU(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Erreur lors de l'ouverture du fichier");
        return -1;
    }

    int sum = 0, num;
    fscanf(file, "%d", &num);
    while (fscanf(file, "%d", &num) != EOF) {
        sum += num;
    }

    fclose(file);
    return sum;
}


int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s grid_size block_size fichier_donnees\n", argv[0]);
        return -1;
    }
    
    
    struct timeval add_start_tv;
    struct timeval add_end_tv;
    gettimeofday(&add_start_tv, NULL);
    
    int result = sumCPU(argv[3]);
    
    gettimeofday(&add_end_tv, NULL);
    
    
    double sum_time_gettimeofday = measureExecutionTimeGettimeofday(add_start_tv, add_end_tv);
    
    printf("%lf\n", sum_time_gettimeofday);
    
    return sum_time_gettimeofday;
}
