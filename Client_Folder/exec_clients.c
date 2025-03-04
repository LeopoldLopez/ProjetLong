#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>

#define CMD_TEMPLATE "./client sum %d"

typedef struct {
    int index;
    int size;
} ThreadData;


// Function to measure execution time using gettimeofday()
double measureExecutionTimeGettimeofday(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
}

void* run_client(void* arg) {
    ThreadData* data = (ThreadData*) arg;
    char command[256];

    // Format the command string
    snprintf(command, sizeof(command), CMD_TEMPLATE, data->size);

    printf("Executing: %s\n", command);

    struct timeval add_start_tv;
    struct timeval add_end_tv;
    gettimeofday(&add_start_tv, NULL);
    system(command);  // Execute the command
    
    gettimeofday(&add_end_tv, NULL);
    double sum_time_gettimeofday = measureExecutionTimeGettimeofday(add_start_tv, add_end_tv);
    
    char filename[256];
    snprintf(filename, sizeof(filename), "output%d.txt", data->index);  // Using data->index for file name

    FILE* file = fopen(filename, "w");
    if (file) {
        // Write the execution time to the file
        fprintf(file, "%lf\n", sum_time_gettimeofday);
        fclose(file);
    } else {
        perror("Failed to open output file");
    }

    free(data);  // Free allocated memory
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <nombre_de_requetes>\n", argv[0]);
        return 1;
    }

    int num_requests = atoi(argv[1]);
    pthread_t threads[num_requests];

    // Launch threads
    for (int i = 0; i < num_requests; i++) {
        ThreadData* data = malloc(sizeof(ThreadData));
        if (!data) {
            perror("Failed to allocate memory");
            return 1;
        }
        data->index = i + 1;
        data->size = (i + 1) * 1000;

        if (pthread_create(&threads[i], NULL, run_client, data) != 0) {
            perror("Failed to create thread");
            return 1;
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_requests; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All client requests completed.\n");
    return 0;
}
