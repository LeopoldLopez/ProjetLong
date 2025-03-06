#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>

#define CMD_TEMPLATE "./client sum %d 2>&1 | tee output%d.txt"

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
    snprintf(command, sizeof(command), CMD_TEMPLATE, data->size, data->index);

    printf("Executing: %s\n", command);

    system(command);  // Execute the command


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
