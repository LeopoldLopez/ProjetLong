#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <cuda_runtime.h>

#define PORT 8080
#define BUFFER_SIZE 1024
#define THREAD_POOL_SIZE 5

pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t queue_cv = PTHREAD_COND_INITIALIZER;
int client_queue[100];
int queue_front = 0, queue_rear = 0;
int server_running = 1;


__device__ int strcmp_dev(const char *s1, const char *s2) {
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    return *(const unsigned char*)s1 - *(const unsigned char*)s2;
}


__global__ void process_func(char *func_name, int *args, int count, int *result) {
    int idx = threadIdx.x;  // Unique thread index

    if (idx >= count) return;  // Prevent out-of-bounds access

	
    __shared__ int sum;
    
    if (idx == 0) sum = 0;

    __syncthreads();

    
    __syncthreads(); 

    if (!strcmp_dev(func_name, "sum")) {


        atomicAdd(&sum, args[idx]); // Atomic add to prevent race conditions

       

    }

     __syncthreads(); 


    if (idx == 0) *result = sum;


}





void handle_client(int client_socket) {
    char buffer[BUFFER_SIZE] = {0};

    
    // Read from client
    ssize_t bytes_read = recv(client_socket, buffer, BUFFER_SIZE, 0);
    if (bytes_read > 0) {
        printf("Client: %s\n", buffer);
        
        // Allocate memory on GPU
        //char *d_buffer;
        //cudaMalloc((void**)&d_buffer, BUFFER_SIZE);
        //cudaMemcpy(d_buffer, buffer, BUFFER_SIZE, cudaMemcpyHostToDevice);

        // Process data on GPU
        //int blockSize = 256;
        //int numBlocks = (BUFFER_SIZE + blockSize - 1) / blockSize;
        //process_data_on_gpu<<<numBlocks, blockSize>>>(d_buffer, bytes_read);
        //cudaDeviceSynchronize();

        // Copy processed data back to host
        //cudaMemcpy(buffer, d_buffer, BUFFER_SIZE, cudaMemcpyDeviceToHost);
        //cudaFree(d_buffer);


	int args[100];
	int count = 0;
	int result = 0;
	
	char *func_name = strtok(buffer, ",");

	char *token = strtok(NULL, ",");
	
	while(token != NULL) {
		args[count++] = atoi(token);
		token = strtok(NULL, ",");
	}

	char *d_func_name;
    	int *d_args, *d_result;

    	cudaMalloc((void **)&d_func_name, sizeof(func_name));
    	cudaMalloc((void **)&d_args, count * sizeof(int));
    	cudaMalloc((void **)&d_result, sizeof(int));

	cudaMemcpy(d_func_name, func_name, sizeof(func_name), cudaMemcpyHostToDevice);
	cudaMemcpy(d_args, args, count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

	process_func<<<1, count>>>(d_func_name, d_args, count, d_result);

	cudaDeviceSynchronize(); // Ensure all kernel calls complete before exit

	
	cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        
        printf("Processed data: %s\n", buffer);

	char message[512] = {0};
	sprintf(message, "%d", result);
	// Send response to client
	send(client_socket, message, strlen(message), 0);
	printf("Message sent\n");


	cudaFree(d_func_name);
	cudaFree(d_args);
	cudaFree(d_result);

    }
    
    
    // Close client socket
    close(client_socket);
    printf("Client disconnected.\n");
}

void* worker_thread(void* arg) {
    while (server_running) {
        int client_socket;
        pthread_mutex_lock(&queue_mutex);
        while (queue_front == queue_rear && server_running) {
            pthread_cond_wait(&queue_cv, &queue_mutex);
        }
        if (!server_running) {
            pthread_mutex_unlock(&queue_mutex);
            return NULL;
        }
        client_socket = client_queue[queue_front];
        queue_front = (queue_front + 1) % 100;
        pthread_mutex_unlock(&queue_mutex);
        handle_client(client_socket);
    }
    return NULL;
}

int main() {
    int server_fd;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    // Create socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    // Bind socket to port
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // Listen for connections
    if (listen(server_fd, 10) < 0) { // Allow up to 10 pending connections
        perror("Listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    printf("Server is listening on port %d\n", PORT);

    // Create thread pool
    pthread_t thread_pool[THREAD_POOL_SIZE];
    for (int i = 0; i < THREAD_POOL_SIZE; ++i) {
        pthread_create(&thread_pool[i], NULL, worker_thread, NULL);
    }

    while (server_running) {
        int new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
        if (new_socket < 0) {
            perror("Accept failed");
            continue;
        }

        printf("Client connected.\n");

        pthread_mutex_lock(&queue_mutex);
        client_queue[queue_rear] = new_socket;
        queue_rear = (queue_rear + 1) % 100;
        pthread_mutex_unlock(&queue_mutex);
        pthread_cond_signal(&queue_cv);
    }

    // Cleanup
    server_running = 0;
    pthread_cond_broadcast(&queue_cv);
    for (int i = 0; i < THREAD_POOL_SIZE; ++i) {
        pthread_join(thread_pool[i], NULL);
    }
    close(server_fd);
    return 0;
}
