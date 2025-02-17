#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>

#define PORT 8080
#define BUFFER_SIZE 1024
#define THREAD_POOL_SIZE 5

pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t queue_cv = PTHREAD_COND_INITIALIZER;
int client_queue[100];
int queue_front = 0, queue_rear = 0;
int server_running = 1;



void handle_client(int client_socket) {
    char buffer[BUFFER_SIZE] = {0};

    
    // Read from client
    ssize_t bytes_read = recv(client_socket, buffer, BUFFER_SIZE, 0);
    if (bytes_read > 0) {
        printf("Client: %s\n", buffer);


	char *args[100];
	int count = 0;
	int result = 0;
	
	// Name of the shell script to execute for viewing GPU details
    char *script_name = "./script.sh";
    args[count++] = script_name;
	
	char *func_name = strtok(buffer, ",");
    args[count++] = func_name;


	char *token = strtok(NULL, ",");
	
	while(token != NULL) {
		args[count++] = atoi(token);
		token = strtok(NULL, ",");
	}
	
	args[count] = NULL;
	
	execvp(script_name, args);

        
    printf("Processed data: %s\n", buffer);

	char message[512] = {0};
	sprintf(message, "%d", result);
	// Send response to client
	send(client_socket, message, strlen(message), 0);
	printf("Message sent\n");
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
