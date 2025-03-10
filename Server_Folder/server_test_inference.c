#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>

#define PORT 8080
#define BUFFER_SIZE 8192
#define THREAD_POOL_SIZE 5

pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t queue_cv = PTHREAD_COND_INITIALIZER;
int client_queue[1000];
int queue_front = 0, queue_rear = 0;
int server_running = 1;


// Fonction pour exécuter un script et récupérer sa sortie
void execute_script(const char *script, char *output, int output_size) {

    printf("Executing command: %s\n", script);
    FILE *fp = popen(script, "r");
    if (fp == NULL) {

        snprintf(output, output_size, "Erreur lors de l'exécution du script.");
        return;
    }


    output[BUFFER_SIZE] = '\0';
    char temp[256];
    while (fgets(temp, sizeof(temp), fp) != NULL) {
        strncat(output, temp, output_size - strlen(output) - 1);
    }
    pclose(fp);
}



void handle_client(int client_socket) {
    char buffer[BUFFER_SIZE] = {0};

    // Lire la requête du client
    ssize_t bytes_read = recv(client_socket, buffer, BUFFER_SIZE, 0);
    if (bytes_read <= 0) {
        close(client_socket);
        return;
    }

    printf("Client sent: %s\n", buffer);

    char result[BUFFER_SIZE] = {0};

    execute_script(buffer, result, sizeof(result));

    send(client_socket, result, strlen(result), 0);

    close(client_socket);
    printf("Client déconnecté.\n");

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

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Erreur socket");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Erreur bind");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 1000) < 0) {
        perror("Erreur listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    printf("Serveur en écoute sur le port %d\n", PORT);

    pthread_t thread_pool[THREAD_POOL_SIZE];
    for (int i = 0; i < THREAD_POOL_SIZE; ++i) {
        pthread_create(&thread_pool[i], NULL, worker_thread, NULL);
    }

    while (server_running) {
        int new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
        if (new_socket < 0) {
            perror("Erreur accept");
            continue;
        }

        printf("Client connecté.\n");



        pthread_mutex_lock(&queue_mutex);
        client_queue[queue_rear] = new_socket;
        queue_rear = (queue_rear + 1) % 100;
        pthread_mutex_unlock(&queue_mutex);
        pthread_cond_signal(&queue_cv);
    }

    server_running = 0;
    pthread_cond_broadcast(&queue_cv);
    for (int i = 0; i < THREAD_POOL_SIZE; ++i) {
        pthread_join(thread_pool[i], NULL);
    }
    close(server_fd);
    return 0;
}
