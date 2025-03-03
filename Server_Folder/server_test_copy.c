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

void initializeArrayToFile(const char *filename, int size) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Erreur lors de l'ouverture du fichier");
        return;
    }

    //nombre d'arguments en première ligne
    fprintf(file, "%d\n", size);

    for (int i = 0; i < size; i++) {
        fprintf(file, "9 ");
    }
    
    fclose(file);
}

// Fonction pour exécuter un script et récupérer sa sortie
void execute_script(const char *script, const char *args_filename, char *output, int output_size, int count) {

    if (!args_filename) {
        snprintf(output, output_size, "Erreur: Aucun fichier d'arguments fourni.");
        return;
    }

    // Choisir block_size optimal (puissance de 2)
    int block_size = 1;
    while (block_size * 2 <= count && block_size * 2 <= 1024) {
        block_size *= 2;
    }

    // Calculer grid_size
    int grid_size = (count + block_size - 1) / block_size;

    // Construire les arguments pour la fonction
    size_t command_size = (2 * (count + 3) + strlen(script) + 4 + BUFFER_SIZE) * sizeof(char);
    char* command = (char *)malloc(command_size); //On compte aussi le nom de la fonction, les espaces entre les arguments (2*) et les méta-paramètres(+3)
    snprintf(command, command_size, "%s %d %d %s", script, grid_size, block_size, args_filename);
    
    FILE *fp = popen(command, "r");
    if (fp == NULL) {

        snprintf(output, output_size, "Erreur lors de l'exécution du script.");
        return;
    }


    output[0] = '\0';
    char temp[256];
    while (fgets(temp, sizeof(temp), fp) != NULL) {
        strncat(output, temp, output_size - strlen(output) - 1);
    }
    free(command);
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

    char *func_name = strtok(buffer, ",");
    char *args = strtok(NULL, ",");
    int arg = 0;

    if (args != NULL) {
        arg = atoi(args);
    }

    char result[BUFFER_SIZE] = {0};

    char commande[strlen(func_name) + 3];
    snprintf(commande, sizeof(commande), "./%s", func_name);

    char args_filename[] = "args.txt";
    initializeArrayToFile(args_filename, arg);

    execute_script(commande, args_filename, result, sizeof(result), arg);

    send(client_socket, result, strlen(result), 0);

    close(client_socket);
    printf("Client déconnecté.\n");

    remove(args_filename);
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

    if (listen(server_fd, 10) < 0) {
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
