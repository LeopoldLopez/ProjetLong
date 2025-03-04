#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>

#define PORT 8080
#define BUFFER_SIZE 1024


int main(int argc, char *argv[]) {
    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[BUFFER_SIZE] = {0};
    char *message = "Hello from client";

    // Create socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation error");
        exit(EXIT_FAILURE);
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    
    // Convert IPv4 and IPv6 addresses from text to binary form
    // IP address Jetson : 147.127.113.137
    if (inet_pton(AF_INET, "147.127.113.137", &serv_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        exit(EXIT_FAILURE);
    }

    // Connect to server
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection failed");
        exit(EXIT_FAILURE);
    }
    
    buffer[0] = '\0';  // Ensure buffer is empty
    for (int i = 1; i < argc; i++) {  // Start from 1 to skip program name
        strncat(buffer, argv[i], sizeof(buffer) - strlen(buffer) - 2);
        strcat(buffer, ",");  // Add space between arguments
    }
    
    // Send message to server
    send(sock, buffer, sizeof(buffer), 0);
    
    memset(buffer, 0, BUFFER_SIZE);

    // Read response from server
    read(sock, buffer, BUFFER_SIZE);

    //printf("%s", buffer);

    // Close socket
    close(sock);
    
    return 0;
}
