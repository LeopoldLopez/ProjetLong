#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/time.h>

#define PORT 8080
#define BUFFER_SIZE 1024

// Function to measure execution time using gettimeofday()
double measureExecutionTimeGettimeofday(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
}

int main(int argc, char *argv[]) {
    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[BUFFER_SIZE] = {0};

    // Create socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation error");
        exit(EXIT_FAILURE);
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    
    // Convert IPv4 and IPv6 addresses from text to binary form
    // IP address Jetson : 147.127.113.137
    if (inet_pton(AF_INET, "172.22.220.234", &serv_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        exit(EXIT_FAILURE);
    }

    // Connect to server
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection failed");
        exit(EXIT_FAILURE);
    }

    struct timeval add_start_tv;
    struct timeval add_end_tv;
    gettimeofday(&add_start_tv, NULL);
    
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
    
    gettimeofday(&add_end_tv, NULL);

    close(sock);
    
    double sum_time_gettimeofday = measureExecutionTimeGettimeofday(add_start_tv, add_end_tv);

    printf("%lf\n", sum_time_gettimeofday);
        
    return 0;
}