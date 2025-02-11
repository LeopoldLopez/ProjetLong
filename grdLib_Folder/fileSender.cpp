#include <iostream>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[BUFFER_SIZE] = {0};
    std::string filename = "fileSend.txt";  // Specify the file you want to send

    // Create socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation failed");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if (inet_pton(AF_INET, "147.127.113.137", &serv_addr.sin_addr) <= 0) {
        perror("Invalid address or address not supported");
        return -1;
    }

    // Connect to the server
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection failed");
        return -1;
    }

    // Send the filename to the server
    send(sock, filename.c_str(), filename.length(), 0);
    std::cout << "File name sent to server: " << filename << std::endl;
    
    sleep(5);
    // Open the file to send
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        perror("File opening failed");
        close(sock);
        return -1;
    }

    // Read the file content and send it to the server
    while (file.read(buffer, sizeof(buffer))) {
        send(sock, buffer, file.gcount(), 0);
    }

    // Send any remaining bytes if file size isn't a perfect multiple of BUFFER_SIZE
    if (file.gcount() > 0) {
        send(sock, buffer, file.gcount(), 0);
    }

    std::cout << "File sent to server." << std::endl;

    // Close the file and socket
    file.close();
    close(sock);
    return 0;
}
