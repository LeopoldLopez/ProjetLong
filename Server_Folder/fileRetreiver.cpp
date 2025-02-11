#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <fstream>

#define PORT 8080
#define BUFFER_SIZE 1024


void receiveFile(int new_socket, const std::string& filename) {
	char buffer[BUFFER_SIZE];
	std::ofstream file(filename, std::ios::binary);
	if(!file) {
		perror("File creation failed");
		return;
	}

	ssize_t bytes_received;
	while ((bytes_received = read(new_socket, buffer, BUFFER_SIZE)) > 0) {
		file.write(buffer, bytes_received);
	}

	if (bytes_received < 0) {
		perror("Error while receiving the file");
	}

	file.close();
	std::cout << "File received and saved as " << filename << std::endl;
}



int main() {
	int server_fd, new_socket;
	struct sockaddr_in address;
	int addrlen = sizeof(address);
	char buffer[BUFFER_SIZE] = {0};
	std::string message = "Hello from server";

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
	if (listen(server_fd, 3) < 0) {
		perror("Listen failed");
		close(server_fd);
		exit(EXIT_FAILURE);
	}

	std::cout << "Server is listening on port " << PORT << std::endl;

	while(true) {

		// Accept a client connection
		if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
			perror("Accept failed");
			close(server_fd);
			exit(EXIT_FAILURE);
		}

		std::cout << "Client connected." << std::endl;


		// Read from client
		memset(buffer, 0, BUFFER_SIZE);
		ssize_t bytes_read = read(new_socket, buffer, BUFFER_SIZE);
		if (bytes_read > 0) {
			std::string filename(buffer);
			std::cout << "Recieving file: " << filename << std::endl;

			receiveFile(new_socket, filename);
		} else {
			std::cerr << "Failed to read filename" << std::endl;
		}

		// Close the socket
		close(new_socket);
		std::cout << "Client disconnected. Waiting for new connections..." << std::endl;
	}

	close(server_fd);

	return 0;
}









