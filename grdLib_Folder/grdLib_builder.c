#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>

#define HEADER_FILE "/usr/include/cuda_runtime_api.h"
#define OUTPUT_FILE "grdLib.c"

// Regex pattern to extract function headers
const char *pattern = "extern[ ]+__host__[ ]+__cudart_builtin__[ ]+([a-zA-Z_0-9*]+)[ ]+CUDARTAPI[ ]+([a-zA-Z_0-9]+)\\(([^;]*)\\);";


void generate_interceptor(FILE *out, const char *return_type, const char *func_name, const char *params) {
    fprintf(out, "// Interceptor for %s\n", func_name);
    fprintf(out, "extern %s %s(%s) {\n", return_type, func_name, params);
    fprintf(out, "\n    char response[128];\n");
    fprintf(out, "\n    if (send_to_server(\"%s\", %s, response, sizeof(response)) == 0) {\n", func_name, params);
    fprintf(out, "        return (%s)atoi(response); // Assuming server returns an integer status\n", return_type);
    fprintf(out, "    }\n\n    // If the server fails, use real CUDA\n");
    fprintf(out, "    static %s (*real_%s)(%s) = NULL;\n", return_type, func_name, params);
    fprintf(out, "    if (!real_%s) real_%s = (%s (*)(%s)) dlsym(RTLD_NEXT, \"%s\");\n", func_name, func_name, return_type, params, func_name);
    fprintf(out, "    return real_%s(%s);\n", func_name, params);
    fprintf(out, "}\n\n");
}

int main() {
    FILE *header = fopen(HEADER_FILE, "r");
    FILE *out = fopen(OUTPUT_FILE, "w");
    if (!header || !out) {
        perror("Error opening file");
        return 1;
    }

    fprintf(out, "#define _GNU_SOURCE\n"
             "#include <cuda_runtime.h>\n"
             "#include <dlfcn.h>\n"
             "#include <stdio.h>\n"
             "#include <stdlib.h>\n"
             "#include <string.h>\n"
             "#include <sys/socket.h>\n"
             "#include <netinet/in.h>\n"
             "#include <arpa/inet.h>\n"
             "#include <unistd.h>\n\n");

    fprintf(out, "#define SERVER_IP \"127.0.0.1\"  // Change to your server's IP\n"
             "#define SERVER_PORT 5000\n\n");

    fprintf(out, "int sockfd = -1;\n\n");

    // Constructor function
    fprintf(out, "__attribute__((constructor))\n"
             "void init_client() {\n"
             "    struct sockaddr_in server_addr;\n\n"
             "    sockfd = socket(AF_INET, SOCK_STREAM, 0);\n"
             "    if (sockfd < 0) {\n"
             "        perror(\"Socket creation failed\");\n"
             "        return;\n"
             "    }\n\n"
             "    server_addr.sin_family = AF_INET;\n"
             "    server_addr.sin_port = htons(SERVER_PORT);\n"
             "    inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr);\n\n"
             "    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {\n"
             "        perror(\"Connection to CUDA server failed. Using real CUDA.\");\n"
             "        close(sockfd);\n"
             "        sockfd = -1;\n"
             "    } else {\n"
             "        printf(\"Connected to CUDA server.\\n\");\n"
             "    }\n"
             "}\n\n");

    // Destructor function
    fprintf(out, "__attribute__((destructor))\n"
             "void close_client() {\n"
             "    if (sockfd >= 0) {\n"
             "        close(sockfd);\n"
             "        printf(\"CUDA Interceptor: Disconnected from server.\\n\");\n"
             "    }\n"
             "}\n\n");

    // send_to_server function
    fprintf(out, "int send_to_server(const char *func_name, const char *args, char *response, int resp_size) {\n"
             "    if (sockfd < 0) return -1;  // Server is not available, fallback to real CUDA\n\n"
             "    char buffer[512];\n"
             "    snprintf(buffer, sizeof(buffer), \"%%s %%s\\n\", func_name, args);\n\n"
             "    if (send(sockfd, buffer, strlen(buffer), 0) < 0) {\n"
             "        perror(\"Send failed\");\n"
             "        return -1;\n"
             "    }\n\n"
             "    if (recv(sockfd, response, resp_size, 0) < 0) {\n"
             "        perror(\"Receive failed\");\n"
             "        return -1;\n"
             "    }\n\n"
             "    return 0;\n"
             "}\n");

    char line[1024];
    regex_t regex;
    regmatch_t matches[4];

    if (regcomp(&regex, pattern, REG_EXTENDED)) {
        fprintf(stderr, "Could not compile regex\n");
        return 1;
    }

    while (fgets(line, sizeof(line), header)) {
        if (regexec(&regex, line, 4, matches, 0) == 0) {
            char return_type[128], func_name[128], params[512];

            strncpy(return_type, line + matches[1].rm_so, matches[1].rm_eo - matches[1].rm_so);
            return_type[matches[1].rm_eo - matches[1].rm_so] = '\0';

            strncpy(func_name, line + matches[2].rm_so, matches[2].rm_eo - matches[2].rm_so);
            func_name[matches[2].rm_eo - matches[2].rm_so] = '\0';

            strncpy(params, line + matches[3].rm_so, matches[3].rm_eo - matches[3].rm_so);
            params[matches[3].rm_eo - matches[3].rm_so] = '\0';

            generate_interceptor(out, return_type, func_name, params);
        }
    }

    regfree(&regex);
    fclose(header);
    fclose(out);
    printf("Interceptor generated: %s\n", OUTPUT_FILE);
    return 0;
}

