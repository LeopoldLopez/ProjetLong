#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#define SERVER_IP "127.0.0.1"  // Change to your server's IP
#define SERVER_PORT 5000

int sockfd = -1;

__attribute__((constructor))
void init_client() {
    struct sockaddr_in server_addr;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr);

    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection to CUDA server failed. Using real CUDA.");
        close(sockfd);
        sockfd = -1;
    } else {
        printf("Connected to CUDA server.\n");
    }
}

__attribute__((destructor))
void close_client() {
    if (sockfd >= 0) {
        close(sockfd);
        printf("CUDA Interceptor: Disconnected from server.\n");
    }
}

int send_to_server(const char *func_name, const char *args, char *response, int resp_size) {
    if (sockfd < 0) return -1;  // Server is not available, fallback to real CUDA

    char buffer[512];
    snprintf(buffer, sizeof(buffer), "%s %s\n", func_name, args);

    if (send(sockfd, buffer, strlen(buffer), 0) < 0) {
        perror("Send failed");
        return -1;
    }

    if (recv(sockfd, response, resp_size, 0) < 0) {
        perror("Receive failed");
        return -1;
    }

    return 0;
}
// Interceptor for cudaDeviceSynchronize
extern cudaError_t cudaDeviceSynchronize(void) {

    char response[128];

    if (send_to_server("cudaDeviceSynchronize", void, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaDeviceSynchronize)(void) = NULL;
    if (!real_cudaDeviceSynchronize) real_cudaDeviceSynchronize = (cudaError_t (*)(void)) dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    return real_cudaDeviceSynchronize(void);
}

// Interceptor for cudaDeviceGetLimit
extern cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {

    char response[128];

    if (send_to_server("cudaDeviceGetLimit", size_t *pValue, enum cudaLimit limit, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaDeviceGetLimit)(size_t *pValue, enum cudaLimit limit) = NULL;
    if (!real_cudaDeviceGetLimit) real_cudaDeviceGetLimit = (cudaError_t (*)(size_t *pValue, enum cudaLimit limit)) dlsym(RTLD_NEXT, "cudaDeviceGetLimit");
    return real_cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
}

// Interceptor for cudaDeviceGetCacheConfig
extern cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) {

    char response[128];

    if (send_to_server("cudaDeviceGetCacheConfig", enum cudaFuncCache *pCacheConfig, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaDeviceGetCacheConfig)(enum cudaFuncCache *pCacheConfig) = NULL;
    if (!real_cudaDeviceGetCacheConfig) real_cudaDeviceGetCacheConfig = (cudaError_t (*)(enum cudaFuncCache *pCacheConfig)) dlsym(RTLD_NEXT, "cudaDeviceGetCacheConfig");
    return real_cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
}

// Interceptor for cudaDeviceGetStreamPriorityRange
extern cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {

    char response[128];

    if (send_to_server("cudaDeviceGetStreamPriorityRange", int *leastPriority, int *greatestPriority, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaDeviceGetStreamPriorityRange)(int *leastPriority, int *greatestPriority) = NULL;
    if (!real_cudaDeviceGetStreamPriorityRange) real_cudaDeviceGetStreamPriorityRange = (cudaError_t (*)(int *leastPriority, int *greatestPriority)) dlsym(RTLD_NEXT, "cudaDeviceGetStreamPriorityRange");
    return real_cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
}

// Interceptor for cudaDeviceGetSharedMemConfig
extern cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig) {

    char response[128];

    if (send_to_server("cudaDeviceGetSharedMemConfig", enum cudaSharedMemConfig *pConfig, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaDeviceGetSharedMemConfig)(enum cudaSharedMemConfig *pConfig) = NULL;
    if (!real_cudaDeviceGetSharedMemConfig) real_cudaDeviceGetSharedMemConfig = (cudaError_t (*)(enum cudaSharedMemConfig *pConfig)) dlsym(RTLD_NEXT, "cudaDeviceGetSharedMemConfig");
    return real_cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
}

// Interceptor for cudaGetLastError
extern cudaError_t cudaGetLastError(void) {

    char response[128];

    if (send_to_server("cudaGetLastError", void, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaGetLastError)(void) = NULL;
    if (!real_cudaGetLastError) real_cudaGetLastError = (cudaError_t (*)(void)) dlsym(RTLD_NEXT, "cudaGetLastError");
    return real_cudaGetLastError(void);
}

// Interceptor for cudaPeekAtLastError
extern cudaError_t cudaPeekAtLastError(void) {

    char response[128];

    if (send_to_server("cudaPeekAtLastError", void, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaPeekAtLastError)(void) = NULL;
    if (!real_cudaPeekAtLastError) real_cudaPeekAtLastError = (cudaError_t (*)(void)) dlsym(RTLD_NEXT, "cudaPeekAtLastError");
    return real_cudaPeekAtLastError(void);
}

// Interceptor for cudaGetDeviceCount
extern cudaError_t cudaGetDeviceCount(int *count) {

    char response[128];

    if (send_to_server("cudaGetDeviceCount", int *count, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaGetDeviceCount)(int *count) = NULL;
    if (!real_cudaGetDeviceCount) real_cudaGetDeviceCount = (cudaError_t (*)(int *count)) dlsym(RTLD_NEXT, "cudaGetDeviceCount");
    return real_cudaGetDeviceCount(int *count);
}

// Interceptor for cudaGetDeviceProperties
extern cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {

    char response[128];

    if (send_to_server("cudaGetDeviceProperties", struct cudaDeviceProp *prop, int device, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaGetDeviceProperties)(struct cudaDeviceProp *prop, int device) = NULL;
    if (!real_cudaGetDeviceProperties) real_cudaGetDeviceProperties = (cudaError_t (*)(struct cudaDeviceProp *prop, int device)) dlsym(RTLD_NEXT, "cudaGetDeviceProperties");
    return real_cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
}

// Interceptor for cudaDeviceGetAttribute
extern cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {

    char response[128];

    if (send_to_server("cudaDeviceGetAttribute", int *value, enum cudaDeviceAttr attr, int device, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaDeviceGetAttribute)(int *value, enum cudaDeviceAttr attr, int device) = NULL;
    if (!real_cudaDeviceGetAttribute) real_cudaDeviceGetAttribute = (cudaError_t (*)(int *value, enum cudaDeviceAttr attr, int device)) dlsym(RTLD_NEXT, "cudaDeviceGetAttribute");
    return real_cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
}

// Interceptor for cudaDeviceGetP2PAttribute
extern cudaError_t cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {

    char response[128];

    if (send_to_server("cudaDeviceGetP2PAttribute", int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaDeviceGetP2PAttribute)(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) = NULL;
    if (!real_cudaDeviceGetP2PAttribute) real_cudaDeviceGetP2PAttribute = (cudaError_t (*)(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice)) dlsym(RTLD_NEXT, "cudaDeviceGetP2PAttribute");
    return real_cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
}

// Interceptor for cudaGetDevice
extern cudaError_t cudaGetDevice(int *device) {

    char response[128];

    if (send_to_server("cudaGetDevice", int *device, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaGetDevice)(int *device) = NULL;
    if (!real_cudaGetDevice) real_cudaGetDevice = (cudaError_t (*)(int *device)) dlsym(RTLD_NEXT, "cudaGetDevice");
    return real_cudaGetDevice(int *device);
}

// Interceptor for cudaStreamCreateWithFlags
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {

    char response[128];

    if (send_to_server("cudaStreamCreateWithFlags", cudaStream_t *pStream, unsigned int flags, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamCreateWithFlags)(cudaStream_t *pStream, unsigned int flags) = NULL;
    if (!real_cudaStreamCreateWithFlags) real_cudaStreamCreateWithFlags = (cudaError_t (*)(cudaStream_t *pStream, unsigned int flags)) dlsym(RTLD_NEXT, "cudaStreamCreateWithFlags");
    return real_cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
}

// Interceptor for cudaStreamCreateWithPriority
extern cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) {

    char response[128];

    if (send_to_server("cudaStreamCreateWithPriority", cudaStream_t *pStream, unsigned int flags, int priority, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamCreateWithPriority)(cudaStream_t *pStream, unsigned int flags, int priority) = NULL;
    if (!real_cudaStreamCreateWithPriority) real_cudaStreamCreateWithPriority = (cudaError_t (*)(cudaStream_t *pStream, unsigned int flags, int priority)) dlsym(RTLD_NEXT, "cudaStreamCreateWithPriority");
    return real_cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority);
}

// Interceptor for cudaStreamGetPriority
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority) {

    char response[128];

    if (send_to_server("cudaStreamGetPriority", cudaStream_t hStream, int *priority, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamGetPriority)(cudaStream_t hStream, int *priority) = NULL;
    if (!real_cudaStreamGetPriority) real_cudaStreamGetPriority = (cudaError_t (*)(cudaStream_t hStream, int *priority)) dlsym(RTLD_NEXT, "cudaStreamGetPriority");
    return real_cudaStreamGetPriority(cudaStream_t hStream, int *priority);
}

// Interceptor for cudaStreamGetFlags
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {

    char response[128];

    if (send_to_server("cudaStreamGetFlags", cudaStream_t hStream, unsigned int *flags, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamGetFlags)(cudaStream_t hStream, unsigned int *flags) = NULL;
    if (!real_cudaStreamGetFlags) real_cudaStreamGetFlags = (cudaError_t (*)(cudaStream_t hStream, unsigned int *flags)) dlsym(RTLD_NEXT, "cudaStreamGetFlags");
    return real_cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);
}

// Interceptor for cudaStreamDestroy
extern cudaError_t cudaStreamDestroy(cudaStream_t stream) {

    char response[128];

    if (send_to_server("cudaStreamDestroy", cudaStream_t stream, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamDestroy)(cudaStream_t stream) = NULL;
    if (!real_cudaStreamDestroy) real_cudaStreamDestroy = (cudaError_t (*)(cudaStream_t stream)) dlsym(RTLD_NEXT, "cudaStreamDestroy");
    return real_cudaStreamDestroy(cudaStream_t stream);
}

// Interceptor for cudaStreamWaitEvent
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {

    char response[128];

    if (send_to_server("cudaStreamWaitEvent", cudaStream_t stream, cudaEvent_t event, unsigned int flags, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamWaitEvent)(cudaStream_t stream, cudaEvent_t event, unsigned int flags) = NULL;
    if (!real_cudaStreamWaitEvent) real_cudaStreamWaitEvent = (cudaError_t (*)(cudaStream_t stream, cudaEvent_t event, unsigned int flags)) dlsym(RTLD_NEXT, "cudaStreamWaitEvent");
    return real_cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
}

// Interceptor for cudaStreamAttachMemAsync
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags __dv(cudaMemAttachSingle)) {

    char response[128];

    if (send_to_server("cudaStreamAttachMemAsync", cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags __dv(cudaMemAttachSingle), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamAttachMemAsync)(cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags __dv(cudaMemAttachSingle)) = NULL;
    if (!real_cudaStreamAttachMemAsync) real_cudaStreamAttachMemAsync = (cudaError_t (*)(cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags __dv(cudaMemAttachSingle))) dlsym(RTLD_NEXT, "cudaStreamAttachMemAsync");
    return real_cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags __dv(cudaMemAttachSingle));
}

// Interceptor for cudaEventCreateWithFlags
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {

    char response[128];

    if (send_to_server("cudaEventCreateWithFlags", cudaEvent_t *event, unsigned int flags, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaEventCreateWithFlags)(cudaEvent_t *event, unsigned int flags) = NULL;
    if (!real_cudaEventCreateWithFlags) real_cudaEventCreateWithFlags = (cudaError_t (*)(cudaEvent_t *event, unsigned int flags)) dlsym(RTLD_NEXT, "cudaEventCreateWithFlags");
    return real_cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
}

// Interceptor for cudaEventRecord
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaEventRecord", cudaEvent_t event, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaEventRecord)(cudaEvent_t event, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaEventRecord) real_cudaEventRecord = (cudaError_t (*)(cudaEvent_t event, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaEventRecord");
    return real_cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0));
}

// Interceptor for cudaEventDestroy
extern cudaError_t cudaEventDestroy(cudaEvent_t event) {

    char response[128];

    if (send_to_server("cudaEventDestroy", cudaEvent_t event, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaEventDestroy)(cudaEvent_t event) = NULL;
    if (!real_cudaEventDestroy) real_cudaEventDestroy = (cudaError_t (*)(cudaEvent_t event)) dlsym(RTLD_NEXT, "cudaEventDestroy");
    return real_cudaEventDestroy(cudaEvent_t event);
}

// Interceptor for cudaFuncGetAttributes
extern cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {

    char response[128];

    if (send_to_server("cudaFuncGetAttributes", struct cudaFuncAttributes *attr, const void *func, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaFuncGetAttributes)(struct cudaFuncAttributes *attr, const void *func) = NULL;
    if (!real_cudaFuncGetAttributes) real_cudaFuncGetAttributes = (cudaError_t (*)(struct cudaFuncAttributes *attr, const void *func)) dlsym(RTLD_NEXT, "cudaFuncGetAttributes");
    return real_cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
}

// Interceptor for cudaFuncSetAttribute
extern cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {

    char response[128];

    if (send_to_server("cudaFuncSetAttribute", const void *func, enum cudaFuncAttribute attr, int value, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaFuncSetAttribute)(const void *func, enum cudaFuncAttribute attr, int value) = NULL;
    if (!real_cudaFuncSetAttribute) real_cudaFuncSetAttribute = (cudaError_t (*)(const void *func, enum cudaFuncAttribute attr, int value)) dlsym(RTLD_NEXT, "cudaFuncSetAttribute");
    return real_cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);
}

// Interceptor for cudaOccupancyMaxActiveBlocksPerMultiprocessor
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {

    char response[128];

    if (send_to_server("cudaOccupancyMaxActiveBlocksPerMultiprocessor", int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaOccupancyMaxActiveBlocksPerMultiprocessor)(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) = NULL;
    if (!real_cudaOccupancyMaxActiveBlocksPerMultiprocessor) real_cudaOccupancyMaxActiveBlocksPerMultiprocessor = (cudaError_t (*)(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize)) dlsym(RTLD_NEXT, "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    return real_cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
}

// Interceptor for cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {

    char response[128];

    if (send_to_server("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) = NULL;
    if (!real_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) real_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = (cudaError_t (*)(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags)) dlsym(RTLD_NEXT, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    return real_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
}

// Interceptor for cudaMallocManaged
extern cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags __dv(cudaMemAttachGlobal)) {

    char response[128];

    if (send_to_server("cudaMallocManaged", void **devPtr, size_t size, unsigned int flags __dv(cudaMemAttachGlobal), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMallocManaged)(void **devPtr, size_t size, unsigned int flags __dv(cudaMemAttachGlobal)) = NULL;
    if (!real_cudaMallocManaged) real_cudaMallocManaged = (cudaError_t (*)(void **devPtr, size_t size, unsigned int flags __dv(cudaMemAttachGlobal))) dlsym(RTLD_NEXT, "cudaMallocManaged");
    return real_cudaMallocManaged(void **devPtr, size_t size, unsigned int flags __dv(cudaMemAttachGlobal));
}

// Interceptor for cudaMalloc
extern cudaError_t cudaMalloc(void **devPtr, size_t size) {

    char response[128];

    if (send_to_server("cudaMalloc", void **devPtr, size_t size, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMalloc)(void **devPtr, size_t size) = NULL;
    if (!real_cudaMalloc) real_cudaMalloc = (cudaError_t (*)(void **devPtr, size_t size)) dlsym(RTLD_NEXT, "cudaMalloc");
    return real_cudaMalloc(void **devPtr, size_t size);
}

// Interceptor for cudaFree
extern cudaError_t cudaFree(void *devPtr) {

    char response[128];

    if (send_to_server("cudaFree", void *devPtr, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaFree)(void *devPtr) = NULL;
    if (!real_cudaFree) real_cudaFree = (cudaError_t (*)(void *devPtr)) dlsym(RTLD_NEXT, "cudaFree");
    return real_cudaFree(void *devPtr);
}

// Interceptor for cudaMemcpy3DAsync
extern cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemcpy3DAsync", const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemcpy3DAsync)(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemcpy3DAsync) real_cudaMemcpy3DAsync = (cudaError_t (*)(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemcpy3DAsync");
    return real_cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0));
}

// Interceptor for cudaMemcpyAsync
extern cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemcpyAsync", void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemcpyAsync)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemcpyAsync) real_cudaMemcpyAsync = (cudaError_t (*)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return real_cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
}

// Interceptor for cudaMemcpy2DAsync
extern cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemcpy2DAsync", void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemcpy2DAsync)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemcpy2DAsync) real_cudaMemcpy2DAsync = (cudaError_t (*)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemcpy2DAsync");
    return real_cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
}

// Interceptor for cudaMemsetAsync
extern cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemsetAsync", void *devPtr, int value, size_t count, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemsetAsync)(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemsetAsync) real_cudaMemsetAsync = (cudaError_t (*)(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemsetAsync");
    return real_cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0));
}

// Interceptor for cudaMemset2DAsync
extern cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemset2DAsync", void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemset2DAsync)(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemset2DAsync) real_cudaMemset2DAsync = (cudaError_t (*)(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemset2DAsync");
    return real_cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0));
}

// Interceptor for cudaMemset3DAsync
extern cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemset3DAsync", struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemset3DAsync)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemset3DAsync) real_cudaMemset3DAsync = (cudaError_t (*)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemset3DAsync");
    return real_cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0));
}

// Interceptor for cudaRuntimeGetVersion
extern cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {

    char response[128];

    if (send_to_server("cudaRuntimeGetVersion", int *runtimeVersion, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaRuntimeGetVersion)(int *runtimeVersion) = NULL;
    if (!real_cudaRuntimeGetVersion) real_cudaRuntimeGetVersion = (cudaError_t (*)(int *runtimeVersion)) dlsym(RTLD_NEXT, "cudaRuntimeGetVersion");
    return real_cudaRuntimeGetVersion(int *runtimeVersion);
}

// Interceptor for cudaMemcpyAsync
extern cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemcpyAsync", void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemcpyAsync)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemcpyAsync) real_cudaMemcpyAsync = (cudaError_t (*)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return real_cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
}

// Interceptor for cudaMemcpy2DAsync
extern cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemcpy2DAsync", void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemcpy2DAsync)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemcpy2DAsync) real_cudaMemcpy2DAsync = (cudaError_t (*)(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemcpy2DAsync");
    return real_cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
}

// Interceptor for cudaMemcpy3DAsync
extern cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemcpy3DAsync", const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemcpy3DAsync)(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemcpy3DAsync) real_cudaMemcpy3DAsync = (cudaError_t (*)(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemcpy3DAsync");
    return real_cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0));
}

// Interceptor for cudaMemsetAsync
extern cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemsetAsync", void *devPtr, int value, size_t count, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemsetAsync)(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemsetAsync) real_cudaMemsetAsync = (cudaError_t (*)(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemsetAsync");
    return real_cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0));
}

// Interceptor for cudaMemset2DAsync
extern cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemset2DAsync", void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemset2DAsync)(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemset2DAsync) real_cudaMemset2DAsync = (cudaError_t (*)(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemset2DAsync");
    return real_cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0));
}

// Interceptor for cudaMemset3DAsync
extern cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaMemset3DAsync", struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaMemset3DAsync)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaMemset3DAsync) real_cudaMemset3DAsync = (cudaError_t (*)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaMemset3DAsync");
    return real_cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0));
}

// Interceptor for cudaStreamGetFlags
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {

    char response[128];

    if (send_to_server("cudaStreamGetFlags", cudaStream_t hStream, unsigned int *flags, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamGetFlags)(cudaStream_t hStream, unsigned int *flags) = NULL;
    if (!real_cudaStreamGetFlags) real_cudaStreamGetFlags = (cudaError_t (*)(cudaStream_t hStream, unsigned int *flags)) dlsym(RTLD_NEXT, "cudaStreamGetFlags");
    return real_cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);
}

// Interceptor for cudaStreamGetPriority
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority) {

    char response[128];

    if (send_to_server("cudaStreamGetPriority", cudaStream_t hStream, int *priority, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamGetPriority)(cudaStream_t hStream, int *priority) = NULL;
    if (!real_cudaStreamGetPriority) real_cudaStreamGetPriority = (cudaError_t (*)(cudaStream_t hStream, int *priority)) dlsym(RTLD_NEXT, "cudaStreamGetPriority");
    return real_cudaStreamGetPriority(cudaStream_t hStream, int *priority);
}

// Interceptor for cudaEventRecord
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0)) {

    char response[128];

    if (send_to_server("cudaEventRecord", cudaEvent_t event, cudaStream_t stream __dv(0), response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaEventRecord)(cudaEvent_t event, cudaStream_t stream __dv(0)) = NULL;
    if (!real_cudaEventRecord) real_cudaEventRecord = (cudaError_t (*)(cudaEvent_t event, cudaStream_t stream __dv(0))) dlsym(RTLD_NEXT, "cudaEventRecord");
    return real_cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0));
}

// Interceptor for cudaStreamWaitEvent
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {

    char response[128];

    if (send_to_server("cudaStreamWaitEvent", cudaStream_t stream, cudaEvent_t event, unsigned int flags, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamWaitEvent)(cudaStream_t stream, cudaEvent_t event, unsigned int flags) = NULL;
    if (!real_cudaStreamWaitEvent) real_cudaStreamWaitEvent = (cudaError_t (*)(cudaStream_t stream, cudaEvent_t event, unsigned int flags)) dlsym(RTLD_NEXT, "cudaStreamWaitEvent");
    return real_cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
}

// Interceptor for cudaStreamAttachMemAsync
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length, unsigned int flags) {

    char response[128];

    if (send_to_server("cudaStreamAttachMemAsync", cudaStream_t stream, void *devPtr, size_t length, unsigned int flags, response, sizeof(response)) == 0) {
        return (cudaError_t)atoi(response); // Assuming server returns an integer status
    }

    // If the server fails, use real CUDA
    static cudaError_t (*real_cudaStreamAttachMemAsync)(cudaStream_t stream, void *devPtr, size_t length, unsigned int flags) = NULL;
    if (!real_cudaStreamAttachMemAsync) real_cudaStreamAttachMemAsync = (cudaError_t (*)(cudaStream_t stream, void *devPtr, size_t length, unsigned int flags)) dlsym(RTLD_NEXT, "cudaStreamAttachMemAsync");
    return real_cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length, unsigned int flags);
}

