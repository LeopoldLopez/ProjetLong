#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>



int main(int argc, char *argv[]) {
    if (argc < 1) {
        printf("Usage: %s num1 num2 [...]\n", argv[0]);
        return -1;
    }
    int sum = 0;
    
    for (int i = 1; i < argc; i++) {
        sum += atoi(argv[i]);
    }
    printf("Sum: %d\n", sum);
    
    return sum;
}
