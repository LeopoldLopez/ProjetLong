#!/bin/bash

gcc ./server_test.c -pthread -o server_test
gcc ./sumCPU.c -o sumCPU
nvcc ./sum.cu -o sum
